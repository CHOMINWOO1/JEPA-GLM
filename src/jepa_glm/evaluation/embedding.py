from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jepa_glm.data.tokenizer import SequenceTokenizer
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.models.pooling import pool_hidden


def load_context_encoder_state(model: JepaGlmModel, checkpoint_model_state: dict[str, torch.Tensor]) -> int:
    """Load only shape-compatible context encoder tensors for representation evaluation."""

    return load_context_encoder_module_state(model.context_encoder, checkpoint_model_state)


def load_context_encoder_module_state(
    context_encoder: nn.Module,
    checkpoint_model_state: dict[str, torch.Tensor],
) -> int:
    """Load a pretraining checkpoint into a standalone context encoder."""

    current = context_encoder.state_dict()
    selected = {
        name.removeprefix("context_encoder."): tensor
        for name, tensor in checkpoint_model_state.items()
        if name.startswith("context_encoder.")
        and name.removeprefix("context_encoder.") in current
        and current[name.removeprefix("context_encoder.")].shape == tensor.shape
    }
    if not selected:
        raise ValueError("Checkpoint contains no shape-compatible context encoder tensors")
    context_encoder.load_state_dict(selected, strict=False)
    return len(selected)


class SequenceDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, sequences: list[str], labels: list[int], tokenizer: SequenceTokenizer, max_length: int) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        self.input_ids = [tokenizer.encode(sequence, max_length=max_length) for sequence in sequences]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.input_ids[index], int(self.labels[index])


@torch.no_grad()
def extract_sequence_embeddings(
    model: JepaGlmModel,
    sequences: list[str],
    tokenizer: SequenceTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    labels = [0] * len(sequences)
    dataset = SequenceDataset(sequences, labels, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    embeddings: list[torch.Tensor] = []
    for input_ids, _ in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        hidden = model.context_encoder(input_ids, attention_mask)
        embeddings.append(pool_hidden(hidden, attention_mask, model.pooling).cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)
