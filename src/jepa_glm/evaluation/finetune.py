from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from jepa_glm.models.pooling import pool_hidden


@dataclass(frozen=True)
class FineTuneSummary:
    epochs: int
    optimizer_steps: int
    epoch_losses: list[float]
    trainable_encoder_parameters: int
    trainable_classifier_parameters: int
    mixed_precision: bool


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        context_encoder: nn.Module,
        hidden_size: int,
        pooling: str = "cls",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.pooling = pooling
        pooled_size = hidden_size * 2 if pooling == "mean_max" else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pooled_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.context_encoder(input_ids, attention_mask)
        pooled = pool_hidden(hidden, attention_mask, self.pooling)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def train_sequence_classifier(
    model: SequenceClassifier,
    dataloader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    encoder_learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    mixed_precision: bool = True,
) -> FineTuneSummary:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    encoder_parameters = [parameter for parameter in model.context_encoder.parameters() if parameter.requires_grad]
    classifier_parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    parameter_groups: list[dict[str, object]] = []
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_learning_rate, "name": "encoder"})
    parameter_groups.append({"params": classifier_parameters, "lr": classifier_learning_rate, "name": "classifier"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.BCEWithLogitsLoss()
    model.to(device)
    optimizer_steps = 0
    epoch_losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        example_count = 0
        batch_count = len(dataloader)
        for batch_index, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            attention_mask = input_ids.ne(0)
            accumulation_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
            accumulation_size = min(gradient_accumulation_steps, batch_count - accumulation_start)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                batch_loss = loss_function(logits, labels)
                loss = batch_loss / accumulation_size
            scaler.scale(loss).backward()
            batch_examples = int(labels.shape[0])
            loss_sum += float(batch_loss.detach().item()) * batch_examples
            example_count += batch_examples
            accumulation_end = min(accumulation_start + gradient_accumulation_steps, batch_count)
            if batch_index + 1 == accumulation_end:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        epoch_losses.append(loss_sum / max(1, example_count))
    return FineTuneSummary(
        epochs=epochs,
        optimizer_steps=optimizer_steps,
        epoch_losses=epoch_losses,
        trainable_encoder_parameters=sum(parameter.numel() for parameter in encoder_parameters),
        trainable_classifier_parameters=sum(parameter.numel() for parameter in classifier_parameters),
        mixed_precision=use_amp,
    )


@torch.no_grad()
def sequence_classifier_scores(
    model: SequenceClassifier,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    labels: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    for input_ids, batch_labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = input_ids.ne(0)
        logits = model(input_ids, attention_mask)
        labels.append(batch_labels.cpu())
        scores.append(torch.sigmoid(logits).cpu())
    if not labels:
        raise ValueError("Cannot evaluate an empty dataloader")
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def write_sequence_classifier_scores(
    path: str | Path,
    sequences: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
) -> str:
    if not (len(sequences) == labels.shape[0] == scores.shape[0]):
        raise ValueError("sequences, labels, and scores must have the same length")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "sequence_sha256", "label", "score"])
        writer.writeheader()
        for index, (sequence, label, score) in enumerate(zip(sequences, labels, scores, strict=True)):
            writer.writerow(
                {
                    "row_index": index,
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "label": int(label),
                    "score": format(float(score), ".17g"),
                }
            )
    return _sha256_file(output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
