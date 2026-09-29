from __future__ import annotations

from pathlib import Path
import warnings

import torch
from torch import nn


class TinyDnaBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_length: int = 1024) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        return self.norm(self.encoder(hidden, src_key_padding_mask=key_padding_mask))


class HuggingFaceBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_policy: str = "none",
        trainable_last_n_layers: int = 0,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        local_files_only: bool | None = None,
        load_pretrained_mlm_head: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install the 'hf' extra to use Hugging Face backbones.") from exc
        if local_files_only is None:
            local_files_only = Path(model_name).exists()
        if load_pretrained_mlm_head:
            try:
                from transformers import AutoModelForMaskedLM
            except ImportError as exc:
                raise ImportError("The installed Transformers package does not expose AutoModelForMaskedLM.") from exc
            masked_model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            self.model = masked_model.base_model
            mlm_head = getattr(masked_model, "cls", None)
            if not isinstance(mlm_head, nn.Module):
                raise TypeError("The configured masked-language model does not expose a reusable 'cls' MLM head")
            self.pretrained_mlm_head = mlm_head
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        self.hidden_size = int(self.model.config.hidden_size)
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except ValueError as exc:
                warnings.warn(
                    f"Gradient checkpointing was requested but is not supported by this backbone; continuing without it: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        apply_freeze_policy(self.model, freeze_policy, trainable_last_n_layers)

    def take_pretrained_mlm_head(self) -> nn.Module | None:
        return self._modules.pop("pretrained_mlm_head", None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError(f"Unsupported Hugging Face backbone output type: {type(output)!r}")


def build_backbone(config: dict) -> nn.Module:
    name = config.get("backbone_name")
    if name:
        model_source = str(config.get("local_model_dir") or name)
        freeze_policy = str(config.get("freeze_policy", "all" if bool(config.get("freeze_backbone", False)) else "none"))
        return HuggingFaceBackbone(
            model_source,
            freeze_policy=freeze_policy,
            trainable_last_n_layers=int(config.get("trainable_last_n_layers", 0)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            local_files_only=bool(config.get("local_files_only", Path(model_source).exists())),
            load_pretrained_mlm_head=bool(config.get("use_pretrained_mlm_head", False)),
        )
    return TinyDnaBackbone(
        vocab_size=int(config.get("vocab_size", 8)),
        hidden_size=int(config.get("hidden_size", 128)),
        max_length=int(config.get("max_length", 1024)),
    )


def apply_freeze_policy(model: nn.Module, freeze_policy: str, trainable_last_n_layers: int = 0) -> None:
    policy = freeze_policy.lower()
    if policy in {"none", "full_unfreeze", "unfrozen"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if policy in {"all", "frozen", "freeze"}:
        for parameter in model.parameters():
            parameter.requires_grad = False
        return
    if policy in {"embeddings", "freeze_embeddings"}:
        for name, parameter in model.named_parameters():
            if "emb" in name.lower():
                parameter.requires_grad = False
        return
    if policy in {"encoder_last_n", "last_n"}:
        for parameter in model.parameters():
            parameter.requires_grad = False
        layers = _find_encoder_layers(model)
        if not layers:
            raise ValueError("Could not find encoder layers for encoder_last_n freeze policy")
        for layer in layers[-max(1, trainable_last_n_layers) :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        return
    raise ValueError(f"Unknown freeze_policy: {freeze_policy}")


def _find_encoder_layers(model: nn.Module) -> list[nn.Module]:
    for path in (
        "encoder.layer",
        "bert.encoder.layer",
        "model.encoder.layer",
        "backbone.encoder.layer",
    ):
        current: object = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, (nn.ModuleList, list, tuple)):
            return list(current)
    return []
