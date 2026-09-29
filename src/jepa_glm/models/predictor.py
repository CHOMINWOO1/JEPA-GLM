from __future__ import annotations

import math

import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, hidden_size: int, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("Predictor layers must be >= 1")
        blocks = []
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size)])
        blocks.append(nn.Linear(hidden_size, hidden_size))
        self.net = nn.Sequential(*blocks)

    def forward(self, embedding):
        return self.net(embedding)


class AttentionPredictor(nn.Module):
    """Predict a global target latent from a re-masked sequence of context latents."""

    def __init__(
        self,
        hidden_size: int,
        *,
        embed_dim: int = 384,
        layers: int = 4,
        heads: int = 2,
        max_length: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("Attention predictor layers must be >= 1")
        if embed_dim < 1 or heads < 1 or embed_dim % heads != 0:
            raise ValueError("Attention predictor embed_dim must be positive and divisible by heads")
        if max_length < 1:
            raise ValueError("Attention predictor max_length must be positive")
        self.input_projection = nn.Linear(hidden_size, embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_encoding(max_length, embed_dim),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("Attention predictor hidden input must have shape [batch, length, hidden]")
        if hidden.shape[1] > self.positional_encoding.shape[1]:
            raise ValueError("Attention predictor input exceeds configured max_length")
        encoded = self.input_projection(hidden)
        if target_mask is not None:
            if target_mask.shape != hidden.shape[:2]:
                raise ValueError("Attention predictor target_mask shape mismatch")
            encoded = torch.where(target_mask.bool().unsqueeze(-1), self.mask_token, encoded)
        encoded = encoded + self.positional_encoding[:, : hidden.shape[1]].to(encoded.dtype)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        predicted = self.encoder(encoded, src_key_padding_mask=padding_mask)
        return self.output_projection(self.output_norm(predicted[:, 0]))


def _sinusoidal_encoding(max_length: int, embed_dim: int) -> torch.Tensor:
    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / embed_dim)
    )
    encoding = torch.zeros(1, max_length, embed_dim, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(positions * frequencies)
    if embed_dim > 1:
        encoding[0, :, 1::2] = torch.cos(positions * frequencies[: encoding[0, :, 1::2].shape[-1]])
    return encoding
