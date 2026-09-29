from __future__ import annotations

import torch


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None = None, mode: str = "mean") -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, length, hidden]")
    if mode == "cls":
        return hidden[:, 0]
    if attention_mask is None:
        attention_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    if mode == "mean":
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if mode == "max":
        return hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf")).amax(dim=1)
    if mode == "mean_max":
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_value = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf")).amax(dim=1)
        return torch.cat([mean, max_value], dim=-1)
    raise ValueError(f"Unknown pooling mode: {mode}")
