from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_cosine_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1).mean()


def jepa_mse_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, target.detach())


def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, loss_type: str = "cosine") -> torch.Tensor:
    normalized = loss_type.lower()
    if normalized == "cosine":
        return jepa_cosine_loss(predicted, target)
    if normalized == "mse":
        return jepa_mse_loss(predicted, target)
    raise ValueError("JEPA loss type must be 'cosine' or 'mse'")
