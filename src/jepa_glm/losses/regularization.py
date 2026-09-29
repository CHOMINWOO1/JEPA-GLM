from __future__ import annotations

import torch


def variance_regularization(embedding: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(embedding.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def covariance_regularization(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape [batch, hidden]")
    if embedding.shape[0] < 2:
        return embedding.sum() * 0.0
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    denominator = float(embedding.shape[0] - 1)
    # ||X^T X||_F == ||X X^T||_F; the sample-space form avoids a hidden_size^2 matrix.
    sample_gram = centered @ centered.transpose(0, 1)
    covariance_frobenius_sq = sample_gram.square().sum() / (denominator * denominator)
    covariance_diagonal_sq = centered.square().sum(dim=0).square().sum() / (denominator * denominator)
    return (covariance_frobenius_sq - covariance_diagonal_sq).clamp_min(0.0) / embedding.shape[1]


def embedding_std(embedding: torch.Tensor) -> float:
    return float(torch.sqrt(embedding.detach().var(dim=0, unbiased=False) + 1e-4).mean().item())
