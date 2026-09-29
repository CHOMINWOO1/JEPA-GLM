from __future__ import annotations

import torch

from jepa_glm.losses.jepa_loss import jepa_cosine_loss, jepa_loss, jepa_mse_loss
from jepa_glm.losses.regularization import covariance_regularization, variance_regularization


def test_jepa_cosine_loss_is_low_for_identical_embeddings() -> None:
    z = torch.randn(4, 8)
    assert jepa_cosine_loss(z, z).item() < 1e-5


def test_variance_regularization_is_finite() -> None:
    z = torch.randn(4, 8)
    assert torch.isfinite(variance_regularization(z))


def test_jepa_mse_loss_is_zero_for_identical_embeddings() -> None:
    z = torch.randn(4, 8)
    assert jepa_mse_loss(z, z).item() == 0.0
    assert jepa_loss(z, z, "mse").item() == 0.0


def test_covariance_regularization_is_finite_and_differentiable() -> None:
    z = torch.randn(8, 16, requires_grad=True)

    loss = covariance_regularization(z)
    loss.backward()

    assert torch.isfinite(loss)
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_covariance_regularization_matches_feature_space_definition() -> None:
    z = torch.randn(6, 9)
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (z.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    expected = off_diagonal.square().sum() / z.shape[1]

    assert torch.allclose(covariance_regularization(z), expected, rtol=1e-5, atol=1e-6)
