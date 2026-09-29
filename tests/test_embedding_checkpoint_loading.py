from __future__ import annotations

import torch

from jepa_glm.evaluation.embedding import load_context_encoder_module_state, load_context_encoder_state
from jepa_glm.models.backbone import TinyDnaBackbone
from jepa_glm.models.context_encoder import ContextEncoder
from jepa_glm.models.jepa_glm import JepaGlmModel


def test_context_encoder_loading_ignores_incompatible_predictor() -> None:
    source = JepaGlmModel(
        {
            "vocab_size": 8,
            "hidden_size": 16,
            "max_length": 16,
            "jepa_target_mode": "cls",
            "predictor_embed_dim": 8,
            "predictor_layers": 1,
            "predictor_heads": 2,
            "regularization_queue_size": 4,
        }
    )
    target = JepaGlmModel({"vocab_size": 8, "hidden_size": 16, "max_length": 16})
    with torch.no_grad():
        next(source.context_encoder.parameters()).fill_(0.25)

    loaded = load_context_encoder_state(target, source.state_dict())

    assert loaded > 0
    assert torch.allclose(next(target.context_encoder.parameters()), torch.full_like(next(target.context_encoder.parameters()), 0.25))


def test_standalone_context_encoder_loading_strips_checkpoint_prefix() -> None:
    source = JepaGlmModel({"vocab_size": 8, "hidden_size": 16, "max_length": 16})
    target = ContextEncoder(TinyDnaBackbone(vocab_size=8, hidden_size=16, max_length=16))
    with torch.no_grad():
        next(source.context_encoder.parameters()).fill_(0.5)

    loaded = load_context_encoder_module_state(target, source.state_dict())

    assert loaded > 0
    assert torch.allclose(next(target.parameters()), torch.full_like(next(target.parameters()), 0.5))
