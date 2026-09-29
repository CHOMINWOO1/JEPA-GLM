from __future__ import annotations

import json

from jepa_glm.experiments.model_smoke import run_model_smoke, write_model_smoke_json


def test_run_model_smoke_with_tiny_backbone() -> None:
    config = {
        "data": {"sequence_length": 32},
        "model": {
            "vocab_size": 8,
            "hidden_size": 16,
            "max_length": 64,
            "backbone_name": None,
            "tokenizer": {"type": "dna"},
            "pooling": "mean",
            "predictor_layers": 2,
        },
    }

    result = run_model_smoke(config, sequences=["ACGTACGT", "TGCATGCA"], device="cpu", max_length=32)

    assert result.status == "pass"
    assert result.device == "cpu"
    assert result.predicted_embedding_shape == [2, 16]
    assert result.target_embedding_shape == [2, 16]
    assert result.context_hidden_shape == [2, 32, 16]
    assert result.mlm_logits_shape == [2, 32, 7]
    assert result.all_outputs_finite is True
    assert result.total_parameters >= result.trainable_parameters > 0
    assert result.backward_checked is False
    assert result.backward_ok is None


def test_run_model_smoke_can_check_backward_pass() -> None:
    config = {
        "data": {"sequence_length": 16},
        "model": {
            "vocab_size": 8,
            "hidden_size": 16,
            "max_length": 32,
            "backbone_name": None,
            "tokenizer": {"type": "dna"},
            "pooling": "mean",
            "predictor_layers": 1,
        },
    }

    result = run_model_smoke(config, sequences=["ACGTACGT"], device="cpu", max_length=16, run_backward=True)

    assert result.status == "pass"
    assert result.backward_checked is True
    assert result.backward_ok is True
    assert result.gradients_finite is True
    assert result.gradient_norm is not None and result.gradient_norm > 0
    assert result.smoke_loss is not None and result.smoke_loss >= 0


def test_write_model_smoke_json(tmp_path) -> None:
    result = run_model_smoke(
        {
            "data": {"sequence_length": 16},
            "model": {"vocab_size": 8, "hidden_size": 16, "max_length": 32, "tokenizer": {"type": "dna"}},
        },
        sequences=["ACGT"],
        device="cpu",
        max_length=16,
    )
    output = tmp_path / "model_smoke.json"

    write_model_smoke_json(output, result)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["context_hidden_shape"] == [1, 16, 16]
    assert payload["backward_checked"] is False
