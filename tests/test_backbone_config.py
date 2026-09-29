from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.models.backbone import apply_freeze_policy, build_backbone
from jepa_glm.utils.config import load_config
from jepa_glm.utils.validation import validate_config


class ToyBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(8, 4)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])


def test_apply_freeze_policy_all() -> None:
    model = ToyBert()
    apply_freeze_policy(model, "all")
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_apply_freeze_policy_encoder_last_n() -> None:
    model = ToyBert()
    apply_freeze_policy(model, "encoder_last_n", trainable_last_n_layers=1)
    assert not model.encoder.layer[0].weight.requires_grad
    assert model.encoder.layer[-1].weight.requires_grad


def test_validate_config_rejects_hf_backbone_with_dna_tokenizer() -> None:
    with pytest.raises(ValueError, match="matching Hugging Face tokenizer"):
        validate_config({"model": {"backbone_name": "some/model", "tokenizer": {"type": "dna"}}})


def test_validate_config_rejects_nonpositive_discriminative_learning_rate() -> None:
    with pytest.raises(ValueError, match="predictor_learning_rate must be positive"):
        validate_config({"training": {"predictor_learning_rate": 0.0}})


def test_hf_preset_config_validates_without_download() -> None:
    config = load_config("configs/pretrain_dnabert2_hf.yaml")
    assert config["model"]["backbone_name"] == "zhihan1996/DNABERT-2-117M"
    assert config["model"]["tokenizer"]["type"] == "hf"


def test_hf_backbone_prefers_local_model_dir(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, trust_remote_code: bool = True, **_kwargs):
            calls.append((model_name, trust_remote_code))
            fake = ToyBert()
            fake.config = SimpleNamespace(hidden_size=4)
            fake.gradient_checkpointing_enable = lambda: None
            return fake

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    backbone = build_backbone(
        {
            "backbone_name": "zhihan1996/DNABERT-2-117M",
            "local_model_dir": "models/dnabert2-117m",
            "freeze_policy": "all",
            "trust_remote_code": False,
        }
    )

    assert backbone.hidden_size == 4
    assert calls == [("models/dnabert2-117m", False)]


def test_hf_backbone_continues_when_gradient_checkpointing_is_unsupported(monkeypatch) -> None:
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name: str, **_kwargs):
            fake = ToyBert()
            fake.config = SimpleNamespace(hidden_size=4)

            def unsupported_gradient_checkpointing() -> None:
                raise ValueError("not compatible with gradient checkpointing")

            fake.gradient_checkpointing_enable = unsupported_gradient_checkpointing
            return fake

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    with pytest.warns(RuntimeWarning, match="not supported by this backbone"):
        backbone = build_backbone(
            {
                "backbone_name": "zhihan1996/DNABERT-2-117M",
                "freeze_policy": "encoder_last_n",
                "trainable_last_n_layers": 1,
                "gradient_checkpointing": True,
            }
        )

    assert backbone.hidden_size == 4


def test_hf_tokenizer_prefers_local_model_dir(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        pad_token_id = 0
        mask_token_id = 1
        eos_token = "[EOS]"
        unk_token = "[UNK]"

        def __len__(self) -> int:
            return 8

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_name: str, trust_remote_code: bool = True):
            calls.append((tokenizer_name, trust_remote_code))
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    tokenizer = build_tokenizer(
        {
            "model": {
                "backbone_name": "zhihan1996/DNABERT-2-117M",
                "local_model_dir": "models/dnabert2-117m",
                "trust_remote_code": False,
                "tokenizer": {"type": "hf", "name": "zhihan1996/DNABERT-2-117M"},
            }
        }
    )

    assert tokenizer.vocab_size == 8
    assert calls == [("models/dnabert2-117m", False)]
