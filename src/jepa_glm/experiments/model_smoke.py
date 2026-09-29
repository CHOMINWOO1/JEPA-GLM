from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.models.jepa_glm import JepaGlmModel


@dataclass(frozen=True)
class ModelSmokeResult:
    status: str
    device: str
    batch_size: int
    sequence_length: int
    predicted_embedding_shape: list[int]
    target_embedding_shape: list[int]
    context_hidden_shape: list[int]
    mlm_logits_shape: list[int]
    all_outputs_finite: bool
    trainable_parameters: int
    total_parameters: int
    backward_checked: bool = False
    smoke_loss: float | None = None
    backward_ok: bool | None = None
    gradient_norm: float | None = None
    gradients_finite: bool | None = None


def run_model_smoke(
    config: dict,
    sequences: list[str] | None = None,
    device: str | None = None,
    max_length: int | None = None,
    run_backward: bool = False,
) -> ModelSmokeResult:
    model_config = dict(config.get("model", config))
    if "loss" in config:
        model_config.setdefault("loss", config["loss"])
    tokenizer = build_tokenizer(model_config)
    sequences = sequences or ["ACGT" * 32, "TGCA" * 32]
    max_length = max_length or int(config.get("data", {}).get("sequence_length", model_config.get("max_length", 128)))
    input_ids = torch.stack([tokenizer.encode(sequence, max_length=max_length) for sequence in sequences])
    attention_mask = input_ids.ne(int(tokenizer.pad_token_id))
    model_config["vocab_size"] = int(getattr(tokenizer, "vocab_size", model_config.get("vocab_size", 8)))
    model = JepaGlmModel(model_config)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(selected_device)
    input_ids = input_ids.to(selected_device)
    attention_mask = attention_mask.to(selected_device)
    model.train(mode=run_backward)
    if run_backward:
        outputs = model(input_ids, input_ids, attention_mask=attention_mask)
    else:
        with torch.no_grad():
            outputs = model(input_ids, input_ids, attention_mask=attention_mask)
    finite = all(torch.isfinite(value).all().item() for value in outputs.values())
    smoke_loss = None
    backward_ok = None
    gradient_norm = None
    gradients_finite = None
    if run_backward:
        loss = (outputs["predicted_embedding"] - outputs["target_embedding"]).float().square().mean()
        loss = loss + 1e-4 * outputs["mlm_logits"].float().square().mean()
        smoke_loss = float(loss.detach().cpu().item())
        loss.backward()
        grad_squares = []
        grad_finite_flags = []
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            grad_squares.append(grad.square().sum())
            grad_finite_flags.append(bool(torch.isfinite(grad).all().item()))
        if grad_squares:
            gradient_norm = float(torch.sqrt(torch.stack(grad_squares).sum()).cpu().item())
        else:
            gradient_norm = 0.0
        gradients_finite = bool(grad_finite_flags) and all(grad_finite_flags)
        backward_ok = bool(torch.isfinite(loss).item()) and gradients_finite and gradient_norm > 0
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return ModelSmokeResult(
        status="pass" if finite and (backward_ok is not False) else "fail",
        device=str(selected_device),
        batch_size=int(input_ids.shape[0]),
        sequence_length=int(input_ids.shape[1]),
        predicted_embedding_shape=list(outputs["predicted_embedding"].shape),
        target_embedding_shape=list(outputs["target_embedding"].shape),
        context_hidden_shape=list(outputs["context_hidden"].shape),
        mlm_logits_shape=list(outputs["mlm_logits"].shape),
        all_outputs_finite=bool(finite),
        trainable_parameters=int(trainable_parameters),
        total_parameters=int(total_parameters),
        backward_checked=bool(run_backward),
        smoke_loss=smoke_loss,
        backward_ok=backward_ok,
        gradient_norm=gradient_norm,
        gradients_finite=gradients_finite,
    )


def write_model_smoke_json(path: str | Path, result: ModelSmokeResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
