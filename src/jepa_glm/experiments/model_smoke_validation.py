from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSmokeValidationCheck:
    name: str
    status: str
    detail: str


def validate_model_smoke(
    model_smoke_json: str | Path,
    require_backward: bool = True,
    require_cuda: bool = False,
) -> list[ModelSmokeValidationCheck]:
    path = Path(model_smoke_json)
    if not path.exists():
        return [ModelSmokeValidationCheck("model_smoke.exists", "fail", str(path))]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return [ModelSmokeValidationCheck("model_smoke.mapping", "fail", str(path))]
    checks = [
        ModelSmokeValidationCheck("model_smoke.exists", "pass", str(path)),
        _check("model_smoke.status", payload.get("status") == "pass", str(payload.get("status"))),
        _check("model_smoke.outputs_finite", payload.get("all_outputs_finite") is True, str(payload.get("all_outputs_finite"))),
        _positive_int("model_smoke.batch_size", payload.get("batch_size")),
        _positive_int("model_smoke.sequence_length", payload.get("sequence_length")),
        _shape_check("model_smoke.predicted_embedding_shape", payload.get("predicted_embedding_shape"), expected_rank=2),
        _shape_check("model_smoke.target_embedding_shape", payload.get("target_embedding_shape"), expected_rank=2),
        _shape_check("model_smoke.context_hidden_shape", payload.get("context_hidden_shape"), expected_rank=3),
        _shape_check("model_smoke.mlm_logits_shape", payload.get("mlm_logits_shape"), expected_rank=3),
        _positive_int("model_smoke.trainable_parameters", payload.get("trainable_parameters")),
        _positive_int("model_smoke.total_parameters", payload.get("total_parameters")),
    ]
    checks.append(
        _check(
            "model_smoke.parameter_count_order",
            int(payload.get("total_parameters") or 0) >= int(payload.get("trainable_parameters") or 0) > 0,
            f"trainable={payload.get('trainable_parameters')}; total={payload.get('total_parameters')}",
        )
    )
    if require_backward:
        checks.extend(
            [
                _check("model_smoke.backward_checked", payload.get("backward_checked") is True, str(payload.get("backward_checked"))),
                _check("model_smoke.backward_ok", payload.get("backward_ok") is True, str(payload.get("backward_ok"))),
                _check("model_smoke.gradients_finite", payload.get("gradients_finite") is True, str(payload.get("gradients_finite"))),
                _positive_float("model_smoke.gradient_norm", payload.get("gradient_norm")),
                _nonnegative_float("model_smoke.loss", payload.get("smoke_loss")),
            ]
        )
    if require_cuda:
        checks.append(_check("model_smoke.device.cuda", str(payload.get("device", "")).startswith("cuda"), str(payload.get("device"))))
    checks.append(_level_check(checks))
    return checks


def model_smoke_validation_passed(checks: list[ModelSmokeValidationCheck]) -> bool:
    return any(check.name == "model_smoke.level" and check.status == "pass" for check in checks)


def write_model_smoke_validation_json(path: str | Path, checks: list[ModelSmokeValidationCheck]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shape_check(name: str, value: Any, expected_rank: int) -> ModelSmokeValidationCheck:
    ok = isinstance(value, list) and len(value) == expected_rank and all(isinstance(item, int) and item > 0 for item in value)
    return _check(name, ok, str(value))


def _positive_int(name: str, value: Any) -> ModelSmokeValidationCheck:
    return _check(name, isinstance(value, int) and value > 0, str(value))


def _positive_float(name: str, value: Any) -> ModelSmokeValidationCheck:
    return _check(name, isinstance(value, (float, int)) and float(value) > 0, str(value))


def _nonnegative_float(name: str, value: Any) -> ModelSmokeValidationCheck:
    return _check(name, isinstance(value, (float, int)) and float(value) >= 0, str(value))


def _level_check(checks: list[ModelSmokeValidationCheck]) -> ModelSmokeValidationCheck:
    failed = [check.name for check in checks if check.status == "fail"]
    if failed:
        return ModelSmokeValidationCheck("model_smoke.level", "fail", f"failed={failed}")
    return ModelSmokeValidationCheck("model_smoke.level", "pass", "model_smoke_ready")


def _check(name: str, ok: bool, detail: str) -> ModelSmokeValidationCheck:
    return ModelSmokeValidationCheck(name, "pass" if ok else "fail", detail)
