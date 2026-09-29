from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    num_sequences: int
    sequence_length: int
    source: str
    split: str = "synthetic"
    seed: int | None = None


@dataclass(frozen=True)
class ModelManifest:
    backbone_name: str
    hidden_size: int
    trainable_parameters: int
    total_parameters: int
    pooling: str
    predictor_layers: int


@dataclass(frozen=True)
class ExperimentManifest:
    run_id: str
    created_at_utc: str
    config_sha256: str
    python: str
    torch: str
    platform: str
    device: str
    dataset: DatasetManifest
    model: ModelManifest
    config: dict[str, Any]


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def build_experiment_manifest(
    config: dict[str, Any],
    model: torch.nn.Module,
    dataset: DatasetManifest,
    device: torch.device,
    run_id: str,
) -> ExperimentManifest:
    trainable, total = parameter_counts(model)
    model_cfg = config.get("model", {})
    model_manifest = ModelManifest(
        backbone_name=str(model_cfg.get("backbone_name") or "TinyDnaBackbone"),
        hidden_size=int(model_cfg.get("hidden_size", getattr(model, "hidden_size", 0) or 0)),
        trainable_parameters=trainable,
        total_parameters=total,
        pooling=str(model_cfg.get("pooling", "mean")),
        predictor_layers=int(model_cfg.get("predictor_layers", 2)),
    )
    return ExperimentManifest(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=stable_json_sha256(config),
        python=platform.python_version(),
        torch=torch.__version__,
        platform=platform.platform(),
        device=str(device),
        dataset=dataset,
        model=model_manifest,
        config=config,
    )


def write_manifest(path: str | Path, manifest: ExperimentManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
