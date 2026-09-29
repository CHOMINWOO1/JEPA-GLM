from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from jepa_glm.data.fasta_dataset import ManifestWindowDataset, ReverseComplementManifestSubset, SyntheticDnaDataset
from jepa_glm.data.sampling import ResumableRandomSampler
from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.training.checkpoint import capture_rng_state, checkpoint_state, load_checkpoint, restore_rng_state, save_checkpoint
from jepa_glm.training.optimizer import build_cosine_scheduler, build_training_optimizer, optimizer_group_manifest
from jepa_glm.training.schedule import plan_training_segment
from jepa_glm.training.trainer import train_steps
from jepa_glm.utils.config import load_config
from jepa_glm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from", help="Checkpoint to resume model, optimizer, scheduler, and global step from.")
    parser.add_argument("--max-new-steps", type=int, help="Run at most this many new steps, capped by training.steps minus the resume step.")
    parser.add_argument("--max-new-optimizer-steps", type=int, help="Run at most this many new optimizer updates.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    resume_state = load_checkpoint(args.resume_from, map_location=str(device)) if args.resume_from else None
    start_step = int(resume_state.get("step", 0)) if resume_state is not None else 0
    grad_accum = int(config.get("training", {}).get("gradient_accumulation_steps", 1))
    start_optimizer_step = (
        int(resume_state.get("optimizer_step", math.ceil(start_step / grad_accum)))
        if resume_state is not None
        else 0
    )
    if resume_state is not None:
        checkpoint_seed = int(resume_state.get("config", {}).get("seed", seed))
        if checkpoint_seed != seed:
            raise SystemExit(f"Resume seed mismatch: checkpoint={checkpoint_seed}; requested={seed}")

    tokenizer = build_tokenizer(config)
    train_dataset, validation_dataset, dataset_info = _build_datasets(config, tokenizer, seed)
    batch_size = int(config.get("training", {}).get("batch_size", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ResumableRandomSampler(
            train_dataset,
            seed=seed,
            start_step=start_step,
            batch_size=batch_size,
        ),
        generator=torch.Generator().manual_seed(seed + 1_000_000),
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 2_000_000),
        )
        if validation_dataset is not None
        else None
    )
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("vocab_size", tokenizer.vocab_size)
    model_cfg.setdefault("pad_token_id", tokenizer.pad_token_id)
    model = JepaGlmModel(model_cfg)
    optimizer = build_training_optimizer(model, config.get("training", {}))
    scheduler = _build_scheduler(optimizer, config)
    scaler_state = None
    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=True)
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if scheduler is not None and "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        scaler_state = resume_state.get("scaler")
        if not restore_rng_state(resume_state.get("rng_state")):
            warnings.warn(
                "Checkpoint has no RNG state; resume remains functional but is not bitwise-equivalent to a v3 checkpoint continuation.",
                RuntimeWarning,
                stacklevel=2,
            )
        del resume_state
    try:
        segment_plan = plan_training_segment(
            config.get("training", {}),
            start_micro_steps=start_step,
            start_optimizer_steps=start_optimizer_step,
            max_new_micro_steps=args.max_new_steps,
            max_new_optimizer_steps=args.max_new_optimizer_steps,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    target_steps = segment_plan.target_micro_steps
    remaining_steps = segment_plan.segment_micro_steps
    if remaining_steps <= 0:
        raise SystemExit(f"No training steps remaining: start_step={start_step}; target_steps={target_steps}")
    effective_config = json.loads(json.dumps(config))
    effective_config.setdefault("training", {})["steps"] = remaining_steps
    summary = train_steps(
        model,
        train_loader,
        effective_config,
        tokenizer,
        device,
        optimizer=optimizer,
        scheduler=scheduler,
        start_step=start_step,
        start_optimizer_step=start_optimizer_step,
        validation_dataloader=validation_loader,
        scaler_state=scaler_state,
    )
    summary_payload = asdict(summary)
    summary_payload.pop("step_metrics", None)
    saved_scaler_state = summary_payload.pop("scaler_state", {})
    summary_payload["condition"] = str(config.get("mode", args.run_id or output_dir.name))
    summary_payload["seed"] = seed
    summary_payload["target_steps"] = target_steps
    summary_payload["step_unit"] = segment_plan.step_unit
    summary_payload["target_micro_steps"] = segment_plan.target_micro_steps
    summary_payload["target_optimizer_steps"] = segment_plan.target_optimizer_steps
    summary_payload["new_steps"] = remaining_steps
    summary_payload["new_micro_steps"] = remaining_steps
    summary_payload["new_optimizer_steps"] = segment_plan.segment_optimizer_steps
    dataset_info.update(_augmentation_stats(train_dataset))
    _write_json(output_dir / "summary.json", summary_payload)
    normalized_new_metrics = _normalize_step_metrics(summary.step_metrics)
    _write_metrics_csv(output_dir / "metrics.csv", _normalize_step_metrics(_merge_step_metrics(output_dir / "metrics.csv", summary.step_metrics, start_step=start_step)))
    _write_metrics_segment(output_dir, normalized_new_metrics, start_step=start_step, completed_steps=summary.completed_steps)
    _write_json(
        output_dir / "manifest.json",
        _manifest(args.run_id or output_dir.name, args.config, config, dataset_info, model, optimizer, device),
    )
    save_checkpoint(
        output_dir / "checkpoint.pt",
        checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            summary=summary_payload,
            step=summary.completed_steps,
            optimizer_step=summary.completed_optimizer_steps,
            scaler_state=saved_scaler_state,
            rng_state=capture_rng_state(),
        ),
    )
    print(
        f"run_dir={output_dir} micro_steps={summary.completed_steps} "
        f"optimizer_steps={summary.completed_optimizer_steps} final_loss={summary.final_loss:.6g}"
    )


def _build_datasets(config: dict[str, Any], tokenizer, seed: int):
    data_cfg = config.get("data", {})
    validation_fraction = float(config.get("training", {}).get("validation_fraction", 0.0))
    if bool(data_cfg.get("synthetic", False)):
        dataset = SyntheticDnaDataset(int(data_cfg.get("num_sequences", 128)), int(data_cfg.get("sequence_length", 128)), tokenizer, seed=seed)
        info = {"kind": "synthetic", "num_sequences": len(dataset), "sequence_length": int(data_cfg.get("sequence_length", 128))}
    else:
        dataset = ManifestWindowDataset(
            data_cfg["fasta_path"],
            data_cfg["manifest_csv"],
            tokenizer,
            split=str(data_cfg.get("split", "train")),
            max_length=int(config.get("model", {}).get("max_length", data_cfg.get("sequence_length", 128))),
            max_windows=int(data_cfg["max_windows"]) if data_cfg.get("max_windows") is not None else None,
        )
        info = {
            "kind": "manifest_window",
            "fasta_path": str(data_cfg["fasta_path"]),
            "manifest_csv": str(data_cfg["manifest_csv"]),
            "split": str(data_cfg.get("split", "train")),
            "num_sequences": len(dataset),
            "max_windows": int(data_cfg["max_windows"]) if data_cfg.get("max_windows") is not None else 0,
        }
    if validation_fraction <= 0.0:
        if isinstance(dataset, ManifestWindowDataset) and float(data_cfg.get("reverse_complement_probability", 0.0)) > 0.0:
            full_subset = torch.utils.data.Subset(dataset, list(range(len(dataset))))
            dataset = ReverseComplementManifestSubset(
                full_subset,
                float(data_cfg["reverse_complement_probability"]),
            )
            info["reverse_complement_probability"] = float(data_cfg["reverse_complement_probability"])
        return dataset, None, info
    n_validation = max(1, int(len(dataset) * validation_fraction))
    n_train = len(dataset) - n_validation
    if n_train <= 0:
        return dataset, None, info
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(dataset, [n_train, n_validation], generator=generator)
    reverse_complement_probability = float(data_cfg.get("reverse_complement_probability", 0.0))
    if isinstance(dataset, ManifestWindowDataset) and reverse_complement_probability > 0.0:
        train_dataset = ReverseComplementManifestSubset(train_dataset, reverse_complement_probability)
        info["reverse_complement_probability"] = reverse_complement_probability
    return train_dataset, validation_dataset, {**info, "train_sequences": len(train_dataset), "validation_sequences": len(validation_dataset)}


def _augmentation_stats(dataset) -> dict[str, int | float]:
    if hasattr(dataset, "augmentation_stats"):
        return dict(dataset.augmentation_stats())
    return {}


def _build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    return build_cosine_scheduler(optimizer, config.get("training", {}))


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _manifest(
    run_id: str,
    config_path: str,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_path": config_path,
        "config": config,
        "dataset": dataset_info,
        "model": {
            "class": type(model).__name__,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "jepa_target_mode": model.jepa_target_mode,
            "target_encoder_eval_mode": model.target_encoder_eval_mode,
            "regularization_queue_size": model.regularization_queue_size,
            "mlm_head_source": model.mlm_head_source,
        },
        "optimizer": {
            "class": type(optimizer).__name__,
            "parameter_groups": optimizer_group_manifest(optimizer),
        },
        "device": str(device),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["step", "global_step", "global_optimizer_step", "loss", "jepa_loss", "mlm_loss", "variance_reg", "covariance_reg", "regularization_batch_size", "regularization_gradient_scale", "embedding_std", "ema_delta", "ema_momentum", "mask_fraction", "mask_fraction_eligible", "target_tokens", "prediction_count", "optimizer_step", "grad_norm", "learning_rate", "backbone_learning_rate", "predictor_learning_rate", "mlm_head_learning_rate"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_segment(output_dir: Path, rows: list[dict[str, float]], start_step: int, completed_steps: int) -> Path | None:
    if not rows:
        return None
    segment_dir = output_dir / "metrics_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    path = segment_dir / f"steps_{start_step + 1:06d}_{completed_steps:06d}.csv"
    _write_metrics_csv(path, rows)
    return path


def _merge_step_metrics(path: Path, new_rows: list[dict[str, float]], start_step: int) -> list[dict[str, float]]:
    if start_step <= 0 or not path.exists():
        return new_rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        previous = [dict(row) for row in reader]
    retained = [row for row in previous if _metric_global_step(row) <= start_step]
    return [*_coerce_metric_rows(retained), *new_rows]


def _metric_global_step(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("global_step", row.get("step", 0))))
    except (TypeError, ValueError):
        return 0


def _coerce_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    coerced = []
    for row in rows:
        coerced.append({key: float(value) for key, value in row.items() if value not in (None, "")})
    return coerced


def _normalize_step_metrics(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    normalized = []
    for row in rows:
        item = dict(row)
        if "global_step" in item:
            item["step"] = float(item["global_step"])
        normalized.append(item)
    return normalized


if __name__ == "__main__":
    main()
