from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auroc_delta_bootstrap(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 7,
) -> dict[str, float | int]:
    if not (labels.shape == candidate_scores.shape == reference_scores.shape):
        raise ValueError("Paired score arrays must have identical shapes")
    if labels.ndim != 1 or labels.shape[0] == 0 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Paired AUROC bootstrap requires nonempty binary labels")
    point = float(roc_auc_score(labels, candidate_scores) - roc_auc_score(labels, reference_scores))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, labels.shape[0], size=labels.shape[0])
        sampled_labels = labels[indices]
        if len(set(sampled_labels.tolist())) < 2:
            continue
        deltas.append(
            float(
                roc_auc_score(sampled_labels, candidate_scores[indices])
                - roc_auc_score(sampled_labels, reference_scores[indices])
            )
        )
    if not deltas:
        raise ValueError("No valid paired bootstrap replicates")
    values = np.asarray(deltas, dtype=float)
    if values.shape[0] < 2:
        raise ValueError("At least two valid paired bootstrap replicates are required")
    return {
        "auroc_delta": point,
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
        "probability_delta_positive": float(np.mean(values > 0.0)),
        "bootstrap_standard_error": float(np.std(values, ddof=1)),
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_valid": int(values.shape[0]),
        "seed": seed,
    }


def build_paired_finetune_analysis(
    decision_json: str | Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 7,
) -> dict[str, Any]:
    decision_path = Path(decision_json)
    decision = _read_json(decision_path)
    candidate = str(decision["candidate"])
    root = Path(decision["results_root"])
    task_results = {}
    for task, detail in decision["tasks"].items():
        reference = str(detail["best_reference"])
        candidate_metrics_path = root / task / candidate / "seed_7.json"
        reference_metrics_path = root / task / reference / "seed_7.json"
        candidate_metrics = _read_json(candidate_metrics_path)
        reference_metrics = _read_json(reference_metrics_path)
        candidate_rows = _read_score_rows(candidate_metrics)
        reference_rows = _read_score_rows(reference_metrics)
        if (
            candidate_rows[0] != reference_rows[0]
            or candidate_rows[1] != reference_rows[1]
            or not np.array_equal(candidate_rows[2], reference_rows[2])
        ):
            raise ValueError(f"Paired score identities or labels do not align for {task}")
        labels = candidate_rows[2]
        analysis = paired_auroc_delta_bootstrap(
            labels,
            candidate_rows[3],
            reference_rows[3],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if not np.isclose(float(analysis["auroc_delta"]), float(detail["delta"]), atol=1e-12):
            raise ValueError(f"Paired score delta does not reproduce decision for {task}")
        task_results[task] = {
            "candidate": candidate,
            "reference": reference,
            "n_rows": int(labels.shape[0]),
            **analysis,
            "candidate_scores_csv": candidate_metrics["scores_csv"],
            "candidate_scores_sha256": candidate_metrics["scores_csv_sha256"],
            "reference_scores_csv": reference_metrics["scores_csv"],
            "reference_scores_sha256": reference_metrics["scores_csv_sha256"],
        }
    return {
        "status": "paired_posthoc_uncertainty_complete",
        "gate_reopened": False,
        "decision_status_unchanged": decision["status"],
        "decision_json": str(decision_path),
        "decision_json_sha256": _sha256_file(decision_path),
        "candidate": candidate,
        "tasks": task_results,
        "test_or_confirmation_scores_used": False,
    }


def write_paired_finetune_analysis(
    decision_json: str | Path,
    output_json: str | Path,
    output_md: str | Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 7,
) -> dict[str, Any]:
    payload = build_paired_finetune_analysis(decision_json, n_bootstrap=n_bootstrap, seed=seed)
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Paired Fine-Tuning AUROC Delta Analysis",
        "",
        "This post-hoc uncertainty analysis does not reopen the preregistered development gate.",
        "",
        "|task|reference|delta|paired 95% CI|P(delta > 0)|rows|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, result in payload["tasks"].items():
        lines.append(
            f"|{task}|{result['reference']}|{result['auroc_delta']:+.6f}|"
            f"[{result['ci95_lower']:+.6f}, {result['ci95_upper']:+.6f}]|"
            f"{result['probability_delta_positive']:.4f}|{result['n_rows']}|"
        )
    lines.extend(["", "No test or confirmation score was used.", ""])
    Path(output_md).write_text("\n".join(lines), encoding="utf-8")
    return payload


def _read_score_rows(metrics: dict[str, Any]) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    path = Path(metrics["scores_csv"])
    if _sha256_file(path) != metrics["scores_csv_sha256"]:
        raise ValueError(f"Score artifact changed: {path}")
    indices: list[int] = []
    identities: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row_index", "sequence_sha256", "label", "score"]:
            raise ValueError(f"Unexpected score schema: {path}")
        for row in reader:
            indices.append(int(row["row_index"]))
            identities.append(row["sequence_sha256"])
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return indices, identities, np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
