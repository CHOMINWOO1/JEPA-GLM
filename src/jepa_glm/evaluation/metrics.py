from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


def binary_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def binary_classification_metrics_with_ci(
    y_true,
    y_score,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> dict[str, object]:
    metrics = binary_classification_metrics(y_true, y_score, threshold=threshold)
    intervals = bootstrap_binary_metric_intervals(
        y_true,
        y_score,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {**metrics, "confidence_intervals": intervals}


def bootstrap_binary_metric_intervals(
    y_true,
    y_score,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> dict[str, dict[str, float | int]]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if y_true.shape[0] == 0:
        raise ValueError("Cannot bootstrap metrics for an empty dataset")
    alpha = (1.0 - confidence_level) / 2.0
    lower_q = 100.0 * alpha
    upper_q = 100.0 * (1.0 - alpha)
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {name: [] for name in ("auroc", "auprc", "accuracy", "f1", "mcc")}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, y_true.shape[0], size=y_true.shape[0])
        if len(set(y_true[indices].tolist())) < 2:
            continue
        sample_metrics = binary_classification_metrics(y_true[indices], y_score[indices], threshold=threshold)
        for name, value in sample_metrics.items():
            if np.isfinite(value):
                sampled[name].append(float(value))
    intervals: dict[str, dict[str, float | int]] = {}
    for name, values in sampled.items():
        if values:
            array = np.asarray(values, dtype=float)
            intervals[name] = {
                "lower": float(np.percentile(array, lower_q)),
                "upper": float(np.percentile(array, upper_q)),
                "n_bootstrap_valid": int(array.shape[0]),
            }
        else:
            intervals[name] = {"lower": float("nan"), "upper": float("nan"), "n_bootstrap_valid": 0}
    return intervals
