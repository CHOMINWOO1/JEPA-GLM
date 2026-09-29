from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from jepa_glm.evaluation.metrics import binary_classification_metrics


def embedding_distribution_metrics(
    embeddings,
    *,
    active_std_threshold: float = 1e-3,
) -> dict[str, float]:
    x = _matrix(embeddings)
    centered = x - x.mean(axis=0, keepdims=True)
    feature_std = centered.std(axis=0, ddof=1)
    covariance = centered.T @ centered / max(1, x.shape[0] - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total > 0.0:
        probabilities = eigenvalues / total
        positive = probabilities > 0.0
        effective_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
        participation_ratio = float(total * total / max(float(np.square(eigenvalues).sum()), 1e-30))
    else:
        effective_rank = 0.0
        participation_ratio = 0.0
    diagonal_abs = float(np.abs(np.diag(covariance)).sum())
    off_diagonal_count = max(1, covariance.size - covariance.shape[0])
    mean_abs_off_diagonal = float((np.abs(covariance).sum() - diagonal_abs) / off_diagonal_count)
    norms = np.linalg.norm(x, axis=1)
    return {
        "feature_std_mean": float(feature_std.mean()),
        "feature_std_median": float(np.median(feature_std)),
        "feature_std_min": float(feature_std.min()),
        "active_feature_fraction": float(np.mean(feature_std >= active_std_threshold)),
        "total_variance": total,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "mean_abs_off_diagonal_covariance": mean_abs_off_diagonal,
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std(ddof=1)),
    }


def class_geometry_metrics(embeddings, labels) -> dict[str, float]:
    x = _matrix(embeddings)
    y = np.asarray(labels)
    classes = np.unique(y)
    if classes.shape[0] != 2:
        raise ValueError("class geometry requires exactly two classes")
    groups = [x[y == label] for label in classes]
    if any(group.shape[0] < 2 for group in groups):
        raise ValueError("each class must contain at least two examples")
    centroids = [group.mean(axis=0) for group in groups]
    centroid_distance = float(np.linalg.norm(centroids[1] - centroids[0]))
    within_variances = [float(np.mean(np.sum(np.square(group - centroid), axis=1))) for group, centroid in zip(groups, centroids)]
    pooled_within_rms = float(np.sqrt(np.mean(within_variances)))
    return {
        "centroid_distance": centroid_distance,
        "pooled_within_rms": pooled_within_rms,
        "centroid_separation": float(centroid_distance / max(pooled_within_rms, 1e-12)),
    }


def linear_cka(embeddings, reference_embeddings) -> float:
    x = _matrix(embeddings)
    y = _matrix(reference_embeddings)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must contain the same examples in the same order")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    x_self = x.T @ x
    y_self = y.T @ y
    denominator = np.linalg.norm(x_self, ord="fro") * np.linalg.norm(y_self, ord="fro")
    if denominator <= 0.0:
        return float("nan")
    return float(np.square(np.linalg.norm(cross, ord="fro")) / denominator)


def knn_probe_metrics(
    embeddings,
    labels,
    *,
    splits: list[str] | None = None,
    n_neighbors: int = 5,
    seed: int = 7,
    test_size: float = 0.2,
) -> dict[str, float | int]:
    x = _matrix(embeddings)
    y = np.asarray(labels, dtype=int)
    if splits is not None:
        split_array = np.asarray(splits)
        train_mask = split_array == "train"
        test_mask = split_array == "test"
        if not train_mask.any() or not test_mask.any():
            raise ValueError("explicit splits must contain train and test rows")
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[test_mask], y[test_mask]
    else:
        stratify = y if np.unique(y).shape[0] > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    neighbors = min(int(n_neighbors), int(x_train.shape[0]))
    if neighbors < 1:
        raise ValueError("k-NN requires at least one training example")
    classifier = KNeighborsClassifier(
        n_neighbors=neighbors,
        metric="cosine",
        weights="distance",
        algorithm="brute",
    )
    classifier.fit(x_train, y_train)
    classes = classifier.classes_.tolist()
    if 1 not in classes:
        raise ValueError("k-NN training split must contain positive examples")
    scores = classifier.predict_proba(x_test)[:, classes.index(1)]
    metrics = binary_classification_metrics(y_test, scores)
    return {
        **metrics,
        "n_neighbors": neighbors,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def _matrix(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("embeddings must be a non-empty two-dimensional matrix with at least two rows")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    return array
