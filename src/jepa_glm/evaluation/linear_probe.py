from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from jepa_glm.evaluation.metrics import binary_classification_metrics, binary_classification_metrics_with_ci


@dataclass(frozen=True)
class LabeledSequenceTable:
    sequences: list[str]
    labels: list[int]
    splits: list[str] | None = None


def read_labeled_sequence_csv(
    path: str | Path,
    sequence_column: str = "sequence",
    label_column: str = "label",
    split_column: str = "split",
    include_splits: set[str] | None = None,
) -> LabeledSequenceTable:
    sequences: list[str] = []
    labels: list[int] = []
    splits: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if sequence_column not in (reader.fieldnames or []) or label_column not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain '{sequence_column}' and '{label_column}' columns")
        has_split = split_column in (reader.fieldnames or [])
        for row in reader:
            if include_splits is not None:
                if not has_split:
                    raise ValueError("include_splits requires a split column")
                if row[split_column] not in include_splits:
                    continue
            sequences.append(row[sequence_column])
            labels.append(int(row[label_column]))
            if has_split:
                splits.append(row[split_column])
    return LabeledSequenceTable(sequences, labels, splits if splits else None)


def sample_labeled_sequence_table(
    table: LabeledSequenceTable,
    max_examples_per_split: int | None = None,
    seed: int = 7,
    balanced: bool = True,
) -> LabeledSequenceTable:
    if max_examples_per_split is None or max_examples_per_split <= 0:
        return table
    rng = random.Random(seed)
    splits = table.splits or ["all"] * len(table.labels)
    selected: list[int] = []
    for split in sorted(set(splits)):
        split_indices = [index for index, value in enumerate(splits) if value == split]
        if balanced:
            by_label: dict[int, list[int]] = {}
            for index in split_indices:
                by_label.setdefault(table.labels[index], []).append(index)
            labels = sorted(by_label)
            per_label = max(1, max_examples_per_split // max(1, len(labels)))
            split_selected: list[int] = []
            for label in labels:
                candidates = list(by_label[label])
                rng.shuffle(candidates)
                split_selected.extend(candidates[:per_label])
            selected_set = set(split_selected)
            if len(split_selected) < max_examples_per_split:
                remaining = [index for index in split_indices if index not in selected_set]
                rng.shuffle(remaining)
                split_selected.extend(remaining[: max_examples_per_split - len(split_selected)])
            selected.extend(split_selected[:max_examples_per_split])
        else:
            candidates = list(split_indices)
            rng.shuffle(candidates)
            selected.extend(candidates[:max_examples_per_split])
    selected.sort()
    return LabeledSequenceTable(
        [table.sequences[index] for index in selected],
        [table.labels[index] for index in selected],
        [splits[index] for index in selected] if table.splits is not None else None,
    )


def fit_linear_probe(x_train, y_train) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    return clf.fit(x_train, y_train)


def evaluate_linear_probe(
    embeddings,
    labels,
    seed: int = 7,
    test_size: float = 0.2,
    splits: list[str] | None = None,
    fit_splits: tuple[str, ...] = ("train",),
    evaluation_split: str = "test",
) -> dict[str, float]:
    y_test, scores = linear_probe_scores(
        embeddings,
        labels,
        seed=seed,
        test_size=test_size,
        splits=splits,
        fit_splits=fit_splits,
        evaluation_split=evaluation_split,
    )
    return binary_classification_metrics(y_test, scores)


def evaluate_linear_probe_with_ci(
    embeddings,
    labels,
    seed: int = 7,
    test_size: float = 0.2,
    splits: list[str] | None = None,
    n_bootstrap: int = 1000,
    fit_splits: tuple[str, ...] = ("train",),
    evaluation_split: str = "test",
) -> dict[str, object]:
    y_test, scores = linear_probe_scores(
        embeddings,
        labels,
        seed=seed,
        test_size=test_size,
        splits=splits,
        fit_splits=fit_splits,
        evaluation_split=evaluation_split,
    )
    return binary_classification_metrics_with_ci(y_test, scores, n_bootstrap=n_bootstrap, seed=seed)


def linear_probe_scores(
    embeddings,
    labels,
    seed: int = 7,
    test_size: float = 0.2,
    splits: list[str] | None = None,
    fit_splits: tuple[str, ...] = ("train",),
    evaluation_split: str = "test",
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings)
    y = np.asarray(labels)
    if splits is not None:
        split_array = np.asarray(splits)
        train_mask = np.isin(split_array, np.asarray(fit_splits))
        test_mask = split_array == evaluation_split
        if train_mask.any() and test_mask.any():
            clf = fit_linear_probe(x[train_mask], y[train_mask])
            scores = clf.predict_proba(x[test_mask])[:, 1]
            return y[test_mask], scores
        available = sorted(set(split_array.tolist()))
        raise ValueError(
            f"Requested fit_splits={list(fit_splits)} and evaluation_split={evaluation_split!r} "
            f"are not available with nonempty rows; available={available}"
        )
    stratify = y if len(set(y.tolist())) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    clf = fit_linear_probe(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return y_test, scores
