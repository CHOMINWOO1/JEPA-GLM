from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from jepa_glm.evaluation.metrics import binary_classification_metrics, binary_classification_metrics_with_ci


def variant_scores(reference_embedding: torch.Tensor, alternate_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "l2": torch.linalg.vector_norm(alternate_embedding - reference_embedding, dim=-1),
        "cosine": 1.0 - F.cosine_similarity(alternate_embedding, reference_embedding, dim=-1),
    }


@dataclass(frozen=True)
class VariantSequenceTable:
    reference_sequences: list[str]
    alternate_sequences: list[str]
    labels: list[int]


def read_variant_sequence_csv(
    path: str | Path,
    reference_column: str = "ref_sequence",
    alternate_column: str = "alt_sequence",
    label_column: str = "label",
) -> VariantSequenceTable:
    references: list[str] = []
    alternates: list[str] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {reference_column, alternate_column, label_column}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            references.append(row[reference_column])
            alternates.append(row[alternate_column])
            labels.append(int(row[label_column]))
    return VariantSequenceTable(references, alternates, labels)


def write_variant_scores_csv(
    path: str | Path,
    table: VariantSequenceTable,
    scores: dict[str, torch.Tensor],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ref_sequence", "alt_sequence", "label", "l2", "cosine"])
        writer.writeheader()
        for index, label in enumerate(table.labels):
            writer.writerow(
                {
                    "ref_sequence": table.reference_sequences[index],
                    "alt_sequence": table.alternate_sequences[index],
                    "label": label,
                    "l2": float(scores["l2"][index].item()),
                    "cosine": float(scores["cosine"][index].item()),
                }
            )


def evaluate_variant_scores(labels: list[int], scores: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {
        name: binary_classification_metrics(labels, tensor.detach().cpu().numpy())
        for name, tensor in scores.items()
    }


def evaluate_variant_scores_with_ci(
    labels: list[int],
    scores: dict[str, torch.Tensor],
    n_bootstrap: int = 1000,
    seed: int = 7,
) -> dict[str, dict[str, object]]:
    return {
        name: binary_classification_metrics_with_ci(labels, tensor.detach().cpu().numpy(), n_bootstrap=n_bootstrap, seed=seed)
        for name, tensor in scores.items()
    }
