from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def prepare_gue_external_task(
    *,
    task: str,
    task_dir: str | Path,
    output_csv: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    root = Path(task_dir)
    paths = {"train": root / "train.csv", "validation": root / "dev.csv", "test": root / "test.csv"}
    tables = {split: _read_gue_csv(path) for split, path in paths.items()}
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_{right}"] = len(set(tables[left][0]) & set(tables[right][0]))
    train_sequences = set(tables["train"][0])
    validation_rows = [
        (sequence, label)
        for sequence, label in zip(*tables["validation"], strict=True)
        if sequence not in train_sequences
    ]
    filtered_tables = {
        "train": tables["train"],
        "validation": (
            [sequence for sequence, _ in validation_rows],
            [label for _, label in validation_rows],
        ),
    }
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "sequence", "label"])
        writer.writeheader()
        for split in ("train", "validation"):
            sequences, labels = filtered_tables[split]
            for sequence, label in zip(sequences, labels, strict=True):
                writer.writerow({"split": split, "sequence": sequence, "label": label})
    records = {}
    for split, path in paths.items():
        sequences, labels = tables[split]
        records[split] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "rows": len(labels),
            "label_counts": {str(label): labels.count(label) for label in sorted(set(labels))},
            "unique_sequences": len(set(sequences)),
            "sequence_length_min": min(map(len, sequences)),
            "sequence_length_max": max(map(len, sequences)),
        }
    manifest = {
        "status": "prepared_with_confirmation_locked",
        "benchmark": "Genome Understanding Evaluation",
        "task": task,
        "raw_files": records,
        "exact_sequence_overlaps": overlaps,
        "development_csv": str(output_path),
        "development_csv_sha256": _sha256_file(output_path),
        "development_splits": ["train", "validation"],
        "development_rows": len(filtered_tables["train"][1]) + len(filtered_tables["validation"][1]),
        "development_split_rows": {
            "train": len(filtered_tables["train"][1]),
            "validation": len(filtered_tables["validation"][1]),
        },
        "development_overlap_exclusions": {
            "validation_exact_matches_to_train": len(tables["validation"][1]) - len(filtered_tables["validation"][1]),
        },
        "development_exact_sequence_overlap_after_filter": len(
            set(filtered_tables["train"][0]) & set(filtered_tables["validation"][0])
        ),
        "confirmation_split": "test",
        "confirmation_status": "locked_unevaluated",
        "confirmation_rows_in_development_csv": 0,
        "confirmation_rows_loaded_for_integrity_audit_only": len(tables["test"][1]),
        "confirmation_rows_tokenized_or_embedded": False,
        "confirmation_scores_or_metrics_persisted": False,
    }
    manifest_path = Path(output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_gue_csv(path: Path) -> tuple[list[str], list[int]]:
    sequences: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sequence", "label"]:
            raise ValueError(f"Unexpected GUE schema in {path}: {reader.fieldnames}")
        for row in reader:
            sequence = row["sequence"].strip().upper()
            label = int(row["label"])
            if not sequence or set(sequence) - set("ACGTN"):
                raise ValueError(f"Invalid DNA sequence in {path}")
            if label not in {0, 1}:
                raise ValueError(f"Expected binary labels in {path}")
            sequences.append(sequence)
            labels.append(label)
    if not sequences or set(labels) != {0, 1}:
        raise ValueError(f"GUE split must be nonempty and contain both labels: {path}")
    return sequences, labels


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
