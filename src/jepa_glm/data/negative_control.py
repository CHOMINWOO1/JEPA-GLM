from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShuffledLabelManifest:
    source_csv: str
    output_csv: str
    seed: int
    sequence_column: str
    label_column: str
    split_column: str
    source_sha256: str
    preserve_split_distribution: bool
    label_counts_by_split: dict[str, dict[str, int]]


def create_shuffled_label_control(
    input_csv: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    seed: int = 7,
    sequence_column: str = "sequence",
    label_column: str = "label",
    split_column: str = "split",
    preserve_split_distribution: bool = True,
) -> ShuffledLabelManifest:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    manifest_path = Path(manifest_json)
    rows = _read_rows(input_path, {sequence_column, label_column, split_column})
    rng = random.Random(seed)
    shuffled_rows = [dict(row) for row in rows]
    if preserve_split_distribution:
        by_split: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(shuffled_rows):
            by_split[row[split_column]].append(index)
        for indices in by_split.values():
            labels = [shuffled_rows[index][label_column] for index in indices]
            labels = _shuffled_copy(labels, rng)
            for index, label in zip(indices, labels):
                shuffled_rows[index][label_column] = label
    else:
        labels = [row[label_column] for row in shuffled_rows]
        labels = _shuffled_copy(labels, rng)
        for row, label in zip(shuffled_rows, labels):
            row[label_column] = label
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [split_column, sequence_column, label_column]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shuffled_rows)
    manifest = ShuffledLabelManifest(
        source_csv=str(input_path),
        output_csv=str(output_path),
        seed=seed,
        sequence_column=sequence_column,
        label_column=label_column,
        split_column=split_column,
        source_sha256=_sha256_file(input_path),
        preserve_split_distribution=preserve_split_distribution,
        label_counts_by_split=_label_counts_by_split(shuffled_rows, split_column, label_column),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not required.issubset(columns):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        return list(reader)


def _label_counts_by_split(rows: list[dict[str, str]], split_column: str, label_column: str) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counters[row[split_column]][row[label_column]] += 1
    return {split: dict(sorted(counter.items())) for split, counter in sorted(counters.items())}


def _shuffled_copy(labels: list[str], rng: random.Random) -> list[str]:
    shuffled = list(labels)
    if len(set(labels)) < 2:
        return shuffled
    for _ in range(20):
        rng.shuffle(shuffled)
        if any(before != after for before, after in zip(labels, shuffled)):
            return shuffled
    return shuffled


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
