from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from jepa_glm.data.manifest import split_for_chromosome


@dataclass(frozen=True)
class BenchmarkSplitSummary:
    split: str
    n_examples: int
    label_counts: dict[str, int]
    chromosomes: list[str]


@dataclass(frozen=True)
class BenchmarkManifest:
    source_csv: str
    output_csv: str
    seed: int
    sequence_column: str
    label_column: str
    split_strategy: str
    chromosome_column: str | None
    dropped_rows: int
    source_sha256: str
    splits: list[BenchmarkSplitSummary]


def create_labeled_benchmark_split(
    input_csv: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    seed: int = 7,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    sequence_column: str = "sequence",
    label_column: str = "label",
    chromosome_column: str = "chrom",
) -> BenchmarkManifest:
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must satisfy train > 0, validation >= 0, and train + validation < 1")
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    manifest_json = Path(manifest_json)
    rows = _read_rows(input_csv, sequence_column, label_column)
    if rows and chromosome_column in rows[0]:
        split_rows = _chromosome_split_rows(rows, sequence_column, label_column, chromosome_column)
        split_strategy = "chromosome_disjoint"
        manifest_chromosome_column: str | None = chromosome_column
    else:
        split_rows = _random_split_rows(rows, seed, train_fraction, validation_fraction, sequence_column, label_column)
        split_strategy = "seeded_stratified_random"
        manifest_chromosome_column = None
    dropped_rows = len(rows) - len(split_rows)
    split_rows.sort(key=lambda row: (row["split"], row[label_column], row.get(chromosome_column, ""), row[sequence_column]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames_for_rows(["split", sequence_column, label_column, chromosome_column, "start", "end"], split_rows))
        writer.writeheader()
        writer.writerows(split_rows)
    summaries = []
    for split in ("train", "validation", "test"):
        split_subset = [row for row in split_rows if row["split"] == split]
        labels = [row[label_column] for row in split_subset]
        chromosomes = sorted({_normalize_chromosome(row[chromosome_column]) for row in split_subset if chromosome_column in row})
        summaries.append(BenchmarkSplitSummary(split, len(labels), dict(sorted(Counter(labels).items())), chromosomes))
    manifest = BenchmarkManifest(
        source_csv=str(input_csv),
        output_csv=str(output_csv),
        seed=seed,
        sequence_column=sequence_column,
        label_column=label_column,
        split_strategy=split_strategy,
        chromosome_column=manifest_chromosome_column,
        dropped_rows=dropped_rows,
        source_sha256=_sha256_file(input_csv),
        splits=summaries,
    )
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _read_rows(path: Path, sequence_column: str, label_column: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {sequence_column, label_column}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        return [row for row in reader]


def _chromosome_split_rows(
    rows: list[dict[str, str]],
    sequence_column: str,
    label_column: str,
    chromosome_column: str,
) -> list[dict[str, str]]:
    split_rows = []
    for row in rows:
        split = split_for_chromosome(row[chromosome_column])
        if split is None:
            continue
        split_rows.append(_project_row(row, split, sequence_column, label_column))
    return split_rows


def _random_split_rows(
    rows: list[dict[str, str]],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    sequence_column: str,
    label_column: str,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[label_column], []).append(row)
    split_rows: list[dict[str, str]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train, n_validation = _allocate_split_counts(n, train_fraction, validation_fraction)
        for split, subset in (
            ("train", label_rows[:n_train]),
            ("validation", label_rows[n_train : n_train + n_validation]),
            ("test", label_rows[n_train + n_validation :]),
        ):
            for row in subset:
                split_rows.append(_project_row(row, split, sequence_column, label_column))
    return split_rows


def _project_row(row: dict[str, str], split: str, sequence_column: str, label_column: str) -> dict[str, str]:
    projected = dict(row)
    projected["split"] = split
    projected[sequence_column] = row[sequence_column]
    projected[label_column] = row[label_column]
    return projected


def _fieldnames_for_rows(preferred: list[str], rows: list[dict[str, str]]) -> list[str]:
    seen = set()
    fieldnames = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    return fieldnames or preferred


def _normalize_chromosome(chromosome: str) -> str:
    chromosome = chromosome.strip()
    if not chromosome:
        return chromosome
    return chromosome if chromosome.startswith("chr") else f"chr{chromosome}"


def _allocate_split_counts(n: int, train_fraction: float, validation_fraction: float) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    n_train = max(1, math.floor(n * train_fraction))
    n_validation = math.floor(n * validation_fraction)
    if n >= 3:
        n_validation = max(1, n_validation)
        if n_train + n_validation >= n:
            n_train = max(1, n - n_validation - 1)
    elif n == 2:
        n_validation = 0
        n_train = 1
    else:
        n_validation = 0
        n_train = 1
    return n_train, n_validation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
