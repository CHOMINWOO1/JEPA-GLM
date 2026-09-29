from __future__ import annotations

import bisect
import csv
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"sequence", "label", "chrom", "start", "end"}


def create_decontaminated_chromosome_probe_benchmark(
    source_csv: str | Path,
    exposed_compact_csv: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    *,
    development_chromosome: str = "chr20",
    confirmation_chromosome: str = "chr21",
    coordinate_buffer_bp: int = 1024,
    examples_per_split: int = 1000,
    selection_seed: int = 20260716,
    use_source_split: bool = True,
    train_chromosomes: tuple[str, ...] = tuple(f"chr{index}" for index in range(1, 20)),
) -> dict[str, Any]:
    if examples_per_split <= 0 or examples_per_split % 2 != 0:
        raise ValueError("examples_per_split must be a positive even integer")
    if coordinate_buffer_bp < 0:
        raise ValueError("coordinate_buffer_bp must be nonnegative")
    source = Path(source_csv)
    exposed = Path(exposed_compact_csv)
    development_chromosome = _normalize_chromosome(development_chromosome)
    confirmation_chromosome = _normalize_chromosome(confirmation_chromosome)
    if development_chromosome == confirmation_chromosome:
        raise ValueError("Development and confirmation chromosomes must differ")
    normalized_train_chromosomes = {_normalize_chromosome(chrom) for chrom in train_chromosomes}
    if development_chromosome in normalized_train_chromosomes or confirmation_chromosome in normalized_train_chromosomes:
        raise ValueError("Train, development, and confirmation chromosomes must be disjoint")

    forbidden = _load_forbidden_intervals(
        exposed,
        chromosomes={development_chromosome, confirmation_chromosome},
        buffer_bp=coordinate_buffer_bp,
    )
    per_label = examples_per_split // 2
    capacity = per_label * 4
    reservoirs: dict[tuple[str, int], list[tuple[int, int, dict[str, str]]]] = {
        (role, label): []
        for role in ("train", "validation", "test")
        for label in (0, 1)
    }
    eligible_counts: Counter[tuple[str, int]] = Counter()
    excluded_near_exposed: Counter[str] = Counter()
    source_rows = 0
    fieldnames: list[str] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required_columns = REQUIRED_COLUMNS | ({"split"} if use_source_split else set())
        missing = required_columns.difference(fieldnames)
        if missing:
            raise ValueError(f"Source CSV is missing required columns: {sorted(missing)}")
        if "split" not in fieldnames:
            fieldnames = ["split", *fieldnames]
        for index, row in enumerate(reader):
            source_rows += 1
            role = _benchmark_role(
                row,
                development_chromosome,
                confirmation_chromosome,
                use_source_split=use_source_split,
                train_chromosomes=normalized_train_chromosomes,
            )
            if role is None:
                continue
            label = int(row["label"])
            if label not in {0, 1}:
                continue
            if role in {"validation", "test"} and _overlaps_forbidden(row, forbidden):
                excluded_near_exposed[role] += 1
                continue
            eligible_counts[(role, label)] += 1
            priority = _priority(selection_seed, role, index)
            _keep_lowest_priority(reservoirs[(role, label)], capacity, priority, index, row)

    selected = _select_balanced_unique(reservoirs, per_label=per_label)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, int]] = Counter()
    selected_indices: dict[str, list[int]] = {role: [] for role in ("train", "validation", "test")}
    selected_sequences: dict[str, list[str]] = {role: [] for role in ("train", "validation", "test")}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for role in ("train", "validation", "test"):
            for index, row in sorted(selected[role], key=lambda item: item[0]):
                output_row = dict(row)
                output_row["split"] = role
                writer.writerow(output_row)
                label = int(output_row["label"])
                counts[(role, label)] += 1
                selected_indices[role].append(index)
                selected_sequences[role].append(output_row["sequence"])

    expected_total = examples_per_split * 3
    if sum(counts.values()) != expected_total:
        raise ValueError(f"Benchmark is incomplete: written={sum(counts.values())}; expected={expected_total}")
    sequence_sets = {role: set(values) for role, values in selected_sequences.items()}
    sequence_overlap = sum(
        len(sequence_sets[left].intersection(sequence_sets[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if sequence_overlap:
        raise ValueError(f"Selected splits contain exact sequence overlap: {sequence_overlap}")

    payload: dict[str, Any] = {
        "status": "locked_unevaluated",
        "development_status": "unevaluated",
        "confirmation_status": "locked_unevaluated",
        "source_csv": str(source),
        "source_sha256": _sha256_file(source),
        "exposed_compact_csv": str(exposed),
        "exposed_compact_sha256": _sha256_file(exposed),
        "output_csv": str(output),
        "output_sha256": _sha256_file(output),
        "selection_seed": selection_seed,
        "coordinate_buffer_bp": coordinate_buffer_bp,
        "development_chromosome": development_chromosome,
        "confirmation_chromosome": confirmation_chromosome,
        "train_chromosomes": sorted(normalized_train_chromosomes),
        "source_split_policy": "existing_split_column" if use_source_split else "chromosome_assignment_from_raw_source",
        "examples_per_split": examples_per_split,
        "source_rows_scanned": source_rows,
        "forbidden_interval_counts": {chrom: len(intervals) for chrom, intervals in sorted(forbidden.items())},
        "excluded_near_exposed": dict(sorted(excluded_near_exposed.items())),
        "eligible_counts": [
            {"split": role, "label": label, "count": eligible_counts[(role, label)]}
            for role in ("train", "validation", "test")
            for label in (0, 1)
        ],
        "counts": [
            {"split": role, "label": label, "count": counts[(role, label)]}
            for role in ("train", "validation", "test")
            for label in (0, 1)
        ],
        "selected_index_sha256": {
            role: _sha256_values(str(index) for index in selected_indices[role])
            for role in ("train", "validation", "test")
        },
        "selected_sequence_sha256": {
            role: _sha256_values(sorted(selected_sequences[role]))
            for role in ("train", "validation", "test")
        },
        "exact_sequence_overlap_across_splits": sequence_overlap,
        "selection_method": "streaming_deterministic_priority_sample_after_exposed_coordinate_buffer_exclusion",
    }
    manifest = Path(manifest_json)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _benchmark_role(
    row: dict[str, str],
    development_chromosome: str,
    confirmation_chromosome: str,
    *,
    use_source_split: bool,
    train_chromosomes: set[str],
) -> str | None:
    chrom = _normalize_chromosome(row["chrom"])
    if use_source_split:
        split = str(row["split"])
        if split == "train":
            return "train"
        if split == "validation" and chrom == development_chromosome:
            return "validation"
        if split == "validation" and chrom == confirmation_chromosome:
            return "test"
    else:
        if chrom in train_chromosomes:
            return "train"
        if chrom == development_chromosome:
            return "validation"
        if chrom == confirmation_chromosome:
            return "test"
    return None


def _load_forbidden_intervals(
    path: Path,
    *,
    chromosomes: set[str],
    buffer_bp: int,
) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = (REQUIRED_COLUMNS | {"split"}).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Exposed compact CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != "validation":
                continue
            chrom = _normalize_chromosome(row["chrom"])
            if chrom not in chromosomes:
                continue
            start = max(0, int(row["start"]) - buffer_bp)
            end = int(row["end"]) + buffer_bp
            intervals[chrom].append((start, end))
    return {chrom: _merge_intervals(values) for chrom, values in intervals.items()}


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _overlaps_forbidden(
    row: dict[str, str],
    forbidden: dict[str, list[tuple[int, int]]],
) -> bool:
    chrom = _normalize_chromosome(row["chrom"])
    intervals = forbidden.get(chrom, [])
    if not intervals:
        return False
    start = int(row["start"])
    end = int(row["end"])
    starts = [interval[0] for interval in intervals]
    position = bisect.bisect_left(starts, end)
    return position > 0 and intervals[position - 1][1] > start


def _priority(seed: int, role: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}|{role}|{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _keep_lowest_priority(
    heap: list[tuple[int, int, dict[str, str]]],
    capacity: int,
    priority: int,
    index: int,
    row: dict[str, str],
) -> None:
    item = (-priority, -index, dict(row))
    if len(heap) < capacity:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _select_balanced_unique(
    reservoirs: dict[tuple[str, int], list[tuple[int, int, dict[str, str]]]],
    *,
    per_label: int,
) -> dict[str, list[tuple[int, dict[str, str]]]]:
    selected: dict[str, list[tuple[int, dict[str, str]]]] = {
        role: [] for role in ("train", "validation", "test")
    }
    seen_sequences: set[str] = set()
    for role in ("train", "validation", "test"):
        for label in (0, 1):
            candidates = sorted(
                [(-priority, -index, row) for priority, index, row in reservoirs[(role, label)]],
                key=lambda item: (item[0], item[1]),
            )
            chosen = 0
            for _, index, row in candidates:
                sequence = row["sequence"]
                if sequence in seen_sequences:
                    continue
                seen_sequences.add(sequence)
                selected[role].append((index, row))
                chosen += 1
                if chosen == per_label:
                    break
            if chosen != per_label:
                raise ValueError(
                    f"Not enough unique rows for split={role}, label={label}: selected={chosen}; required={per_label}"
                )
    return selected


def _normalize_chromosome(value: str) -> str:
    text = str(value).strip()
    return text if text.lower().startswith("chr") else f"chr{text}"


def _sha256_values(values) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
