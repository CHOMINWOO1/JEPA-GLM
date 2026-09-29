from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def create_locked_linear_probe_holdout(
    source_csv: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    *,
    exposed_seed: int = 7,
    lock_seed: int = 20260715,
    examples_per_split: int = 1000,
) -> dict[str, Any]:
    source = Path(source_csv)
    if examples_per_split <= 0:
        raise ValueError("examples_per_split must be positive")
    fieldnames, groups = _index_groups(source)
    required = {"split", "sequence", "label"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Source CSV must contain {sorted(required)}")

    exposed_by_split = _sample_indices(groups, seed=exposed_seed, examples_per_split=examples_per_split)
    exposed_test = set(exposed_by_split.get("test", []))
    locked_test = _sample_locked_test(
        groups,
        excluded=exposed_test,
        seed=lock_seed,
        examples=examples_per_split,
    )
    selected = {
        *exposed_by_split.get("train", []),
        *exposed_by_split.get("validation", []),
        *locked_test,
    }
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, str]] = Counter()
    with source.open("r", encoding="utf-8", newline="") as source_handle, output.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        for index, row in enumerate(reader):
            if index not in selected:
                continue
            writer.writerow(row)
            counts[(str(row["split"]), str(row["label"]))] += 1

    overlap = exposed_test.intersection(locked_test)
    payload = {
        "status": "locked_unevaluated",
        "source_csv": str(source),
        "source_sha256": _sha256_file(source),
        "output_csv": str(output),
        "output_sha256": _sha256_file(output),
        "exposed_seed": int(exposed_seed),
        "lock_seed": int(lock_seed),
        "examples_per_split": int(examples_per_split),
        "exposed_test_count": len(exposed_test),
        "locked_test_count": len(locked_test),
        "exposed_test_index_sha256": _sha256_indices(exposed_test),
        "locked_test_index_sha256": _sha256_indices(locked_test),
        "exposed_locked_overlap": len(overlap),
        "selection_method": "exclude_exact_prior_balanced_sample_then_seeded_balanced_sample_without_replacement",
        "counts": [
            {"split": split, "label": int(label), "count": count}
            for (split, label), count in sorted(counts.items())
        ],
    }
    expected_total = examples_per_split * 3
    if len(selected) != expected_total or sum(counts.values()) != expected_total:
        raise ValueError(
            f"Locked benchmark is incomplete: selected={len(selected)}; written={sum(counts.values())}; expected={expected_total}"
        )
    if overlap:
        raise ValueError("Locked test overlaps the previously exposed test sample")
    manifest = Path(manifest_json)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _index_groups(path: Path) -> tuple[set[str], dict[tuple[str, int], list[int]]]:
    groups: dict[tuple[str, int], list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "split" not in fieldnames or "label" not in fieldnames:
            return fieldnames, groups
        for index, row in enumerate(reader):
            groups.setdefault((str(row["split"]), int(row["label"])), []).append(index)
    return fieldnames, groups


def _sample_indices(
    groups: dict[tuple[str, int], list[int]],
    *,
    seed: int,
    examples_per_split: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    result: dict[str, list[int]] = {}
    splits = sorted({split for split, _ in groups})
    for split in splits:
        labels = sorted(label for group_split, label in groups if group_split == split)
        per_label = max(1, examples_per_split // max(1, len(labels)))
        chosen: list[int] = []
        for label in labels:
            candidates = list(groups[(split, label)])
            rng.shuffle(candidates)
            chosen.extend(candidates[:per_label])
        chosen_set = set(chosen)
        if len(chosen) < examples_per_split:
            remaining = [
                index
                for label in labels
                for index in groups[(split, label)]
                if index not in chosen_set
            ]
            rng.shuffle(remaining)
            chosen.extend(remaining[: examples_per_split - len(chosen)])
        result[split] = sorted(chosen[:examples_per_split])
    return result


def _sample_locked_test(
    groups: dict[tuple[str, int], list[int]],
    *,
    excluded: set[int],
    seed: int,
    examples: int,
) -> list[int]:
    rng = random.Random(seed)
    labels = sorted(label for split, label in groups if split == "test")
    if not labels:
        raise ValueError("Source CSV has no test split")
    per_label = max(1, examples // len(labels))
    selected: list[int] = []
    for label in labels:
        candidates = [index for index in groups[("test", label)] if index not in excluded]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_label])
    if len(selected) < examples:
        selected_set = set(selected)
        remaining = [
            index
            for label in labels
            for index in groups[("test", label)]
            if index not in excluded and index not in selected_set
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: examples - len(selected)])
    if len(selected) < examples:
        raise ValueError(f"Not enough unexposed test rows: selected={len(selected)}; required={examples}")
    return sorted(selected[:examples])


def _sha256_indices(indices: set[int] | list[int]) -> str:
    payload = ",".join(str(index) for index in sorted(indices)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
