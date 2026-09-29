from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestCheck:
    name: str
    status: str
    detail: str


REQUIRED_COLUMNS = {"split", "chromosome", "start", "end", "length"}


def validate_window_manifest(
    manifest_csv: str | Path,
    required_splits: set[str] | None = None,
    min_windows_per_split: int = 1,
    max_non_acgt_fraction: float | None = None,
) -> list[ManifestCheck]:
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        return [ManifestCheck("manifest.exists", "fail", str(manifest_path))]
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        checks = [_check("manifest.columns", REQUIRED_COLUMNS.issubset(fieldnames), ",".join(sorted(fieldnames)))]
        rows = list(reader)
    if not REQUIRED_COLUMNS.issubset(fieldnames):
        return checks

    required_splits = required_splits or {"train", "validation", "test"}
    split_counts: dict[str, int] = {}
    split_chromosomes: dict[str, set[str]] = {}
    intervals: set[tuple[str, int, int]] = set()
    duplicate_count = 0
    coordinate_errors = 0
    length_errors = 0
    non_acgt_errors = 0
    max_seen_non_acgt = 0.0
    for row in rows:
        split = row["split"]
        chromosome = row["chromosome"]
        split_counts[split] = split_counts.get(split, 0) + 1
        split_chromosomes.setdefault(split, set()).add(chromosome)
        try:
            start = int(row["start"])
            end = int(row["end"])
            length = int(row["length"])
        except ValueError:
            coordinate_errors += 1
            continue
        if start < 0 or end <= start:
            coordinate_errors += 1
        if end - start != length:
            length_errors += 1
        interval = (chromosome, start, end)
        if interval in intervals:
            duplicate_count += 1
        intervals.add(interval)
        if max_non_acgt_fraction is not None and row.get("non_acgt_fraction") not in (None, ""):
            try:
                non_acgt = float(row["non_acgt_fraction"])
            except ValueError:
                non_acgt_errors += 1
            else:
                max_seen_non_acgt = max(max_seen_non_acgt, non_acgt)
                if non_acgt > max_non_acgt_fraction:
                    non_acgt_errors += 1

    checks.extend(
        [
            _check("manifest.rows", bool(rows), f"n={len(rows)}"),
            _check("manifest.required_splits", required_splits.issubset(split_counts), json.dumps(split_counts, sort_keys=True)),
            _check(
                "manifest.min_windows_per_split",
                all(split_counts.get(split, 0) >= min_windows_per_split for split in required_splits),
                f"min={min_windows_per_split}; counts={json.dumps(split_counts, sort_keys=True)}",
            ),
            _check("manifest.coordinates.valid", coordinate_errors == 0, f"errors={coordinate_errors}"),
            _check("manifest.length.matches_coordinates", length_errors == 0, f"errors={length_errors}"),
            _check("manifest.duplicates", duplicate_count == 0, f"duplicates={duplicate_count}"),
            _check("manifest.chromosomes.disjoint", _chromosomes_disjoint(split_chromosomes), _format_split_chromosomes(split_chromosomes)),
        ]
    )
    if max_non_acgt_fraction is not None:
        checks.append(
            _check(
                "manifest.non_acgt_fraction.max",
                non_acgt_errors == 0,
                f"errors={non_acgt_errors}; max_seen={max_seen_non_acgt:.6f}; threshold={max_non_acgt_fraction}",
            )
        )
    return checks


def manifest_validation_passed(checks: list[ManifestCheck]) -> bool:
    return all(check.status == "pass" for check in checks)


def write_manifest_validation_json(path: str | Path, checks: list[ManifestCheck]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(name: str, ok: bool, detail: str) -> ManifestCheck:
    return ManifestCheck(name=name, status="pass" if ok else "fail", detail=detail)


def _chromosomes_disjoint(split_chromosomes: dict[str, set[str]]) -> bool:
    seen: set[str] = set()
    for chromosomes in split_chromosomes.values():
        if seen.intersection(chromosomes):
            return False
        seen.update(chromosomes)
    return True


def _format_split_chromosomes(split_chromosomes: dict[str, set[str]]) -> str:
    return json.dumps({split: sorted(chromosomes) for split, chromosomes in split_chromosomes.items()}, sort_keys=True)
