from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceBenchmarkCheck:
    name: str
    status: str
    detail: str


def audit_labeled_sequence_source(
    input_csv: str | Path,
    sequence_column: str = "sequence",
    label_column: str = "label",
    min_rows: int = 10,
) -> list[SourceBenchmarkCheck]:
    path = Path(input_csv)
    if not path.exists():
        return [SourceBenchmarkCheck("source_benchmark.exists", "fail", str(path))]
    rows, columns = _read_csv(path)
    required = {sequence_column, label_column}
    checks = [
        SourceBenchmarkCheck("source_benchmark.exists", "pass", str(path)),
        _check("source_benchmark.columns", required.issubset(columns), ",".join(sorted(columns))),
    ]
    if not required.issubset(columns):
        return checks
    sequences = [row.get(sequence_column, "").strip().upper() for row in rows]
    labels = [row.get(label_column, "").strip() for row in rows]
    label_counts = Counter(labels)
    duplicate_count = len(sequences) - len(set(sequences))
    invalid_sequences = [seq for seq in sequences if not seq or any(base not in {"A", "C", "G", "T", "N"} for base in seq)]
    checks.extend(
        [
            _check("source_benchmark.rows", len(rows) >= min_rows, f"n={len(rows)}; min={min_rows}"),
            _check("source_benchmark.label_classes", len(set(labels)) >= 2, json.dumps(dict(sorted(label_counts.items())), sort_keys=True)),
            _check("source_benchmark.empty_labels", all(labels), f"empty={sum(1 for label in labels if not label)}"),
            _check("source_benchmark.sequence_alphabet", not invalid_sequences, f"invalid={len(invalid_sequences)}"),
            _check("source_benchmark.duplicate_sequences", duplicate_count == 0, f"duplicates={duplicate_count}"),
        ]
    )
    checks.append(_source_level(checks))
    return checks


def audit_variant_source(
    input_csv: str | Path,
    chrom_column: str = "chrom",
    pos_column: str = "pos",
    ref_column: str = "ref",
    alt_column: str = "alt",
    label_column: str = "label",
    min_rows: int = 10,
) -> list[SourceBenchmarkCheck]:
    path = Path(input_csv)
    if not path.exists():
        return [SourceBenchmarkCheck("source_variant.exists", "fail", str(path))]
    rows, columns = _read_csv(path)
    required = {chrom_column, pos_column, ref_column, alt_column, label_column}
    checks = [
        SourceBenchmarkCheck("source_variant.exists", "pass", str(path)),
        _check("source_variant.columns", required.issubset(columns), ",".join(sorted(columns))),
    ]
    if not required.issubset(columns):
        return checks
    labels = [row.get(label_column, "").strip() for row in rows]
    label_counts = Counter(labels)
    invalid_pos = 0
    invalid_alleles = 0
    duplicates = 0
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        chrom = row.get(chrom_column, "").strip()
        pos = row.get(pos_column, "").strip()
        ref = row.get(ref_column, "").strip().upper()
        alt = row.get(alt_column, "").strip().upper()
        try:
            if int(pos) <= 0:
                invalid_pos += 1
        except ValueError:
            invalid_pos += 1
        if not ref or not alt or ref == alt or any(base not in {"A", "C", "G", "T"} for base in ref + alt):
            invalid_alleles += 1
        key = (chrom, pos, ref, alt)
        if key in seen:
            duplicates += 1
        seen.add(key)
    checks.extend(
        [
            _check("source_variant.rows", len(rows) >= min_rows, f"n={len(rows)}; min={min_rows}"),
            _check("source_variant.label_classes", len(set(labels)) >= 2, json.dumps(dict(sorted(label_counts.items())), sort_keys=True)),
            _check("source_variant.empty_labels", all(labels), f"empty={sum(1 for label in labels if not label)}"),
            _check("source_variant.positions", invalid_pos == 0, f"invalid={invalid_pos}"),
            _check("source_variant.alleles", invalid_alleles == 0, f"invalid={invalid_alleles}"),
            _check("source_variant.duplicates", duplicates == 0, f"duplicates={duplicates}"),
        ]
    )
    checks.append(_source_level(checks, prefix="source_variant"))
    return checks


def source_benchmark_passed(checks: list[SourceBenchmarkCheck]) -> bool:
    return any(check.name.endswith(".level") and check.status == "pass" for check in checks)


def write_source_benchmark_audit_json(path: str | Path, checks: list[SourceBenchmarkCheck]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        return list(reader), columns


def _source_level(checks: list[SourceBenchmarkCheck], prefix: str = "source_benchmark") -> SourceBenchmarkCheck:
    failed = [check.name for check in checks if check.status == "fail"]
    if failed:
        return SourceBenchmarkCheck(f"{prefix}.level", "fail", f"incomplete_source: {failed}")
    return SourceBenchmarkCheck(f"{prefix}.level", "pass", "source_ready")


def _check(name: str, ok: bool, detail: str) -> SourceBenchmarkCheck:
    return SourceBenchmarkCheck(name, "pass" if ok else "fail", detail)
