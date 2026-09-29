from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from jepa_glm.data.manifest import iter_fasta_records


@dataclass(frozen=True)
class ReferenceFastaCheck:
    name: str
    status: str
    detail: str


DEFAULT_REQUIRED_CHROMOSOMES = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
VALID_BASES = set("ACGTN")


def validate_reference_fasta(
    fasta_path: str | Path,
    *,
    required_chromosomes: set[str] | None = None,
    min_total_bases: int = 1,
    max_non_acgtn_fraction: float = 0.01,
) -> list[ReferenceFastaCheck]:
    path = Path(fasta_path)
    if not path.exists():
        return [ReferenceFastaCheck("reference_fasta.exists", "fail", str(path))]
    required_chromosomes = required_chromosomes or DEFAULT_REQUIRED_CHROMOSOMES
    records = list(iter_fasta_records(path))
    names = [record.name for record in records]
    duplicate_names = _duplicates(names)
    base_counts: Counter[str] = Counter()
    empty_records = 0
    invalid_records = 0
    for record in records:
        if not record.sequence:
            empty_records += 1
        record_counts = Counter(record.sequence)
        base_counts.update(record_counts)
        if set(record_counts) - VALID_BASES:
            invalid_records += 1
    total_bases = sum(base_counts.values())
    invalid_bases = sum(count for base, count in base_counts.items() if base not in VALID_BASES)
    non_acgtn_fraction = invalid_bases / max(1, total_bases)
    observed = set(names)
    missing_chromosomes = sorted(required_chromosomes - observed)
    checks = [
        ReferenceFastaCheck("reference_fasta.exists", "pass", str(path)),
        _check("reference_fasta.records", bool(records), f"n={len(records)}"),
        _check("reference_fasta.record_names_unique", not duplicate_names, f"duplicates={duplicate_names}"),
        _check("reference_fasta.required_chromosomes", not missing_chromosomes, f"missing={missing_chromosomes}"),
        _check("reference_fasta.empty_records", empty_records == 0, f"empty={empty_records}"),
        _check("reference_fasta.total_bases", total_bases >= min_total_bases, f"bases={total_bases}; min={min_total_bases}"),
        _check("reference_fasta.alphabet.records", invalid_records == 0, f"invalid_records={invalid_records}"),
        _check(
            "reference_fasta.alphabet.non_acgtn_fraction",
            non_acgtn_fraction <= max_non_acgtn_fraction,
            f"fraction={non_acgtn_fraction:.6g}; max={max_non_acgtn_fraction:.6g}",
        ),
    ]
    checks.append(_level(checks))
    return checks


def reference_fasta_passed(checks: list[ReferenceFastaCheck]) -> bool:
    return any(check.name == "reference_fasta.level" and check.status == "pass" for check in checks)


def write_reference_fasta_validation_json(path: str | Path, checks: list[ReferenceFastaCheck]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _level(checks: list[ReferenceFastaCheck]) -> ReferenceFastaCheck:
    failed = [check.name for check in checks if check.status == "fail"]
    if failed:
        return ReferenceFastaCheck("reference_fasta.level", "fail", f"reference_fasta_not_ready: {failed}")
    return ReferenceFastaCheck("reference_fasta.level", "pass", "reference_fasta_ready")


def _check(name: str, ok: bool, detail: str) -> ReferenceFastaCheck:
    return ReferenceFastaCheck(name, "pass" if ok else "fail", detail)
