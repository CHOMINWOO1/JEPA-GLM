from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SchemaContract:
    name: str
    required_columns: tuple[str, ...]
    recommended_columns: tuple[str, ...] = ()


SCHEMA_CONTRACTS = {
    "labeled_sequence": SchemaContract("labeled_sequence", ("sequence", "label"), ("split",)),
    "variant_sequence": SchemaContract("variant_sequence", ("ref_sequence", "alt_sequence", "label")),
    "variant_table": SchemaContract("variant_table", ("chrom", "pos", "ref", "alt", "label")),
    "bed3": SchemaContract("bed3", ("chrom", "start", "end")),
}


def validate_csv_schema(path: str | Path, schema_name: str) -> dict[str, object]:
    if schema_name == "bed3":
        return _validate_bed3(path)
    if schema_name not in SCHEMA_CONTRACTS:
        raise ValueError(f"Unknown schema contract: {schema_name}")
    contract = SCHEMA_CONTRACTS[schema_name]
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        row_count = sum(1 for _ in reader)
    missing = [column for column in contract.required_columns if column not in columns]
    return {
        "schema": schema_name,
        "path": str(path),
        "valid": not missing,
        "columns": list(columns),
        "missing_required_columns": missing,
        "recommended_columns": list(contract.recommended_columns),
        "n_rows": row_count,
    }


def _validate_bed3(path: str | Path) -> dict[str, object]:
    n_rows = 0
    invalid_rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            n_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                invalid_rows += 1
                continue
            try:
                int(parts[1])
                int(parts[2])
            except ValueError:
                invalid_rows += 1
    return {
        "schema": "bed3",
        "path": str(path),
        "valid": invalid_rows == 0,
        "columns": ["chrom", "start", "end"],
        "missing_required_columns": [],
        "invalid_rows": invalid_rows,
        "n_rows": n_rows,
    }
