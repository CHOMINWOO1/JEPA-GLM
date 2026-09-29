from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VariantChromosomeHoldoutManifest:
    source_csv: str
    output_csv: str
    source_sha256: str
    holdout_chromosomes: list[str]
    n_rows: int
    label_counts: dict[str, int]
    max_rows: int
    balanced_labels: bool


def create_variant_chromosome_holdout(
    input_csv: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    holdout_chromosomes: list[str],
    chrom_column: str = "chrom",
    label_column: str = "label",
    max_rows: int | None = None,
    balanced_labels: bool = False,
) -> VariantChromosomeHoldoutManifest:
    source = Path(input_csv)
    output = Path(output_csv)
    manifest_path = Path(manifest_json)
    holdouts = [chrom.strip() for chrom in holdout_chromosomes if chrom.strip()]
    if not holdouts:
        raise ValueError("At least one holdout chromosome is required")

    rows: list[dict[str, str]] = []
    label_counts: dict[str, int] = {}
    per_label_target = max(1, max_rows // 2) if balanced_labels and max_rows is not None else None
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {chrom_column, label_column}
        if not required.issubset(set(fieldnames)):
            raise ValueError(f"Variant CSV must contain columns: {sorted(required)}")
        for row in reader:
            if row[chrom_column] not in holdouts:
                continue
            label = row[label_column]
            if per_label_target is not None and label_counts.get(label, 0) >= per_label_target:
                continue
            rows.append(dict(row))
            label_counts[label] = label_counts.get(label, 0) + 1
            if max_rows is not None and len(rows) >= max_rows:
                break
            if per_label_target is not None and label_counts.get("0", 0) >= per_label_target and label_counts.get("1", 0) >= per_label_target:
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = VariantChromosomeHoldoutManifest(
        source_csv=str(source),
        output_csv=str(output),
        source_sha256=_sha256_file(source),
        holdout_chromosomes=holdouts,
        n_rows=len(rows),
        label_counts={key: label_counts[key] for key in sorted(label_counts)},
        max_rows=int(max_rows or 0),
        balanced_labels=bool(balanced_labels),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
