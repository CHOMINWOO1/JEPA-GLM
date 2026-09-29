from __future__ import annotations

import csv
import hashlib
import heapq
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from jepa_glm.data.converters import variant_table_to_sequences


@dataclass(frozen=True)
class VariantReplicationPanelManifest:
    schema_version: int
    protocol_json: str
    protocol_sha256: str
    source_csv: str
    source_sha256: str
    fasta: str
    raw_output_csv: str
    raw_output_sha256: str
    sequence_output_csv: str
    sequence_output_sha256: str
    selection_seed: int
    selection_algorithm: str
    requested_rows: int
    selected_rows: int
    selected_label_counts: dict[str, int]
    selected_chromosome_counts: dict[str, int]
    source_rows_scanned: int
    source_label_counts: dict[str, int]
    excluded_prefix_per_label: int
    excluded_prefix_rows: int
    included_chromosomes: list[str]
    excluded_nonincluded_chromosome_rows: int
    excluded_chromosomes: list[str]
    excluded_chromosome_rows: int
    eligible_label_counts: dict[str, int]
    duplicate_selected_variant_keys: int
    sequence_overlap_counts: dict[str, int]
    flank: int


def create_variant_replication_panel(
    source_csv: str | Path,
    fasta_path: str | Path,
    raw_output_csv: str | Path,
    sequence_output_csv: str | Path,
    manifest_json: str | Path,
    protocol_json: str | Path,
    *,
    n_rows: int,
    seed: int,
    exclude_prefix_per_label: int,
    exclude_chromosomes: Iterable[str],
    include_chromosomes: Iterable[str] = (),
    exclude_sequence_csvs: Iterable[str | Path] = (),
    flank: int = 64,
) -> VariantReplicationPanelManifest:
    if n_rows <= 0 or n_rows % 2:
        raise ValueError("n_rows must be a positive even number")
    source = Path(source_csv)
    fasta = Path(fasta_path)
    raw_output = Path(raw_output_csv)
    sequence_output = Path(sequence_output_csv)
    manifest_path = Path(manifest_json)
    protocol = Path(protocol_json)
    excluded_chromosomes = sorted({item.strip() for item in exclude_chromosomes if item.strip()})
    included_chromosomes = sorted({item.strip() for item in include_chromosomes if item.strip()})
    per_label = n_rows // 2
    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {"0": [], "1": []}
    selected_keys: dict[str, set[str]] = {"0": set(), "1": set()}
    prefix_counts: Counter[str] = Counter()
    prefix_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    source_rows = 0
    excluded_prefix_rows = 0
    excluded_nonincluded_chromosome_rows = 0
    excluded_chromosome_rows = 0

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"chrom", "pos", "ref", "alt", "label"}
        if not required.issubset(set(fieldnames)):
            raise ValueError(f"Variant CSV must contain columns: {sorted(required)}")
        for row in reader:
            source_rows += 1
            label = str(row["label"])
            if label not in heaps:
                continue
            source_counts[label] += 1
            key = _variant_key(row)
            if prefix_counts[label] < exclude_prefix_per_label:
                prefix_counts[label] += 1
                prefix_keys.add(key)
                excluded_prefix_rows += 1
                continue
            if key in prefix_keys:
                excluded_prefix_rows += 1
                continue
            if row["chrom"] in excluded_chromosomes:
                excluded_chromosome_rows += 1
                continue
            if included_chromosomes and row["chrom"] not in included_chromosomes:
                excluded_nonincluded_chromosome_rows += 1
                continue
            eligible_counts[label] += 1
            rank = _selection_rank(seed, key)
            heap = heaps[label]
            if key in selected_keys[label]:
                continue
            candidate = (-rank, key, dict(row))
            if len(heap) < per_label:
                heapq.heappush(heap, candidate)
                selected_keys[label].add(key)
            elif rank < -heap[0][0]:
                removed = heapq.heapreplace(heap, candidate)
                selected_keys[label].remove(removed[1])
                selected_keys[label].add(key)

    if any(len(heap) != per_label for heap in heaps.values()):
        raise ValueError(f"Insufficient eligible variants: { {label: len(heap) for label, heap in heaps.items()} }")
    ranked_rows = []
    for label, heap in heaps.items():
        for negative_rank, key, row in heap:
            ranked_rows.append((-negative_rank, key, label, row))
    ranked_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    selected_rows = [item[3] for item in ranked_rows]
    duplicate_keys = len(selected_rows) - len({_variant_key(row) for row in selected_rows})
    if duplicate_keys:
        raise ValueError(f"Selected panel contains duplicate variant keys: {duplicate_keys}")

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    conversion = variant_table_to_sequences(
        raw_output,
        fasta,
        sequence_output,
        flank=flank,
        summary_json=None,
        max_rows=None,
        balanced_labels=False,
    )
    if conversion.n_rows != n_rows:
        raise ValueError(f"Converted sequence count mismatch: {conversion.n_rows} != {n_rows}")
    sequence_pairs = _read_sequence_pairs(sequence_output)
    overlap_counts = {
        Path(path).name: len(sequence_pairs & _read_sequence_pairs(path))
        for path in exclude_sequence_csvs
    }
    if any(overlap_counts.values()):
        raise ValueError(f"Replication panel overlaps excluded sequence panels: {overlap_counts}")

    manifest = VariantReplicationPanelManifest(
        schema_version=1,
        protocol_json=str(protocol),
        protocol_sha256=_sha256_file(protocol),
        source_csv=str(source),
        source_sha256=_sha256_file(source),
        fasta=str(fasta),
        raw_output_csv=str(raw_output),
        raw_output_sha256=_sha256_file(raw_output),
        sequence_output_csv=str(sequence_output),
        sequence_output_sha256=_sha256_file(sequence_output),
        selection_seed=seed,
        selection_algorithm="smallest_sha256(seed\\tchrom\\tpos\\tref\\talt)",
        requested_rows=n_rows,
        selected_rows=len(selected_rows),
        selected_label_counts=dict(sorted(Counter(str(row["label"]) for row in selected_rows).items())),
        selected_chromosome_counts=dict(sorted(Counter(row["chrom"] for row in selected_rows).items())),
        source_rows_scanned=source_rows,
        source_label_counts=dict(sorted(source_counts.items())),
        excluded_prefix_per_label=exclude_prefix_per_label,
        excluded_prefix_rows=excluded_prefix_rows,
        included_chromosomes=included_chromosomes,
        excluded_nonincluded_chromosome_rows=excluded_nonincluded_chromosome_rows,
        excluded_chromosomes=excluded_chromosomes,
        excluded_chromosome_rows=excluded_chromosome_rows,
        eligible_label_counts=dict(sorted(eligible_counts.items())),
        duplicate_selected_variant_keys=duplicate_keys,
        sequence_overlap_counts=overlap_counts,
        flank=flank,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _variant_key(row: dict[str, Any]) -> str:
    return "\t".join(
        [
            str(row["chrom"]).strip(),
            str(row["pos"]).strip(),
            str(row["ref"]).strip().upper(),
            str(row["alt"]).strip().upper(),
        ]
    )


def _selection_rank(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}\t{key}".encode("ascii")).digest()
    return int.from_bytes(digest, "big")


def _read_sequence_pairs(path_like: str | Path) -> set[tuple[str, str]]:
    path = Path(path_like)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ref_sequence", "alt_sequence"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Sequence CSV must contain columns: {sorted(required)}")
        return {(row["ref_sequence"], row["alt_sequence"]) for row in reader}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
