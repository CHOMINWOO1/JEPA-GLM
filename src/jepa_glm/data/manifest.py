from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from jepa_glm.data.fasta_dataset import DNA_ALPHABET
from jepa_glm.data.splits import DEFAULT_CHROMOSOME_SPLIT, assert_disjoint_splits


@dataclass(frozen=True)
class FastaRecord:
    name: str
    sequence: str


def iter_fasta_records(path: str | Path):
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield FastaRecord(name=name, sequence="".join(chunks).upper())
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield FastaRecord(name=name, sequence="".join(chunks).upper())


def split_for_chromosome(chromosome: str, splits: dict[str, list[str]] | None = None) -> str | None:
    splits = splits or DEFAULT_CHROMOSOME_SPLIT
    assert_disjoint_splits(splits)
    normalized = chromosome if chromosome.startswith("chr") else f"chr{chromosome}"
    for split, chromosomes in splits.items():
        if normalized in chromosomes or chromosome in chromosomes:
            return split
    return None


def write_window_manifest(
    fasta_path: str | Path,
    output_csv: str | Path,
    window_size: int,
    stride: int,
    splits: dict[str, list[str]] | None = None,
    drop_non_acgt_fraction: float = 0.5,
) -> dict[str, int]:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "test": 0, "skipped": 0}
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "chromosome", "start", "end", "length", "non_acgt_fraction"],
        )
        writer.writeheader()
        for record in iter_fasta_records(fasta_path):
            split = split_for_chromosome(record.name, splits)
            if split is None:
                counts["skipped"] += 1
                continue
            for start in range(0, max(0, len(record.sequence) - window_size + 1), stride):
                window = record.sequence[start : start + window_size]
                non_acgt = sum(base not in DNA_ALPHABET for base in window) / max(1, len(window))
                if non_acgt > drop_non_acgt_fraction:
                    counts["skipped"] += 1
                    continue
                writer.writerow(
                    {
                        "split": split,
                        "chromosome": record.name,
                        "start": start,
                        "end": start + window_size,
                        "length": window_size,
                        "non_acgt_fraction": f"{non_acgt:.6f}",
                    }
                )
                counts[split] += 1
    return counts
