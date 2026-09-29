from __future__ import annotations

from pathlib import Path


def load_fasta_dict(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks).upper()
    return records


def fetch_interval(records: dict[str, str], chromosome: str, start: int, end: int, pad_base: str = "N") -> str:
    if end <= start:
        raise ValueError("Interval end must be greater than start")
    if chromosome not in records:
        raise ValueError(f"Chromosome '{chromosome}' not found in reference FASTA")
    sequence = records[chromosome]
    left_pad = max(0, -start)
    right_pad = max(0, end - len(sequence))
    clipped_start = max(0, start)
    clipped_end = min(len(sequence), end)
    return (pad_base * left_pad + sequence[clipped_start:clipped_end] + pad_base * right_pad).upper()
