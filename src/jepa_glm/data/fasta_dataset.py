from __future__ import annotations

import random
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from jepa_glm.data.tokenizer import SequenceTokenizer


DNA_ALPHABET = "ACGT"


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


class SyntheticDnaDataset(Dataset[torch.Tensor]):
    def __init__(self, num_sequences: int, sequence_length: int, tokenizer: SequenceTokenizer, seed: int = 7) -> None:
        rng = random.Random(seed)
        self.samples = [
            tokenizer.encode("".join(rng.choice(DNA_ALPHABET) for _ in range(sequence_length)))
            for _ in range(num_sequences)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


class FastaWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, fasta_path: str | Path, window_size: int, tokenizer: SequenceTokenizer, stride: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride or window_size
        self.sequences = self._load_windows(Path(fasta_path))

    def _load_windows(self, path: Path) -> list[torch.Tensor]:
        windows: list[torch.Tensor] = []
        current: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    windows.extend(self._window_sequence("".join(current)))
                    current = []
                else:
                    current.append(line)
        windows.extend(self._window_sequence("".join(current)))
        return windows

    def _window_sequence(self, sequence: str) -> list[torch.Tensor]:
        cleaned = "".join(base if base.upper() in DNA_ALPHABET else "N" for base in sequence.upper())
        return [
            self.tokenizer.encode(cleaned[start : start + self.window_size], max_length=self.window_size)
            for start in range(0, max(0, len(cleaned) - self.window_size + 1), self.stride)
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequences[index]


class ManifestWindowDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        fasta_path: str | Path,
        manifest_csv: str | Path,
        tokenizer: SequenceTokenizer,
        split: str = "train",
        max_length: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = self._load_records(Path(fasta_path))
        self.windows = self._load_manifest(Path(manifest_csv), split, max_windows=max_windows)
        if not self.windows:
            raise ValueError(f"No windows found for split '{split}' in {manifest_csv}")

    def _load_records(self, path: Path) -> dict[str, str]:
        records: dict[str, str] = {}
        name: str | None = None
        chunks: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
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

    def _load_manifest(self, path: Path, split: str, max_windows: int | None = None) -> list[tuple[str, int, int]]:
        windows: list[tuple[str, int, int]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "chromosome", "start", "end"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest must contain columns: {sorted(required)}")
            for row in reader:
                if row["split"] != split:
                    continue
                chromosome = row["chromosome"]
                if chromosome not in self.records:
                    raise ValueError(f"Manifest chromosome '{chromosome}' is missing from FASTA")
                windows.append((chromosome, int(row["start"]), int(row["end"])))
                if max_windows is not None and len(windows) >= max_windows:
                    break
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.encode_window(index, reverse_complement_view=False)

    def encode_window(self, index: int, *, reverse_complement_view: bool = False) -> torch.Tensor:
        chromosome, start, end = self.windows[index]
        sequence = self.records[chromosome][start:end]
        cleaned = "".join(base if base in DNA_ALPHABET else "N" for base in sequence.upper())
        if reverse_complement_view:
            cleaned = reverse_complement(cleaned)
        return self.tokenizer.encode(cleaned, max_length=self.max_length or len(cleaned))


class ReverseComplementManifestSubset(Dataset[torch.Tensor]):
    def __init__(self, subset: Subset[torch.Tensor], probability: float) -> None:
        if not isinstance(subset.dataset, ManifestWindowDataset):
            raise TypeError("ReverseComplementManifestSubset requires a ManifestWindowDataset subset")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Reverse-complement probability must be in [0, 1]")
        self.subset = subset
        self.probability = probability
        self.access_count = 0
        self.reverse_complement_count = 0

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> torch.Tensor:
        source_index = int(self.subset.indices[index])
        use_reverse_complement = random.random() < self.probability
        self.access_count += 1
        self.reverse_complement_count += int(use_reverse_complement)
        dataset = self.subset.dataset
        return dataset.encode_window(source_index, reverse_complement_view=use_reverse_complement)

    def augmentation_stats(self) -> dict[str, int | float]:
        return {
            "reverse_complement_probability": self.probability,
            "augmentation_accesses": self.access_count,
            "reverse_complement_accesses": self.reverse_complement_count,
            "observed_reverse_complement_fraction": (
                self.reverse_complement_count / self.access_count if self.access_count else 0.0
            ),
        }
