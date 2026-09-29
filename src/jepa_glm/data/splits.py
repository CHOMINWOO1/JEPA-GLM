from __future__ import annotations


DEFAULT_CHROMOSOME_SPLIT = {
    "train": [f"chr{i}" for i in range(1, 20)],
    "validation": ["chr20", "chr21"],
    "test": ["chr22"],
}


def assert_disjoint_splits(splits: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for name, chromosomes in splits.items():
        overlap = seen.intersection(chromosomes)
        if overlap:
            raise ValueError(f"Chromosome split {name} overlaps on {sorted(overlap)}")
        seen.update(chromosomes)
