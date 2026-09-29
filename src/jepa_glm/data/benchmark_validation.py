from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCheck:
    name: str
    status: str
    detail: str


def validate_labeled_benchmark(
    input_csv: str | Path,
    sequence_column: str = "sequence",
    label_column: str = "label",
    split_column: str = "split",
    chromosome_column: str = "chrom",
    required_splits: set[str] | None = None,
    min_examples_per_split: int = 1,
    require_all_labels_in_each_split: bool = True,
) -> list[BenchmarkCheck]:
    path = Path(input_csv)
    if not path.exists():
        return [BenchmarkCheck("benchmark.exists", "fail", str(path))]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required_columns = {sequence_column, label_column, split_column}
        checks = [_check("benchmark.columns", required_columns.issubset(columns), ",".join(sorted(columns)))]
        rows = list(reader)
    if not required_columns.issubset(columns):
        return checks

    required_splits = required_splits or {"train", "validation", "test"}
    split_counts = Counter(row[split_column] for row in rows)
    label_counts = Counter(row[label_column] for row in rows)
    labels = set(label_counts)
    labels_by_split: dict[str, set[str]] = defaultdict(set)
    sequences_by_split: dict[str, set[str]] = defaultdict(set)
    chromosomes_by_split: dict[str, set[str]] = defaultdict(set)
    empty_sequences = 0
    invalid_bases = 0
    total_bases = 0
    row_keys: Counter[tuple[str, str, str]] = Counter()
    has_chromosome = chromosome_column in columns
    for row in rows:
        split = row[split_column]
        sequence = row[sequence_column].strip().upper()
        label = row[label_column]
        if not sequence:
            empty_sequences += 1
        invalid_bases += sum(1 for base in sequence if base not in {"A", "C", "G", "T", "N"})
        total_bases += len(sequence)
        row_keys[(split, sequence, label)] += 1
        labels_by_split[split].add(label)
        sequences_by_split[split].add(sequence)
        if has_chromosome:
            chromosomes_by_split[split].add(_normalize_chromosome(row[chromosome_column]))

    leaking_sequences = _find_cross_split_sequences(sequences_by_split)
    unexpected_splits = sorted(set(split_counts) - required_splits)
    duplicate_rows = sum(count - 1 for count in row_keys.values() if count > 1)
    invalid_base_fraction = invalid_bases / total_bases if total_bases else 0.0
    checks.extend(
        [
            _check("benchmark.rows", bool(rows), f"n={len(rows)}"),
            _check("benchmark.required_splits", required_splits.issubset(split_counts), json.dumps(dict(sorted(split_counts.items())), sort_keys=True)),
            _check("benchmark.no_unexpected_splits", not unexpected_splits, json.dumps(unexpected_splits)),
            _check(
                "benchmark.min_examples_per_split",
                all(split_counts.get(split, 0) >= min_examples_per_split for split in required_splits),
                f"min={min_examples_per_split}; counts={json.dumps(dict(sorted(split_counts.items())), sort_keys=True)}",
            ),
            _check("benchmark.label_classes", len(labels) >= 2, json.dumps(dict(sorted(label_counts.items())), sort_keys=True)),
            _check("benchmark.empty_sequences", empty_sequences == 0, f"empty={empty_sequences}"),
            _check("benchmark.sequence_alphabet", invalid_bases == 0, f"invalid_bases={invalid_bases}; total_bases={total_bases}; fraction={invalid_base_fraction:.6g}"),
            _check("benchmark.duplicate_rows", duplicate_rows == 0, f"duplicates={duplicate_rows}"),
            _check("benchmark.sequence_leakage", not leaking_sequences, f"leaking_sequences={len(leaking_sequences)}"),
        ]
    )
    if has_chromosome:
        overlapping_chromosomes = _find_cross_split_values(chromosomes_by_split)
        checks.append(
            _check(
                "benchmark.chromosomes_disjoint",
                not overlapping_chromosomes,
                json.dumps({split: sorted(values) for split, values in chromosomes_by_split.items()}, sort_keys=True),
            )
        )
    if require_all_labels_in_each_split:
        missing_by_split = {
            split: sorted(labels - labels_by_split.get(split, set()))
            for split in sorted(required_splits)
            if labels - labels_by_split.get(split, set())
        }
        checks.append(_check("benchmark.label_coverage_by_split", not missing_by_split, json.dumps(missing_by_split, sort_keys=True)))
    return checks


def validate_benchmark_manifest(
    manifest_json: str | Path,
    require_chromosome_disjoint: bool = True,
) -> list[BenchmarkCheck]:
    path = Path(manifest_json)
    if not path.exists():
        return [BenchmarkCheck("benchmark_manifest.exists", "fail", str(path))]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [BenchmarkCheck("benchmark_manifest.readable", "fail", str(error))]
    required = {"source_csv", "output_csv", "source_sha256", "split_strategy", "splits"}
    checks = [
        BenchmarkCheck("benchmark_manifest.exists", "pass", str(path)),
        _check("benchmark_manifest.required_fields", required.issubset(payload), f"missing={sorted(required - set(payload))}"),
    ]
    if not required.issubset(payload):
        return checks
    split_strategy = str(payload.get("split_strategy", ""))
    splits = payload.get("splits", [])
    split_names = {str(item.get("split", "")) for item in splits if isinstance(item, dict)}
    checks.extend(
        [
            _check("benchmark_manifest.source_sha256", len(str(payload.get("source_sha256", ""))) == 64, str(payload.get("source_sha256", ""))),
            _check("benchmark_manifest.source_csv.exists", Path(str(payload.get("source_csv", ""))).exists(), str(payload.get("source_csv", ""))),
            _check("benchmark_manifest.output_csv.exists", Path(str(payload.get("output_csv", ""))).exists(), str(payload.get("output_csv", ""))),
            _check("benchmark_manifest.split_strategy", split_strategy in {"chromosome_disjoint", "seeded_stratified_random"}, split_strategy),
            _check("benchmark_manifest.required_splits", {"train", "validation", "test"}.issubset(split_names), json.dumps(sorted(split_names))),
            _check("benchmark_manifest.split_counts", _splits_have_examples(splits), _format_split_sizes(splits)),
            _check("benchmark_manifest.label_counts", _splits_have_label_counts(splits), _format_label_counts(splits)),
            _check("benchmark_manifest.label_coverage_by_split", _manifest_splits_cover_all_labels(splits), _format_label_counts(splits)),
        ]
    )
    if Path(str(payload.get("output_csv", ""))).exists():
        checks.extend(_manifest_matches_output_csv(payload))
    if require_chromosome_disjoint:
        checks.extend(
            [
                _check("benchmark_manifest.chromosome_column", bool(payload.get("chromosome_column")), str(payload.get("chromosome_column"))),
                _check("benchmark_manifest.chromosome_strategy", split_strategy == "chromosome_disjoint", split_strategy),
                _check("benchmark_manifest.chromosomes_present", _splits_have_chromosomes(splits), _format_split_chromosomes(splits)),
                _check("benchmark_manifest.chromosomes_disjoint", _manifest_chromosomes_disjoint(splits), _format_split_chromosomes(splits)),
            ]
        )
    return checks


def benchmark_validation_passed(checks: list[BenchmarkCheck]) -> bool:
    return all(check.status == "pass" for check in checks)


def write_benchmark_validation_json(path: str | Path, checks: list[BenchmarkCheck]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _splits_have_examples(splits) -> bool:
    return isinstance(splits, list) and all(isinstance(item, dict) and int(item.get("n_examples", 0)) > 0 for item in splits)


def _splits_have_label_counts(splits) -> bool:
    return isinstance(splits, list) and all(isinstance(item, dict) and bool(item.get("label_counts")) for item in splits)


def _manifest_splits_cover_all_labels(splits) -> bool:
    if not isinstance(splits, list):
        return False
    label_sets = []
    for item in splits:
        if not isinstance(item, dict) or not isinstance(item.get("label_counts"), dict):
            return False
        labels = {str(label) for label, count in item["label_counts"].items() if int(count) > 0}
        if not labels:
            return False
        label_sets.append(labels)
    if not label_sets:
        return False
    all_labels = set().union(*label_sets)
    return all(labels == all_labels for labels in label_sets)


def _splits_have_chromosomes(splits) -> bool:
    return isinstance(splits, list) and all(isinstance(item, dict) and bool(item.get("chromosomes")) for item in splits)


def _manifest_chromosomes_disjoint(splits) -> bool:
    chromosomes_by_split = {
        str(item.get("split", "")): {_normalize_chromosome(str(chrom)) for chrom in item.get("chromosomes", [])}
        for item in splits
        if isinstance(item, dict)
    }
    return not _find_cross_split_values(chromosomes_by_split)


def _format_split_sizes(splits) -> str:
    if not isinstance(splits, list):
        return "splits_not_list"
    return json.dumps({str(item.get("split", "")): int(item.get("n_examples", 0)) for item in splits if isinstance(item, dict)}, sort_keys=True)


def _format_label_counts(splits) -> str:
    if not isinstance(splits, list):
        return "splits_not_list"
    return json.dumps({str(item.get("split", "")): item.get("label_counts", {}) for item in splits if isinstance(item, dict)}, sort_keys=True)


def _format_split_chromosomes(splits) -> str:
    if not isinstance(splits, list):
        return "splits_not_list"
    return json.dumps({str(item.get("split", "")): item.get("chromosomes", []) for item in splits if isinstance(item, dict)}, sort_keys=True)


def _manifest_matches_output_csv(payload: dict) -> list[BenchmarkCheck]:
    output_csv = Path(str(payload.get("output_csv", "")))
    chromosome_column = str(payload.get("chromosome_column", "") or "chrom")
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    split_column = "split"
    label_column = "label"
    if not {split_column, label_column}.issubset(columns):
        return [
            _check(
                "benchmark_manifest.output_csv.required_columns",
                False,
                f"missing={sorted({split_column, label_column} - columns)}",
            )
        ]
    csv_split_counts = Counter(row[split_column] for row in rows)
    csv_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    csv_chromosomes: dict[str, set[str]] = defaultdict(set)
    has_chromosome_column = chromosome_column in columns
    for row in rows:
        split = row[split_column]
        csv_label_counts[split][row[label_column]] += 1
        if has_chromosome_column:
            csv_chromosomes[split].add(_normalize_chromosome(row[chromosome_column]))
    manifest_split_counts = {
        str(item.get("split", "")): int(item.get("n_examples", 0))
        for item in payload.get("splits", [])
        if isinstance(item, dict)
    }
    manifest_label_counts = {
        str(item.get("split", "")): {str(label): int(count) for label, count in dict(item.get("label_counts", {})).items()}
        for item in payload.get("splits", [])
        if isinstance(item, dict)
    }
    manifest_chromosomes = {
        str(item.get("split", "")): {_normalize_chromosome(str(chrom)) for chrom in item.get("chromosomes", [])}
        for item in payload.get("splits", [])
        if isinstance(item, dict)
    }
    checks = [
        _check(
            "benchmark_manifest.output_csv.split_counts_match",
            dict(csv_split_counts) == manifest_split_counts,
            f"csv={json.dumps(dict(sorted(csv_split_counts.items())), sort_keys=True)}; manifest={json.dumps(manifest_split_counts, sort_keys=True)}",
        ),
        _check(
            "benchmark_manifest.output_csv.label_counts_match",
            {split: dict(counts) for split, counts in csv_label_counts.items()} == manifest_label_counts,
            f"csv={json.dumps({split: dict(counts) for split, counts in sorted(csv_label_counts.items())}, sort_keys=True)}; manifest={json.dumps(manifest_label_counts, sort_keys=True)}",
        ),
    ]
    if payload.get("chromosome_column"):
        checks.append(_check("benchmark_manifest.output_csv.chromosome_column", has_chromosome_column, chromosome_column))
        if has_chromosome_column:
            checks.append(
                _check(
                    "benchmark_manifest.output_csv.chromosomes_match",
                    {split: values for split, values in csv_chromosomes.items()} == manifest_chromosomes,
                    f"csv={json.dumps({split: sorted(values) for split, values in csv_chromosomes.items()}, sort_keys=True)}; manifest={json.dumps({split: sorted(values) for split, values in manifest_chromosomes.items()}, sort_keys=True)}",
                )
            )
    return checks


def _find_cross_split_sequences(sequences_by_split: dict[str, set[str]]) -> set[str]:
    return _find_cross_split_values(sequences_by_split)


def _find_cross_split_values(values_by_split: dict[str, set[str]]) -> set[str]:
    owners: dict[str, str] = {}
    leaking = set()
    for split, values in values_by_split.items():
        for value in values:
            previous = owners.get(value)
            if previous is not None and previous != split:
                leaking.add(value)
            owners[value] = split
    return leaking


def _normalize_chromosome(chromosome: str) -> str:
    chromosome = chromosome.strip()
    if not chromosome:
        return chromosome
    return chromosome if chromosome.startswith("chr") else f"chr{chromosome}"


def _check(name: str, ok: bool, detail: str) -> BenchmarkCheck:
    return BenchmarkCheck(name=name, status="pass" if ok else "fail", detail=detail)
