from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlleleSwapManifest:
    source_csv: str
    output_csv: str
    reference_column: str
    alternate_column: str
    label_column: str
    source_sha256: str
    n_rows: int
    label_policy: str


@dataclass(frozen=True)
class AlleleSwapCheck:
    name: str
    status: str
    detail: str


def create_allele_swap_control(
    input_csv: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    reference_column: str = "ref_sequence",
    alternate_column: str = "alt_sequence",
    label_column: str = "label",
    label_policy: str = "preserve",
) -> AlleleSwapManifest:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    manifest_path = Path(manifest_json)
    rows = _read_rows(input_path, {reference_column, alternate_column, label_column})
    swapped = []
    for row in rows:
        new_row = dict(row)
        new_row[reference_column] = row[alternate_column]
        new_row[alternate_column] = row[reference_column]
        if label_policy == "invert_binary":
            new_row[label_column] = _invert_binary_label(row[label_column])
        elif label_policy != "preserve":
            raise ValueError("label_policy must be 'preserve' or 'invert_binary'")
        swapped.append(new_row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [reference_column, alternate_column, label_column]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(swapped)
    manifest = AlleleSwapManifest(
        source_csv=str(input_path),
        output_csv=str(output_path),
        reference_column=reference_column,
        alternate_column=alternate_column,
        label_column=label_column,
        source_sha256=_sha256_file(input_path),
        n_rows=len(swapped),
        label_policy=label_policy,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_allele_swap_control(
    source_csv: str | Path,
    swapped_csv: str | Path,
    manifest_json: str | Path,
) -> list[AlleleSwapCheck]:
    source_path = Path(source_csv)
    swapped_path = Path(swapped_csv)
    manifest_path = Path(manifest_json)
    checks = [
        _check("allele_swap.source.exists", source_path.exists(), str(source_path)),
        _check("allele_swap.swapped.exists", swapped_path.exists(), str(swapped_path)),
        _check("allele_swap.manifest.exists", manifest_path.exists(), str(manifest_path)),
    ]
    if any(check.status == "fail" for check in checks):
        checks.append(_level(checks))
        return checks
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref_col = str(manifest.get("reference_column", "ref_sequence"))
    alt_col = str(manifest.get("alternate_column", "alt_sequence"))
    label_col = str(manifest.get("label_column", "label"))
    label_policy = str(manifest.get("label_policy", "preserve"))
    source_rows = _read_rows(source_path, {ref_col, alt_col, label_col})
    swapped_rows = _read_rows(swapped_path, {ref_col, alt_col, label_col})
    checks.extend(
        [
            _check("allele_swap.manifest.source_hash", manifest.get("source_sha256") == _sha256_file(source_path), str(source_path)),
            _check("allele_swap.row_count", len(source_rows) == len(swapped_rows) == int(manifest.get("n_rows", -1)), f"source={len(source_rows)}; swapped={len(swapped_rows)}; manifest={manifest.get('n_rows')}"),
            _check("allele_swap.label_policy", label_policy in {"preserve", "invert_binary"}, label_policy),
        ]
    )
    mismatches = 0
    for source, swapped in zip(source_rows, swapped_rows):
        expected_label = source[label_col] if label_policy == "preserve" else _invert_binary_label(source[label_col])
        if swapped[ref_col] != source[alt_col] or swapped[alt_col] != source[ref_col] or swapped[label_col] != expected_label:
            mismatches += 1
    checks.append(_check("allele_swap.mirror_rows", mismatches == 0, f"mismatches={mismatches}"))
    checks.append(_level(checks))
    return checks


def allele_swap_control_passed(checks: list[AlleleSwapCheck]) -> bool:
    return any(check.name == "allele_swap.level" and check.status == "pass" for check in checks)


def write_allele_swap_validation_json(path: str | Path, checks: list[AlleleSwapCheck]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not required.issubset(columns):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        return list(reader)


def _invert_binary_label(label: str) -> str:
    if label == "0":
        return "1"
    if label == "1":
        return "0"
    raise ValueError("invert_binary label policy requires labels '0' or '1'")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _level(checks: list[AlleleSwapCheck]) -> AlleleSwapCheck:
    failed = [check.name for check in checks if check.status == "fail"]
    if failed:
        return AlleleSwapCheck("allele_swap.level", "fail", f"invalid_allele_swap_control: {failed}")
    return AlleleSwapCheck("allele_swap.level", "pass", "allele_swap_control_ready")


def _check(name: str, ok: bool, detail: str) -> AlleleSwapCheck:
    return AlleleSwapCheck(name, "pass" if ok else "fail", detail)
