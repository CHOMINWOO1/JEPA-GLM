from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from jepa_glm.data.reference import fetch_interval, load_fasta_dict

DNA_ALPHABET = {"A", "C", "G", "T"}


@dataclass(frozen=True)
class ConversionSummary:
    converter: str
    source: str
    output: str
    n_rows: int
    notes: dict[str, str | int | float]


def bed_to_labeled_sequences(
    bed_path: str | Path,
    fasta_path: str | Path,
    output_csv: str | Path,
    label: int,
    window_size: int | None = None,
    summary_json: str | Path | None = None,
) -> ConversionSummary:
    records = load_fasta_dict(fasta_path)
    rows: list[dict[str, str | int]] = []
    with Path(bed_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError("BED rows must contain at least chrom, start, end")
            chrom, start_text, end_text = parts[:3]
            start = int(start_text)
            end = int(end_text)
            if window_size is not None:
                center = (start + end) // 2
                start = center - window_size // 2
                end = start + window_size
            sequence = fetch_interval(records, chrom, start, end)
            rows.append({"sequence": sequence, "label": int(label)})
    _write_csv(output_csv, _fieldnames_for_rows(["sequence", "label"], rows), rows)
    summary = ConversionSummary(
        "bed_to_labeled_sequences",
        str(bed_path),
        str(output_csv),
        len(rows),
        {"label": int(label), "window_size": int(window_size or 0)},
    )
    _write_summary(summary_json, summary)
    return summary


def bed_pair_to_labeled_sequences(
    positive_bed: str | Path,
    negative_bed: str | Path,
    fasta_path: str | Path,
    output_csv: str | Path,
    window_size: int | None = None,
    summary_json: str | Path | None = None,
    allow_overlap: bool = False,
) -> ConversionSummary:
    records = load_fasta_dict(fasta_path)
    positive_rows, positive_intervals = _bed_rows_to_sequences(positive_bed, records, label=1, window_size=window_size)
    negative_rows, negative_intervals = _bed_rows_to_sequences(negative_bed, records, label=0, window_size=window_size)
    overlap_count = _count_cross_label_overlaps(positive_intervals, negative_intervals)
    if overlap_count and not allow_overlap:
        raise ValueError(f"Positive and negative BED intervals overlap: n={overlap_count}")
    rows = [*positive_rows, *negative_rows]
    duplicate_count = len([row["sequence"] for row in rows]) - len({row["sequence"] for row in rows})
    _write_csv(output_csv, _fieldnames_for_rows(["sequence", "label", "chrom", "start", "end"], rows), rows)
    summary = ConversionSummary(
        "bed_pair_to_labeled_sequences",
        f"{positive_bed};{negative_bed}",
        str(output_csv),
        len(rows),
        {
            "positive_rows": sum(1 for row in rows if int(row["label"]) == 1),
            "negative_rows": sum(1 for row in rows if int(row["label"]) == 0),
            "cross_label_interval_overlaps": overlap_count,
            "duplicate_sequences": duplicate_count,
            "window_size": int(window_size or 0),
        },
    )
    _write_summary(summary_json, summary)
    return summary


def variant_table_to_sequences(
    input_csv: str | Path,
    fasta_path: str | Path,
    output_csv: str | Path,
    flank: int,
    chrom_column: str = "chrom",
    pos_column: str = "pos",
    ref_column: str = "ref",
    alt_column: str = "alt",
    label_column: str = "label",
    summary_json: str | Path | None = None,
    max_rows: int | None = None,
    balanced_labels: bool = False,
) -> ConversionSummary:
    records = load_fasta_dict(fasta_path)
    rows: list[dict[str, str | int]] = []
    label_counts: dict[int, int] = {}
    per_label_target = max(1, max_rows // 2) if balanced_labels and max_rows is not None else None
    with Path(input_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {chrom_column, pos_column, ref_column, alt_column, label_column}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Variant CSV must contain columns: {sorted(required)}")
        for row in reader:
            chrom = row[chrom_column]
            pos0 = int(row[pos_column]) - 1
            ref = row[ref_column].upper()
            alt = row[alt_column].upper()
            label = int(row[label_column])
            if per_label_target is not None and label_counts.get(label, 0) >= per_label_target:
                continue
            left = fetch_interval(records, chrom, pos0 - flank, pos0)
            right = fetch_interval(records, chrom, pos0 + len(ref), pos0 + len(ref) + flank)
            observed_ref = fetch_interval(records, chrom, pos0, pos0 + len(ref))
            if observed_ref != ref:
                raise ValueError(f"Reference allele mismatch at {chrom}:{pos0 + 1}: expected {ref}, observed {observed_ref}")
            rows.append(
                {
                    "ref_sequence": left + ref + right,
                    "alt_sequence": left + alt + right,
                    "label": label,
                }
            )
            label_counts[label] = label_counts.get(label, 0) + 1
            if max_rows is not None and len(rows) >= max_rows:
                break
            if per_label_target is not None and label_counts.get(0, 0) >= per_label_target and label_counts.get(1, 0) >= per_label_target:
                break
    _write_csv(output_csv, ["ref_sequence", "alt_sequence", "label"], rows)
    summary = ConversionSummary(
        "variant_table_to_sequences",
        str(input_csv),
        str(output_csv),
        len(rows),
        {"flank": int(flank), "max_rows": int(max_rows or 0), "balanced_labels": int(balanced_labels)},
    )
    _write_summary(summary_json, summary)
    return summary


def gencode_gtf_to_splice_sequences(
    input_gtf: str | Path,
    fasta_path: str | Path,
    output_csv: str | Path,
    window_size: int = 512,
    summary_json: str | Path | None = None,
    max_positive_sites: int | None = None,
    negative_offset: int | None = None,
    chrom_prefix: str = "chr",
) -> ConversionSummary:
    records = load_fasta_dict(fasta_path)
    negative_offset = int(negative_offset or window_size * 10)
    rows: list[dict[str, str | int]] = []
    seen_positive: set[tuple[str, int, str]] = set()
    seen_sequences: set[str] = set()
    skipped = {"missing_chrom": 0, "near_edge": 0, "duplicates": 0, "duplicate_sequences": 0}
    with _open_text(input_gtf) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom = _normalize_chrom(parts[0], chrom_prefix)
            if chrom not in records:
                skipped["missing_chrom"] += 1
                continue
            start0 = int(parts[3]) - 1
            end0 = int(parts[4])
            strand = parts[6]
            if strand == "+":
                sites = [("acceptor", start0), ("donor", end0)]
            elif strand == "-":
                sites = [("donor", start0), ("acceptor", end0)]
            else:
                continue
            for site_type, center in sites:
                key = (chrom, center, site_type)
                if key in seen_positive:
                    skipped["duplicates"] += 1
                    continue
                seen_positive.add(key)
                positive = _site_window_row(records, chrom, center, window_size, 1, site_type, strand)
                if positive is None:
                    skipped["near_edge"] += 1
                    continue
                negative_center = _negative_center(center, len(records[chrom]), window_size, negative_offset)
                negative = _site_window_row(records, chrom, negative_center, window_size, 0, f"matched_{site_type}", strand)
                if negative is None:
                    skipped["near_edge"] += 1
                    continue
                if str(positive["sequence"]) in seen_sequences or str(negative["sequence"]) in seen_sequences or positive["sequence"] == negative["sequence"]:
                    skipped["duplicate_sequences"] += 1
                    continue
                seen_sequences.add(str(positive["sequence"]))
                seen_sequences.add(str(negative["sequence"]))
                rows.extend([positive, negative])
                if max_positive_sites is not None and len(rows) // 2 >= max_positive_sites:
                    _write_csv(output_csv, _fieldnames_for_rows(["sequence", "label", "chrom", "start", "end", "site_type", "strand"], rows), rows)
                    summary = _splice_summary(input_gtf, output_csv, len(rows), window_size, skipped, max_positive_sites)
                    _write_summary(summary_json, summary)
                    return summary
    _write_csv(output_csv, _fieldnames_for_rows(["sequence", "label", "chrom", "start", "end", "site_type", "strand"], rows), rows)
    summary = _splice_summary(input_gtf, output_csv, len(rows), window_size, skipped, max_positive_sites)
    _write_summary(summary_json, summary)
    return summary


def gencode_gtf_to_promoter_sequences(
    input_gtf: str | Path,
    fasta_path: str | Path,
    output_csv: str | Path,
    window_size: int = 512,
    summary_json: str | Path | None = None,
    max_positive_sites: int | None = None,
    negative_offset: int | None = None,
    chrom_prefix: str = "chr",
) -> ConversionSummary:
    records = load_fasta_dict(fasta_path)
    negative_offset = int(negative_offset or window_size * 20)
    rows: list[dict[str, str | int]] = []
    seen_tss: set[tuple[str, int, str]] = set()
    seen_sequences: set[str] = set()
    skipped = {"missing_chrom": 0, "near_edge": 0, "duplicates": 0, "duplicate_sequences": 0}
    with _open_text(input_gtf) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "transcript":
                continue
            chrom = _normalize_chrom(parts[0], chrom_prefix)
            if chrom not in records:
                skipped["missing_chrom"] += 1
                continue
            start0 = int(parts[3]) - 1
            end0 = int(parts[4])
            strand = parts[6]
            if strand == "+":
                tss = start0
            elif strand == "-":
                tss = end0
            else:
                continue
            key = (chrom, tss, strand)
            if key in seen_tss:
                skipped["duplicates"] += 1
                continue
            seen_tss.add(key)
            positive = _site_window_row(records, chrom, tss, window_size, 1, "tss_promoter", strand)
            if positive is None:
                skipped["near_edge"] += 1
                continue
            negative_center = _negative_center(tss, len(records[chrom]), window_size, negative_offset)
            negative = _site_window_row(records, chrom, negative_center, window_size, 0, "matched_non_tss", strand)
            if negative is None:
                skipped["near_edge"] += 1
                continue
            if str(positive["sequence"]) in seen_sequences or str(negative["sequence"]) in seen_sequences or positive["sequence"] == negative["sequence"]:
                skipped["duplicate_sequences"] += 1
                continue
            seen_sequences.add(str(positive["sequence"]))
            seen_sequences.add(str(negative["sequence"]))
            rows.extend([positive, negative])
            if max_positive_sites is not None and len(rows) // 2 >= max_positive_sites:
                _write_csv(output_csv, _fieldnames_for_rows(["sequence", "label", "chrom", "start", "end", "site_type", "strand"], rows), rows)
                summary = _promoter_summary(input_gtf, output_csv, len(rows), window_size, skipped, max_positive_sites)
                _write_summary(summary_json, summary)
                return summary
    _write_csv(output_csv, _fieldnames_for_rows(["sequence", "label", "chrom", "start", "end", "site_type", "strand"], rows), rows)
    summary = _promoter_summary(input_gtf, output_csv, len(rows), window_size, skipped, max_positive_sites)
    _write_summary(summary_json, summary)
    return summary


def _bed_rows_to_sequences(
    bed_path: str | Path,
    records: dict[str, str],
    label: int,
    window_size: int | None = None,
) -> tuple[list[dict[str, str | int]], list[tuple[str, int, int]]]:
    rows: list[dict[str, str | int]] = []
    intervals: list[tuple[str, int, int]] = []
    with Path(bed_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError("BED rows must contain at least chrom, start, end")
            chrom, start_text, end_text = parts[:3]
            start = int(start_text)
            end = int(end_text)
            if window_size is not None:
                center = (start + end) // 2
                start = center - window_size // 2
                end = start + window_size
            sequence = fetch_interval(records, chrom, start, end)
            rows.append({"sequence": sequence, "label": int(label), "chrom": chrom, "start": start, "end": end})
            intervals.append((chrom, start, end))
    return rows, intervals


def _site_window_row(
    records: dict[str, str],
    chrom: str,
    center: int,
    window_size: int,
    label: int,
    site_type: str,
    strand: str,
) -> dict[str, str | int] | None:
    start = center - window_size // 2
    end = start + window_size
    if start < 0 or end > len(records[chrom]):
        return None
    sequence = fetch_interval(records, chrom, start, end)
    if any(base not in DNA_ALPHABET for base in sequence):
        return None
    return {"sequence": sequence, "label": int(label), "chrom": chrom, "start": start, "end": end, "site_type": site_type, "strand": strand}


def _negative_center(center: int, chrom_length: int, window_size: int, offset: int) -> int:
    lower = window_size // 2
    upper = chrom_length - (window_size - window_size // 2)
    if upper <= lower:
        return center
    shifted = center + offset
    return lower + ((shifted - lower) % (upper - lower))


def _splice_summary(
    input_gtf: str | Path,
    output_csv: str | Path,
    n_rows: int,
    window_size: int,
    skipped: dict[str, int],
    max_positive_sites: int | None,
) -> ConversionSummary:
    return ConversionSummary(
        "gencode_gtf_to_splice_sequences",
        str(input_gtf),
        str(output_csv),
        n_rows,
        {
            "positive_rows": n_rows // 2,
            "negative_rows": n_rows // 2,
            "window_size": int(window_size),
            "max_positive_sites": int(max_positive_sites or 0),
            "skipped_missing_chrom": skipped["missing_chrom"],
            "skipped_near_edge_or_non_acgt": skipped["near_edge"],
            "skipped_duplicate_sites": skipped["duplicates"],
            "skipped_duplicate_sequences": skipped["duplicate_sequences"],
        },
    )


def _promoter_summary(
    input_gtf: str | Path,
    output_csv: str | Path,
    n_rows: int,
    window_size: int,
    skipped: dict[str, int],
    max_positive_sites: int | None,
) -> ConversionSummary:
    return ConversionSummary(
        "gencode_gtf_to_promoter_sequences",
        str(input_gtf),
        str(output_csv),
        n_rows,
        {
            "positive_rows": n_rows // 2,
            "negative_rows": n_rows // 2,
            "window_size": int(window_size),
            "max_positive_sites": int(max_positive_sites or 0),
            "skipped_missing_chrom": skipped["missing_chrom"],
            "skipped_near_edge_or_non_acgt": skipped["near_edge"],
            "skipped_duplicate_tss": skipped["duplicates"],
            "skipped_duplicate_sequences": skipped["duplicate_sequences"],
        },
    )


def _count_cross_label_overlaps(
    positive_intervals: list[tuple[str, int, int]],
    negative_intervals: list[tuple[str, int, int]],
) -> int:
    count = 0
    negatives_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in negative_intervals:
        negatives_by_chrom.setdefault(chrom, []).append((start, end))
    for intervals in negatives_by_chrom.values():
        intervals.sort()
    for chrom, pos_start, pos_end in positive_intervals:
        for neg_start, neg_end in negatives_by_chrom.get(chrom, []):
            if neg_start >= pos_end:
                break
            if pos_start < neg_end and neg_start < pos_end:
                count += 1
    return count


def clinvar_vcf_to_variant_table(
    input_vcf: str | Path,
    output_csv: str | Path,
    summary_json: str | Path | None = None,
    max_rows: int | None = None,
    chrom_prefix: str = "chr",
) -> ConversionSummary:
    rows: list[dict[str, str | int]] = []
    skipped = {"non_acgt": 0, "unlabeled": 0, "multi_alt": 0}
    with _open_text(input_vcf) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _variant_id, ref, alt_text, _qual, _filter, info_text = parts[:8]
            alts = alt_text.split(",")
            if len(alts) != 1:
                skipped["multi_alt"] += 1
                continue
            alt = alts[0]
            if not _is_acgt(ref) or not _is_acgt(alt):
                skipped["non_acgt"] += 1
                continue
            label = _clinvar_label(info_text)
            if label is None:
                skipped["unlabeled"] += 1
                continue
            rows.append({"chrom": _normalize_chrom(chrom, chrom_prefix), "pos": pos, "ref": ref.upper(), "alt": alt.upper(), "label": label})
            if max_rows is not None and len(rows) >= max_rows:
                break
    _write_csv(output_csv, ["chrom", "pos", "ref", "alt", "label"], rows)
    summary = ConversionSummary(
        "clinvar_vcf_to_variant_table",
        str(input_vcf),
        str(output_csv),
        len(rows),
        {
            "max_rows": int(max_rows or 0),
            "skipped_multi_alt": skipped["multi_alt"],
            "skipped_non_acgt": skipped["non_acgt"],
            "skipped_unlabeled": skipped["unlabeled"],
        },
    )
    _write_summary(summary_json, summary)
    return summary


def _write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, str | int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames_for_rows(preferred: list[str], rows: list[dict[str, str | int]]) -> list[str]:
    seen = set()
    fieldnames = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    return fieldnames or preferred


def _write_summary(path: str | Path | None, summary: ConversionSummary) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _clinvar_label(info_text: str) -> int | None:
    info = _parse_vcf_info(info_text)
    clnsig = info.get("CLNSIG", "")
    normalized = {value.strip().lower() for value in clnsig.replace("|", ",").split(",") if value.strip()}
    pathogenic = {"pathogenic", "likely_pathogenic"}
    benign = {"benign", "likely_benign"}
    if normalized & pathogenic and not normalized & benign:
        return 1
    if normalized & benign and not normalized & pathogenic:
        return 0
    return None


def _parse_vcf_info(info_text: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for item in info_text.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        info[key] = value
    return info


def _is_acgt(allele: str) -> bool:
    return bool(allele) and all(base in {"A", "C", "G", "T"} for base in allele.upper())


def _normalize_chrom(chrom: str, prefix: str) -> str:
    if not prefix or chrom.startswith(prefix):
        return chrom
    return f"{prefix}{chrom}"
