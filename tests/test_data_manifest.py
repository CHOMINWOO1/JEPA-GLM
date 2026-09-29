from __future__ import annotations

from pathlib import Path

from jepa_glm.data.manifest import split_for_chromosome, write_window_manifest


def test_split_for_chromosome_uses_disjoint_default_split() -> None:
    assert split_for_chromosome("chr1") == "train"
    assert split_for_chromosome("20") == "validation"
    assert split_for_chromosome("chr22") == "test"
    assert split_for_chromosome("chrX") is None


def test_write_window_manifest(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1\nACGTACGTACGT\n>chr22\nACGTNNNNACGT\n", encoding="utf-8")
    output = tmp_path / "manifest.csv"
    counts = write_window_manifest(fasta, output, window_size=4, stride=4)
    text = output.read_text(encoding="utf-8")
    assert counts["train"] == 3
    assert counts["test"] == 2
    assert counts["skipped"] == 1
    assert "split,chromosome,start,end,length,non_acgt_fraction" in text
