from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_glm.data.manifest import write_window_manifest
from jepa_glm.data.splits import DEFAULT_CHROMOSOME_SPLIT, assert_disjoint_splits
from jepa_glm.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_jepa.yaml")
    parser.add_argument("--fasta")
    parser.add_argument("--output", default="outputs/grch38_windows.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    assert_disjoint_splits(DEFAULT_CHROMOSOME_SPLIT)
    if not args.fasta:
        print("GRCh38 split scaffold OK. Pass --fasta to create a chromosome-disjoint window manifest.")
        return
    data_cfg = config.get("data", {})
    counts = write_window_manifest(
        fasta_path=args.fasta,
        output_csv=args.output,
        window_size=int(data_cfg.get("sequence_length", 512)),
        stride=int(data_cfg.get("stride", data_cfg.get("sequence_length", 512))),
    )
    summary_path = Path(args.output).with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"manifest={args.output} summary={summary_path} counts={counts}")


if __name__ == "__main__":
    main()
