from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.evaluation.embedding import extract_sequence_embeddings
from jepa_glm.evaluation.zero_shot_variant import (
    evaluate_variant_scores,
    evaluate_variant_scores_with_ci,
    read_variant_sequence_csv,
    variant_scores,
    write_variant_scores_csv,
)
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--scores-csv", default="outputs/variant_scores.csv")
    parser.add_argument("--metrics-json", default="outputs/variant_metrics.json")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--local-model-dir")
    parser.add_argument("--backbone-name")
    parser.add_argument("--freeze-policy", default="all")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tokenizer_name = args.tokenizer_name or args.local_model_dir
    tokenizer_config = {"tokenizer": {"type": "hf", "name": tokenizer_name}} if tokenizer_name else {"tokenizer": {"type": "dna"}}
    tokenizer = build_tokenizer(tokenizer_config)
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": args.hidden_size,
        "max_length": args.max_length,
    }
    if args.local_model_dir or args.backbone_name:
        model_config.update(
            {
                "backbone_name": args.backbone_name or args.local_model_dir,
                "local_model_dir": args.local_model_dir,
                "freeze_policy": args.freeze_policy,
                "trust_remote_code": args.trust_remote_code,
                "local_files_only": args.local_files_only or bool(args.local_model_dir),
            }
        )
    model = JepaGlmModel(model_config)
    if args.checkpoint and Path(args.checkpoint).exists():
        state = load_checkpoint(args.checkpoint)
        model.load_state_dict(state["model"], strict=False)
    table = read_variant_sequence_csv(args.input_csv)
    if args.max_variants is not None and args.max_variants > 0:
        table = type(table)(
            table.reference_sequences[: args.max_variants],
            table.alternate_sequences[: args.max_variants],
            table.labels[: args.max_variants],
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_embeddings = extract_sequence_embeddings(
        model, table.reference_sequences, tokenizer, args.max_length, args.batch_size, device
    )
    alternate_embeddings = extract_sequence_embeddings(
        model, table.alternate_sequences, tokenizer, args.max_length, args.batch_size, device
    )
    scores = variant_scores(reference_embeddings, alternate_embeddings)
    write_variant_scores_csv(args.scores_csv, table, scores)
    if args.bootstrap_samples > 0:
        metrics = evaluate_variant_scores_with_ci(table.labels, scores, n_bootstrap=args.bootstrap_samples, seed=args.seed)
    else:
        metrics = evaluate_variant_scores(table.labels, scores)
    for metric in metrics.values():
        metric["n_variants"] = len(table.labels)
        metric["embedding_dim"] = int(reference_embeddings.shape[1]) if reference_embeddings.ndim == 2 else 0
        metric["model_source"] = str(args.local_model_dir or args.backbone_name or args.checkpoint or "tiny")
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"variant_scores={args.scores_csv} metrics={metrics_path}")


if __name__ == "__main__":
    main()
