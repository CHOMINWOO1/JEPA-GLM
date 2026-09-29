from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.evaluation.embedding import extract_sequence_embeddings, load_context_encoder_state
from jepa_glm.evaluation.linear_probe import (
    evaluate_linear_probe,
    evaluate_linear_probe_with_ci,
    read_labeled_sequence_csv,
    sample_labeled_sequence_table,
)
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.training.checkpoint import load_checkpoint
from jepa_glm.utils.config import load_config
from jepa_glm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream_promoter.yaml")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", default="outputs/linear_probe_metrics.json")
    parser.add_argument("--checkpoint", help="Override model.checkpoint from the config.")
    parser.add_argument("--seed", type=int, help="Override the config seed for repeated downstream probes.")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--max-examples-per-split", type=int)
    parser.add_argument("--unbalanced-sample", action="store_true")
    parser.add_argument("--evaluation-split", default="test", choices=["validation", "test"])
    parser.add_argument(
        "--fit-split",
        action="append",
        dest="fit_splits",
        choices=["train", "validation"],
        help="Split used to fit the probe. Repeat to combine train and validation; default is train.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.checkpoint:
        config.setdefault("model", {})["checkpoint"] = args.checkpoint
    set_seed(int(config.get("seed", 7)))
    tokenizer = build_tokenizer(config)
    fit_splits = tuple(args.fit_splits or ["train"])
    loaded_splits = set(fit_splits) | {args.evaluation_split}
    table = read_labeled_sequence_csv(args.input_csv, include_splits=loaded_splits)
    table = sample_labeled_sequence_table(
        table,
        max_examples_per_split=args.max_examples_per_split,
        seed=int(config.get("seed", 7)),
        balanced=not args.unbalanced_sample,
    )
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("vocab_size", tokenizer.vocab_size)
    model_cfg.setdefault("hidden_size", 64)
    model_cfg.setdefault("max_length", 256)
    model = JepaGlmModel(model_cfg)
    checkpoint = model_cfg.get("checkpoint")
    loaded_context_tensors = 0
    if checkpoint and Path(checkpoint).exists():
        state = load_checkpoint(checkpoint)
        loaded_context_tensors = load_context_encoder_state(model, state["model"])
    max_length = int(model_cfg.get("max_length", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = extract_sequence_embeddings(
        model,
        table.sequences,
        tokenizer,
        max_length=max_length,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 16)),
        device=device,
    )
    metric_kwargs = {
        "seed": int(config.get("seed", 7)),
        "test_size": 1.0 - float(config.get("evaluation", {}).get("train_fraction", 0.8)),
        "splits": table.splits,
        "fit_splits": fit_splits,
        "evaluation_split": args.evaluation_split,
    }
    if args.bootstrap_samples > 0:
        metrics = evaluate_linear_probe_with_ci(embeddings.numpy(), table.labels, n_bootstrap=args.bootstrap_samples, **metric_kwargs)
    else:
        metrics = evaluate_linear_probe(embeddings.numpy(), table.labels, **metric_kwargs)
    metrics["n_examples"] = len(table.labels)
    metrics["embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    metrics["model_source"] = str(model_cfg.get("local_model_dir") or model_cfg.get("backbone_name") or "tiny")
    metrics["pooling"] = str(model.pooling)
    metrics["task"] = str(config.get("task", "unknown"))
    metrics["seed"] = int(config.get("seed", 7))
    metrics["fit_splits"] = list(fit_splits)
    metrics["evaluation_split"] = args.evaluation_split
    metrics["loaded_splits"] = sorted(loaded_splits)
    metrics["split_filter_applied_before_embedding"] = True
    if checkpoint:
        metrics["checkpoint"] = str(checkpoint)
        metrics["checkpoint_load_scope"] = "context_encoder"
        metrics["loaded_context_tensors"] = int(loaded_context_tensors)
    if args.max_examples_per_split:
        metrics["max_examples_per_split"] = int(args.max_examples_per_split)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"linear_probe_metrics={output}")


if __name__ == "__main__":
    main()
