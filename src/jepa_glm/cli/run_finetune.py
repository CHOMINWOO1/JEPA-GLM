from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.evaluation.embedding import SequenceDataset, load_context_encoder_module_state
from jepa_glm.evaluation.finetune import (
    SequenceClassifier,
    sequence_classifier_scores,
    train_sequence_classifier,
    write_sequence_classifier_scores,
)
from jepa_glm.evaluation.linear_probe import read_labeled_sequence_csv, sample_labeled_sequence_table
from jepa_glm.evaluation.metrics import binary_classification_metrics_with_ci
from jepa_glm.models.backbone import build_backbone
from jepa_glm.models.context_encoder import ContextEncoder
from jepa_glm.training.checkpoint import load_checkpoint
from jepa_glm.utils.config import load_config
from jepa_glm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scores-csv")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-examples-per-split", type=int)
    parser.add_argument("--evaluation-split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    tokenizer = build_tokenizer(config)
    loaded_splits = {"train", args.evaluation_split}
    table = read_labeled_sequence_csv(args.input_csv, include_splits=loaded_splits)
    table = sample_labeled_sequence_table(
        table,
        max_examples_per_split=args.max_examples_per_split,
        seed=seed,
        balanced=True,
    )
    if table.splits is None:
        raise ValueError("Fine-tuning requires an explicit split column")
    train_indices = [index for index, split in enumerate(table.splits) if split == "train"]
    evaluation_indices = [index for index, split in enumerate(table.splits) if split == args.evaluation_split]
    if not train_indices or not evaluation_indices:
        raise ValueError("Fine-tuning requires nonempty train and evaluation splits")

    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("vocab_size", tokenizer.vocab_size)
    model_cfg.setdefault("hidden_size", 64)
    model_cfg.setdefault("max_length", 256)
    backbone = build_backbone(model_cfg)
    hidden_size = int(getattr(backbone, "hidden_size"))
    context_encoder = ContextEncoder(backbone)
    checkpoint = args.checkpoint or model_cfg.get("checkpoint")
    loaded_context_tensors = 0
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
        state = load_checkpoint(checkpoint_path)
        loaded_context_tensors = load_context_encoder_module_state(context_encoder, state["model"])

    finetune_cfg = dict(config.get("fine_tuning", {}))
    max_length = int(model_cfg.get("max_length", 256))
    train_dataset = SequenceDataset(
        [table.sequences[index] for index in train_indices],
        [table.labels[index] for index in train_indices],
        tokenizer,
        max_length,
    )
    evaluation_sequences = [table.sequences[index] for index in evaluation_indices]
    evaluation_dataset = SequenceDataset(
        evaluation_sequences,
        [table.labels[index] for index in evaluation_indices],
        tokenizer,
        max_length,
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(finetune_cfg.get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False)
    model = SequenceClassifier(
        context_encoder,
        hidden_size,
        pooling=str(model_cfg.get("pooling", "cls")),
        dropout=float(finetune_cfg.get("dropout", 0.1)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = train_sequence_classifier(
        model,
        train_loader,
        device=device,
        epochs=int(finetune_cfg.get("epochs", 3)),
        encoder_learning_rate=float(finetune_cfg.get("encoder_learning_rate", 2e-5)),
        classifier_learning_rate=float(finetune_cfg.get("classifier_learning_rate", 1e-3)),
        weight_decay=float(finetune_cfg.get("weight_decay", 0.01)),
        gradient_accumulation_steps=int(finetune_cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(finetune_cfg.get("max_grad_norm", 1.0)),
        mixed_precision=bool(finetune_cfg.get("mixed_precision", True)),
    )
    labels, scores = sequence_classifier_scores(model, evaluation_loader, device=device)
    output = Path(args.output_json)
    scores_path = Path(args.scores_csv) if args.scores_csv else output.with_name(f"{output.stem}.scores.csv")
    scores_sha256 = write_sequence_classifier_scores(scores_path, evaluation_sequences, labels, scores)
    metrics = binary_classification_metrics_with_ci(
        labels,
        scores,
        n_bootstrap=args.bootstrap_samples,
        seed=seed,
    )
    metrics.update(
        {
            "task": str(config.get("task", "unknown")),
            "seed": seed,
            "model_source": str(model_cfg.get("local_model_dir") or model_cfg.get("backbone_name") or "tiny"),
            "pooling": model.pooling,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_load_scope": "context_encoder" if checkpoint else None,
            "loaded_context_tensors": loaded_context_tensors,
            "freeze_policy": str(model_cfg.get("freeze_policy", "none")),
            "trainable_last_n_layers": int(model_cfg.get("trainable_last_n_layers", 0)),
            "fit_splits": ["train"],
            "evaluation_split": args.evaluation_split,
            "loaded_splits": sorted(loaded_splits),
            "split_filter_applied_before_tokenization": True,
            "validation_used_for_model_selection": False,
            "n_train": len(train_dataset),
            "n_evaluation": len(evaluation_dataset),
            "scores_csv": str(scores_path),
            "scores_csv_sha256": scores_sha256,
            "scores_rows": int(labels.shape[0]),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "fine_tuning": asdict(summary),
        }
    )
    if args.max_examples_per_split:
        metrics["max_examples_per_split"] = int(args.max_examples_per_split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"finetune_metrics={output} auroc={metrics['auroc']:.6f}")


if __name__ == "__main__":
    main()
