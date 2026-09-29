from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from jepa_glm.data.fasta_dataset import ManifestWindowDataset, SyntheticDnaDataset
from jepa_glm.data.tokenizer import build_tokenizer
from jepa_glm.experiments.manifest import DatasetManifest, build_experiment_manifest, write_manifest
from jepa_glm.experiments.reports import write_step_metrics_csv, write_summary_json
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.training.checkpoint import checkpoint_state, load_checkpoint, save_checkpoint
from jepa_glm.training.trainer import train_steps
from jepa_glm.utils.config import load_config
from jepa_glm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_jepa.yaml")
    parser.add_argument("--output", default="outputs/smoke_checkpoint.pt")
    parser.add_argument("--run-dir", default="outputs/smoke")
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 7)))
    tokenizer = build_tokenizer(config)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    data_cfg = config["data"]
    if data_cfg.get("manifest_csv") and data_cfg.get("fasta_path"):
        dataset = ManifestWindowDataset(
            data_cfg["fasta_path"],
            data_cfg["manifest_csv"],
            tokenizer,
            split=str(data_cfg.get("split", "train")),
            max_length=int(data_cfg.get("sequence_length", 128)),
        )
        dataset_name = f"manifest_{data_cfg.get('split', 'train')}"
        dataset_source = str(data_cfg["manifest_csv"])
    else:
        dataset = SyntheticDnaDataset(
            int(data_cfg.get("num_sequences", 128)),
            int(data_cfg.get("sequence_length", 128)),
            tokenizer,
            seed=int(config.get("seed", 7)),
        )
        dataset_name = "synthetic_dna"
        dataset_source = "deterministic_random_acgt"
    validation_fraction = float(config["training"].get("validation_fraction", 0.0))
    validation_dataloader = None
    if 0.0 < validation_fraction < 1.0 and len(dataset) > 1:
        validation_size = max(1, int(round(len(dataset) * validation_fraction)))
        train_size = len(dataset) - validation_size
        generator = torch.Generator().manual_seed(int(config.get("seed", 7)))
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    else:
        train_dataset = dataset
        validation_dataset = None
    dataloader = DataLoader(train_dataset, batch_size=int(config["training"].get("batch_size", 4)), shuffle=True)
    if validation_dataset is not None:
        validation_dataloader = DataLoader(validation_dataset, batch_size=int(config["training"].get("batch_size", 4)), shuffle=False)
    model = JepaGlmModel(config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"].get("learning_rate", 5e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    scheduler = None
    if str(config["training"].get("scheduler", "none")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config["training"].get("steps", 1))),
            eta_min=float(config["training"].get("min_learning_rate", 0.0)),
        )
    start_step = 0
    if args.resume:
        state = load_checkpoint(args.resume)
        model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_step = int(state.get("step", 0))
    run_dir = Path(args.run_dir)
    dataset_manifest = DatasetManifest(
        name=dataset_name,
        num_sequences=len(dataset),
        sequence_length=int(config["data"].get("sequence_length", 128)),
        source=dataset_source,
        seed=int(config.get("seed", 7)),
    )
    manifest = build_experiment_manifest(config, model, dataset_manifest, device, run_id=run_dir.name)
    write_manifest(run_dir / "manifest.json", manifest)
    summary = train_steps(
        model,
        dataloader,
        config,
        tokenizer,
        device,
        optimizer=optimizer,
        scheduler=scheduler,
        start_step=start_step,
        validation_dataloader=validation_dataloader,
    )
    write_step_metrics_csv(run_dir / "metrics.csv", summary.step_metrics)
    write_summary_json(run_dir / "summary.json", asdict(summary))
    save_checkpoint(
        Path(args.output),
        checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            summary=asdict(summary),
            step=summary.completed_steps,
        ),
    )
    print(f"smoke_test_ok checkpoint={args.output} final_loss={summary.final_loss:.4f}")


if __name__ == "__main__":
    main()
