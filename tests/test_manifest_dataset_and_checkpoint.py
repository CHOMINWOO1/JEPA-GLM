from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_glm.data.fasta_dataset import ManifestWindowDataset, SyntheticDnaDataset
from jepa_glm.data.manifest import write_window_manifest
from jepa_glm.data.tokenizer import DnaTokenizer
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.training.checkpoint import capture_rng_state, checkpoint_state, load_checkpoint, restore_rng_state, save_checkpoint
from jepa_glm.training.trainer import train_steps


def test_manifest_window_dataset_reads_declared_split(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1\nACGTACGT\n>chr22\nTTTTCCCC\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    write_window_manifest(fasta, manifest, window_size=4, stride=4)
    tokenizer = DnaTokenizer()
    dataset = ManifestWindowDataset(fasta, manifest, tokenizer, split="test", max_length=4)
    assert len(dataset) == 2
    assert dataset[0].shape == (4,)


def test_checkpoint_round_trip_with_optimizer_and_resume_step(tmp_path: Path) -> None:
    tokenizer = DnaTokenizer()
    config = {
        "training": {"steps": 1, "batch_size": 2, "learning_rate": 1e-3, "ema_momentum": 0.9},
        "data": {"mask_ratio": 0.25, "mask_span_length": 4},
        "loss": {"lambda_jepa": 1.0, "lambda_mlm": 0.1, "lambda_reg": 0.01},
    }
    model = JepaGlmModel({"vocab_size": tokenizer.vocab_size, "hidden_size": 32, "max_length": 64})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = SyntheticDnaDataset(4, 32, tokenizer)
    summary = train_steps(model, DataLoader(dataset, batch_size=2), config, tokenizer, torch.device("cpu"), optimizer=optimizer)
    path = tmp_path / "checkpoint.pt"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    save_checkpoint(
        path,
        checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            summary=asdict(summary),
            step=summary.completed_steps,
            optimizer_step=summary.completed_optimizer_steps,
            scaler_state={"scale": 65536.0},
            rng_state=capture_rng_state(),
        ),
    )
    loaded = load_checkpoint(path)
    assert loaded["step"] == 1
    assert loaded["checkpoint_version"] == 4
    assert loaded["optimizer_step"] == 1
    assert loaded["scaler"] == {"scale": 65536.0}
    assert restore_rng_state(loaded["rng_state"])
    assert "optimizer" in loaded
    assert "scheduler" in loaded
