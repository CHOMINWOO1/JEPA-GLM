from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = 4


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("checkpoint_version", CHECKPOINT_VERSION)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_cpu_state = state["torch_cpu"]
    if torch.is_tensor(torch_cpu_state):
        torch_cpu_state = torch_cpu_state.detach().cpu()
    torch.set_rng_state(torch_cpu_state)
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        normalized_cuda_states = [item.detach().cpu() if torch.is_tensor(item) else item for item in cuda_states]
        torch.cuda.set_rng_state_all(normalized_cuda_states)
    return True


def checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: dict[str, Any],
    summary: dict[str, Any],
    step: int,
    optimizer_step: int | None = None,
    scaler_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "config": config,
        "summary": summary,
        "step": step,
    }
    if optimizer_step is not None:
        state["optimizer_step"] = int(optimizer_step)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler_state is not None:
        state["scaler"] = scaler_state
    if rng_state is not None:
        state["rng_state"] = rng_state
    return state
