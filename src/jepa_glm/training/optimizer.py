from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


GROUP_SPECS = (
    ("backbone", "context_encoder", "backbone_learning_rate"),
    ("predictor", "predictor", "predictor_learning_rate"),
    ("mlm_head", "mlm_head", "mlm_head_learning_rate"),
)


def build_training_optimizer(model: nn.Module, training: Mapping[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(training.get("learning_rate", 5e-4))
    weight_decay = float(training.get("weight_decay", 0.01))
    discriminative = any(key in training for _, _, key in GROUP_SPECS)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not discriminative:
        return _build_optimizer(trainable, training, lr=base_lr, weight_decay=weight_decay)

    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    for name, attribute, lr_key in GROUP_SPECS:
        module = getattr(model, attribute, None)
        if not isinstance(module, nn.Module):
            continue
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in assigned
        ]
        if not parameters:
            continue
        assigned.update(id(parameter) for parameter in parameters)
        groups.append(
            {
                "name": name,
                "params": parameters,
                "lr": float(training.get(lr_key, base_lr)),
            }
        )
    remaining = [parameter for parameter in trainable if id(parameter) not in assigned]
    if remaining:
        groups.append({"name": "other", "params": remaining, "lr": base_lr})
    if not groups:
        raise ValueError("No trainable parameters are available for the optimizer")
    return _build_optimizer(groups, training, lr=base_lr, weight_decay=weight_decay)


def _build_optimizer(parameters, training: Mapping[str, Any], *, lr: float, weight_decay: float):
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(
                float(training.get("adam_beta1", 0.9)),
                float(training.get("adam_beta2", 0.999)),
            ),
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(training.get("sgd_momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError("training.optimizer must be 'adamw' or 'sgd'")


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    training: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(training.get("scheduler", "")).lower()
    if scheduler_name == "warmup_cosine":
        return _build_warmup_cosine_scheduler(optimizer, training)
    if scheduler_name != "cosine":
        return None
    total_steps = max(1, int(training.get("optimizer_steps", training.get("steps", 1))))
    base_lr = float(training.get("learning_rate", optimizer.param_groups[0]["lr"]))
    min_lr = float(training.get("min_learning_rate", 0.0))
    min_ratio = min(1.0, max(0.0, min_lr / base_lr)) if base_lr > 0.0 else 0.0
    has_named_groups = any("name" in group for group in optimizer.param_groups)
    if not has_named_groups:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )

    def cosine_multiplier(step: int) -> float:
        progress = min(max(step, 0), total_steps) / total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[cosine_multiplier for _ in optimizer.param_groups],
    )


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    training: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(training.get("optimizer_steps", training.get("steps", 1))))
    predictor_warmup = max(0, int(training.get("predictor_warmup_steps", 0)))
    joint_warmup = max(0, int(training.get("joint_warmup_steps", 0)))
    if predictor_warmup + joint_warmup >= total_steps:
        raise ValueError("Warmup steps must leave at least one joint decay step")
    predictor_warmup_lr = float(training.get("predictor_warmup_learning_rate", 1e-5))
    warmup_start_lr = float(training.get("warmup_start_learning_rate", 3e-6))
    peak_lr = float(training.get("learning_rate", 5e-6))
    final_lr = float(training.get("min_learning_rate", 1e-6))

    def multiplier(name: str, initial_lr: float):
        def schedule(step: int) -> float:
            if step < predictor_warmup:
                absolute_lr = predictor_warmup_lr if name == "predictor" else 0.0
            else:
                joint_step = step - predictor_warmup
                if joint_step < joint_warmup:
                    progress = joint_step / max(1, joint_warmup - 1)
                    absolute_lr = warmup_start_lr + (peak_lr - warmup_start_lr) * progress
                else:
                    decay_steps = max(1, total_steps - predictor_warmup - joint_warmup - 1)
                    progress = min(max(joint_step - joint_warmup, 0), decay_steps) / decay_steps
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    absolute_lr = final_lr + (peak_lr - final_lr) * cosine
            return absolute_lr / initial_lr

        return schedule

    lambdas = []
    for index, group in enumerate(optimizer.param_groups):
        initial_lr = float(group.get("initial_lr", group["lr"]))
        if initial_lr <= 0.0:
            raise ValueError("Warmup-cosine optimizer group initial learning rates must be positive")
        name = str(group.get("name", "all" if len(optimizer.param_groups) == 1 else f"group_{index}"))
        lambdas.append(multiplier(name, initial_lr))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def optimizer_group_manifest(optimizer: torch.optim.Optimizer) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        rows.append(
            {
                "name": str(group.get("name", "all" if len(optimizer.param_groups) == 1 else f"group_{index}")),
                "initial_learning_rate": float(group.get("initial_lr", group["lr"])),
                "final_learning_rate": float(group["lr"]),
                "parameter_tensors": len(parameters),
                "parameters": sum(parameter.numel() for parameter in parameters),
            }
        )
    return rows


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", "all" if len(optimizer.param_groups) == 1 else f"group_{index}"))
        rates[name] = float(group["lr"])
    return rates
