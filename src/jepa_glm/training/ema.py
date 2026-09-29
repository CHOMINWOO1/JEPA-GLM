from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, momentum: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        target_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)


def max_parameter_delta(left: nn.Module, right: nn.Module) -> float:
    deltas = [
        (left_param.detach() - right_param.detach()).abs().max().item()
        for left_param, right_param in zip(left.parameters(), right.parameters(), strict=True)
    ]
    return max(deltas) if deltas else 0.0
