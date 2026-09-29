from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingSegmentPlan:
    step_unit: str
    gradient_accumulation_steps: int
    target_micro_steps: int
    target_optimizer_steps: int
    segment_micro_steps: int
    segment_optimizer_steps: int


def infer_completed_optimizer_steps(micro_steps: int, gradient_accumulation_steps: int) -> int:
    if micro_steps <= 0:
        return 0
    return math.ceil(micro_steps / gradient_accumulation_steps)


def plan_training_segment(
    training: dict[str, Any],
    *,
    start_micro_steps: int = 0,
    start_optimizer_steps: int = 0,
    max_new_micro_steps: int | None = None,
    max_new_optimizer_steps: int | None = None,
) -> TrainingSegmentPlan:
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    if grad_accum <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if start_micro_steps < 0 or start_optimizer_steps < 0:
        raise ValueError("completed step counters cannot be negative")
    if max_new_micro_steps is not None and max_new_optimizer_steps is not None:
        raise ValueError("Use only one of max_new_micro_steps and max_new_optimizer_steps")

    has_optimizer_target = training.get("optimizer_steps") is not None
    has_legacy_target = training.get("steps") is not None
    if has_optimizer_target and has_legacy_target:
        raise ValueError("training.optimizer_steps and training.steps are mutually exclusive")

    if has_optimizer_target:
        target_optimizer = int(training["optimizer_steps"])
        if target_optimizer <= 0:
            raise ValueError("training.optimizer_steps must be positive")
        remaining_optimizer = max(0, target_optimizer - start_optimizer_steps)
        segment_optimizer = remaining_optimizer
        if max_new_optimizer_steps is not None:
            segment_optimizer = min(segment_optimizer, max(0, int(max_new_optimizer_steps)))
        if max_new_micro_steps is not None:
            requested_micro = max(0, int(max_new_micro_steps))
            if requested_micro % grad_accum != 0:
                raise ValueError("Optimizer-step schedules require max_new_micro_steps to be divisible by gradient_accumulation_steps")
            segment_optimizer = min(segment_optimizer, requested_micro // grad_accum)
        return TrainingSegmentPlan(
            step_unit="optimizer",
            gradient_accumulation_steps=grad_accum,
            target_micro_steps=start_micro_steps + remaining_optimizer * grad_accum,
            target_optimizer_steps=target_optimizer,
            segment_micro_steps=segment_optimizer * grad_accum,
            segment_optimizer_steps=segment_optimizer,
        )

    target_micro = int(training.get("steps", 10))
    if target_micro <= 0:
        raise ValueError("training.steps must be positive")
    remaining_micro = max(0, target_micro - start_micro_steps)
    segment_micro = remaining_micro
    if max_new_micro_steps is not None:
        segment_micro = min(segment_micro, max(0, int(max_new_micro_steps)))
    if max_new_optimizer_steps is not None:
        segment_micro = min(segment_micro, max(0, int(max_new_optimizer_steps)) * grad_accum)
    return TrainingSegmentPlan(
        step_unit="microbatch_legacy",
        gradient_accumulation_steps=grad_accum,
        target_micro_steps=target_micro,
        target_optimizer_steps=start_optimizer_steps + infer_completed_optimizer_steps(remaining_micro, grad_accum),
        segment_micro_steps=segment_micro,
        segment_optimizer_steps=infer_completed_optimizer_steps(segment_micro, grad_accum),
    )
