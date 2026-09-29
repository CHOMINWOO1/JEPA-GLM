from __future__ import annotations

from collections.abc import Collection

import torch


def contiguous_span_mask(
    input_ids: torch.Tensor,
    mask_ratio: float,
    span_length: int,
    mask_token_id: int,
    pad_token_id: int = 0,
    excluded_token_ids: Collection[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    if span_length < 1:
        raise ValueError("span_length must be >= 1")
    batch, length = input_ids.shape
    context = input_ids.clone()
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    eligible = input_ids.ne(pad_token_id)
    for token_id in excluded_token_ids or ():
        eligible &= input_ids.ne(int(token_id))
    for row in range(batch):
        eligible_count = int(eligible[row].sum().item())
        if eligible_count == 0:
            continue
        target_tokens = min(eligible_count, max(1, int(round(eligible_count * mask_ratio))))
        while int(target_mask[row].sum().item()) < target_tokens:
            remaining = target_tokens - int(target_mask[row].sum().item())
            this_span = min(span_length, remaining)
            available = eligible[row] & ~target_mask[row]
            candidates = [
                start
                for start in range(0, length - this_span + 1)
                if bool(available[start : start + this_span].all().item())
            ]
            if not candidates:
                available_indices = available.nonzero(as_tuple=False).flatten()
                if available_indices.numel() == 0:
                    break
                order = torch.randperm(available_indices.numel(), device=input_ids.device)
                chosen = available_indices[order[:remaining]]
                target_mask[row, chosen] = True
                break
            choice = int(torch.randint(0, len(candidates), (1,), device=input_ids.device).item())
            start = candidates[choice]
            target_mask[row, start : start + this_span] = True
    context[target_mask] = mask_token_id
    labels = input_ids.masked_fill(~target_mask, -100)
    return context, target_mask, labels


def multi_region_span_mask(
    input_ids: torch.Tensor,
    *,
    min_mask_ratio: float,
    max_mask_ratio: float,
    min_regions: int,
    max_regions: int,
    mask_token_id: int,
    pad_token_id: int = 0,
    excluded_token_ids: Collection[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask a sampled fraction of eligible tokens using a small number of spans."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    if not 0.0 < min_mask_ratio <= max_mask_ratio <= 1.0:
        raise ValueError("mask ratio bounds must satisfy 0 < min <= max <= 1")
    if min_regions < 1 or max_regions < min_regions:
        raise ValueError("region bounds must satisfy 1 <= min_regions <= max_regions")
    context = input_ids.clone()
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    eligible = input_ids.ne(pad_token_id)
    for token_id in excluded_token_ids or ():
        eligible &= input_ids.ne(int(token_id))

    for row in range(input_ids.shape[0]):
        eligible_count = int(eligible[row].sum().item())
        if eligible_count == 0:
            continue
        ratio = float(
            torch.empty((), device=input_ids.device).uniform_(min_mask_ratio, max_mask_ratio).item()
        )
        target_tokens = min(eligible_count, max(1, int(round(eligible_count * ratio))))
        region_count = min(
            target_tokens,
            int(torch.randint(min_regions, max_regions + 1, (1,), device=input_ids.device).item()),
        )
        lengths = [target_tokens // region_count] * region_count
        for index in range(target_tokens % region_count):
            lengths[index] += 1
        order = torch.randperm(region_count, device=input_ids.device).tolist()
        lengths = [lengths[index] for index in order]
        selected = _sample_nonoverlapping_regions(eligible[row], lengths)
        target_mask[row] = selected

    context[target_mask] = mask_token_id
    labels = input_ids.masked_fill(~target_mask, -100)
    return context, target_mask, labels


def _sample_nonoverlapping_regions(eligible: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    length = int(eligible.shape[0])
    for _ in range(32):
        selected = torch.zeros_like(eligible, dtype=torch.bool)
        complete = True
        for span_length in lengths:
            available = eligible & ~selected
            candidates = [
                start
                for start in range(0, length - span_length + 1)
                if bool(available[start : start + span_length].all().item())
            ]
            if not candidates:
                complete = False
                break
            choice = int(torch.randint(0, len(candidates), (1,), device=eligible.device).item())
            start = candidates[choice]
            selected[start : start + span_length] = True
        if complete:
            return selected
    raise ValueError("Unable to place the requested non-overlapping mask regions")
