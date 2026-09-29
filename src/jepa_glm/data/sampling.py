from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    """Deterministic epoch shuffling with a batch-aligned resume offset."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        start_step: int = 0,
        batch_size: int = 1,
    ) -> None:
        self.num_samples = len(data_source)
        if self.num_samples <= 0:
            raise ValueError("data_source must contain at least one sample")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_step < 0:
            raise ValueError("start_step cannot be negative")
        batches_per_epoch = math.ceil(self.num_samples / batch_size)
        self.seed = int(seed)
        self.epoch, batch_offset = divmod(int(start_step), batches_per_epoch)
        self.offset = batch_offset * int(batch_size)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.num_samples, generator=generator).tolist()
        offset = self.offset
        self.epoch += 1
        self.offset = 0
        return iter(order[offset:])

    def __len__(self) -> int:
        return self.num_samples - self.offset
