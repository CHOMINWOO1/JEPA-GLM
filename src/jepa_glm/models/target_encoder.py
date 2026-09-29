from __future__ import annotations

import copy

from torch import nn


class TargetEncoder(nn.Module):
    def __init__(self, context_encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(context_encoder)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)
