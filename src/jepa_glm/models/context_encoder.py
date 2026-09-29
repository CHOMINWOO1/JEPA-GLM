from __future__ import annotations

from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids, attention_mask=None):
        return self.backbone(input_ids, attention_mask)
