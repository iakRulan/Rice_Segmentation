from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_patch.losses_v2 import MultiTaskLoss


class BoundaryConsistencyLoss(nn.Module):
    """Small auxiliary term for fragmented crop boundaries."""

    @staticmethod
    def boundary(x: torch.Tensor) -> torch.Tensor:
        high = F.max_pool2d(x, 3, stride=1, padding=1)
        low = -F.max_pool2d(-x, 3, stride=1, padding=1)
        return high - low

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.boundary(torch.sigmoid(logits)), self.boundary(target))


class SegmentationLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        supported = {
            "bce", "dice", "lovasz", "iou", "tversky", "focal_gamma",
            "pos_weight", "tversky_alpha", "tversky_beta", "tversky_gamma", "smooth"
        }
        base_args = {key: value for key, value in config.items() if key in supported}
        self.region = MultiTaskLoss(**base_args)
        self.boundary_weight = float(config.get("boundary", 0.0))
        self.boundary = BoundaryConsistencyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.region(logits, target)
        if self.boundary_weight:
            loss = loss + self.boundary(logits, target) * self.boundary_weight
        return loss
