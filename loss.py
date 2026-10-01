from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 0.03,
                          margin: float = 0.05) -> Tuple[torch.Tensor, int]:
    """L_rank (Eq. 6): mean hinge max(0, m - sign(y_i - y_j)(y_hat_i - y_hat_j)) over P = {(i, j) : |y_i - y_j| > epsilon}."""
    pred, target = pred.reshape(-1), target.reshape(-1)
    target_diff = target[:, None] - target[None, :]
    pairs = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1) & (target_diff.abs() > epsilon)
    if not pairs.any():
        return pred.new_zeros(()), 0
    pred_diff = pred[:, None] - pred[None, :]
    hinge = torch.clamp(margin - torch.sign(target_diff[pairs]) * pred_diff[pairs], min=0.0)
    return hinge.mean(), int(pairs.sum())


class AestheticsAwareLoss(nn.Module):
    """L = L_Huber(y_hat, y) + lambda_rank * L_rank (Eq. 5) with delta = 1, lambda_rank = 0.1, m = 0.05, epsilon = 0.03."""

    def __init__(self, delta: float = 1.0, lambda_rank: float = 0.1, epsilon: float = 0.03,
                 margin: float = 0.05) -> None:
        super().__init__()
        self.delta = delta
        self.lambda_rank = lambda_rank
        self.epsilon = epsilon
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """pred, target: [B, 1]. Returns the total loss and float statistics for logging."""
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must share a shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
        loss_huber = F.huber_loss(pred, target, delta=self.delta)
        loss_rank, num_pairs = pairwise_ranking_loss(pred, target, self.epsilon, self.margin)
        loss = loss_huber + self.lambda_rank * loss_rank
        return loss, {
            "loss": float(loss.detach()),
            "loss_huber": float(loss_huber.detach()),
            "loss_rank": float(loss_rank.detach()),
            "rank_pairs": float(num_pairs),
        }
