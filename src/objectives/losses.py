from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Mean squared error computed only on prediction_mask positions.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        predictions:     [B, F]
        targets:         [B, F]
        prediction_mask: [B, F] bool
        """
        prediction_mask = prediction_mask.bool()

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} != targets {targets.shape}"
            )
        if predictions.shape != prediction_mask.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} != prediction_mask {prediction_mask.shape}"
            )

        valid_mask = prediction_mask & torch.isfinite(targets)

        if valid_mask.sum() == 0:
            # Keep graph/device/dtype consistent
            return predictions.sum() * 0.0

        diff = predictions[valid_mask] - targets[valid_mask]
        loss = diff.pow(2)

        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()


class MaskedMAEMetric(nn.Module):
    """
    Mean absolute error computed only on prediction_mask positions.
    Useful as a metric, not necessarily as the training loss.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction_mask = prediction_mask.bool()
        valid_mask = prediction_mask & torch.isfinite(targets)

        if valid_mask.sum() == 0:
            return predictions.sum() * 0.0

        diff = (predictions[valid_mask] - targets[valid_mask]).abs()
        return diff.mean()


def compute_masked_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convenience function for the main v1 training loss.
    """
    loss_fn = MaskedMSELoss(reduction="mean")
    return loss_fn(predictions, targets, prediction_mask)
