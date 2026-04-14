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
            return predictions.sum() * 0.0

        diff = predictions[valid_mask] - targets[valid_mask]
        loss = diff.pow(2)

        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()


class MaskedMAEMetric(nn.Module):
    """
    Mean absolute error computed only on prediction_mask positions.
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


class ObjectiveManager:
    """
    Bridges batch -> model inputs -> masked reconstruction loss/metrics.
    """

    def __init__(self, reconstruction_loss_weight: float = 1.0) -> None:
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.reconstruction_loss_fn = MaskedMSELoss(reduction="mean")
        self.reconstruction_mae_fn = MaskedMAEMetric()

    def get_model_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Extract the exact inputs expected by TabularFoundationModel.forward().
        """
        return {
            "feature_ids": batch["feature_ids"],
            "values": batch["values"],
            "observed_mask": batch["observed_mask"],
            "input_mask": batch["input_mask"],
        }

    def compute_total_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute total loss and reporting metrics.
        """
        predictions = model_outputs["reconstruction"]      # [B, F]
        targets = batch["masked_targets"]                  # [B, F], NaN outside masked positions
        prediction_mask = batch["prediction_mask"]         # [B, F]

        reconstruction_loss = self.reconstruction_loss_fn(
            predictions=predictions,
            targets=targets,
            prediction_mask=prediction_mask,
        )

        reconstruction_mae = self.reconstruction_mae_fn(
            predictions=predictions,
            targets=targets,
            prediction_mask=prediction_mask,
        )

        total_loss = self.reconstruction_loss_weight * reconstruction_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_mae": reconstruction_mae,
        }
