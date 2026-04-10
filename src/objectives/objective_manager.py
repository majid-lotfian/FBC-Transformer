from __future__ import annotations

from typing import Dict

import torch

from src.objectives.losses import masked_mse_loss


class ObjectiveManager:
    def compute(self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reconstruction = model_outputs["reconstruction"]
        targets = batch["targets"]
        training_mask = batch["training_mask"]

        loss_masked = masked_mse_loss(reconstruction, targets, training_mask)
        total_loss = loss_masked

        return {
            "loss": total_loss,
            "loss_masked": loss_masked.detach(),
        }
