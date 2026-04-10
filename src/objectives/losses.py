from __future__ import annotations

import torch
from torch import nn


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, training_mask: torch.Tensor) -> torch.Tensor:
    if training_mask.sum() == 0:
        return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    return nn.functional.mse_loss(predictions[training_mask], targets[training_mask])
