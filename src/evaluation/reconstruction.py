from __future__ import annotations

from typing import Dict

import torch


def reconstruction_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    mask = mask.bool()
    valid_mask = mask & torch.isfinite(targets)

    if valid_mask.sum() == 0:
        return {"masked_mae": 0.0}

    mae = torch.mean(torch.abs(predictions[valid_mask] - targets[valid_mask])).item()
    return {"masked_mae": float(mae)}
