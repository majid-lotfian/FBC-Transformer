from __future__ import annotations

from typing import Tuple

import torch


def build_masked_targets(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    mask_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    random_mask = torch.rand_like(values).lt(mask_ratio)
    training_mask = random_mask & observed_mask
    masked_values = values.clone()
    masked_values[training_mask] = 0.0
    targets = values.clone()
    return masked_values, training_mask, targets
