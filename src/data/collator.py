from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from src.data.dataset import SampleItem
from src.objectives.masking import build_masked_targets


@dataclass
class MaskedModelingCollator:
    feature_names: List[str]
    mask_ratio: float

    def __call__(self, batch: List[SampleItem]) -> Dict[str, torch.Tensor]:
        values = torch.stack([item.values for item in batch], dim=0)
        observed_mask = torch.stack([item.observed_mask for item in batch], dim=0)
        cohort_ids = torch.tensor([item.cohort_id for item in batch], dtype=torch.long)
        feature_ids = torch.arange(len(self.feature_names), dtype=torch.long).unsqueeze(0).repeat(values.size(0), 1)

        masked_values, training_mask, targets = build_masked_targets(
            values=values,
            observed_mask=observed_mask,
            mask_ratio=self.mask_ratio,
        )
        state_ids = torch.zeros_like(feature_ids, dtype=torch.long)
        state_ids[~observed_mask] = 1
        state_ids[training_mask] = 2

        return {
            "feature_ids": feature_ids,
            "values": masked_values,
            "observed_mask": observed_mask,
            "training_mask": training_mask,
            "state_ids": state_ids,
            "targets": targets,
            "cohort_ids": cohort_ids,
        }
