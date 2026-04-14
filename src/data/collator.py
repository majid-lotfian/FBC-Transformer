from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class FeatureVocabulary:
    feature_names: List[str]

    def __post_init__(self) -> None:
        self.name_to_id = {name: idx for idx, name in enumerate(self.feature_names)}
        self.id_to_name = {idx: name for idx, name in enumerate(self.feature_names)}

    def get_feature_ids(self) -> torch.Tensor:
        return torch.tensor(
            [self.name_to_id[name] for name in self.feature_names],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.feature_names)


class MaskedModelingCollator:
    """
    Build batches for masked feature modeling.

    Output tensors:
    - feature_ids: [B, F]
    - values: [B, F]
    - observed_mask: [B, F]
    - input_mask: [B, F]          -> features available to the model
    - masked_targets: [B, F]      -> original values for masked positions, NaN elsewhere
    - prediction_mask: [B, F]     -> positions used in loss
    """

    
    '''def __init__(
        self,
        feature_names: List[str],
        *,
        masking_ratio: float = 0.15,
        min_masked_features: int = 1,
        seed: Optional[int] = None,
    ) -> None:'''
    def __init__(
        self,
        feature_names: List[str],
        *,
        masking_ratio: float = 0.15,
        min_masked_features: int = 1,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty.")
        if not (0.0 <= masking_ratio <= 1.0):
            raise ValueError("masking_ratio must be between 0 and 1.")
        if min_masked_features < 0:
            raise ValueError("min_masked_features must be >= 0.")

        self.feature_names = feature_names
        self.vocab = FeatureVocabulary(feature_names)
        self.masking_ratio = masking_ratio
        self.min_masked_features = min_masked_features
        self.seed = seed if seed is not None else 0
        self.deterministic = deterministic

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def _stack_tensor_field(
        self,
        batch: List[Dict[str, Any]],
        field_name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensors = [item[field_name] for item in batch]
        return torch.stack(tensors).to(dtype=dtype)

    '''def _build_prediction_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Randomly choose a subset of observed features to mask.

        observed_mask: [B, F] bool
        returns: prediction_mask [B, F] bool
        """
        batch_size, num_features = observed_mask.shape

        random_scores = torch.rand(
            (batch_size, num_features),
            generator=self.generator,
            device=observed_mask.device,
        )

        candidate_mask = observed_mask.clone()
        prediction_mask = torch.zeros_like(observed_mask, dtype=torch.bool)

        for row_idx in range(batch_size):
            observed_indices = torch.where(candidate_mask[row_idx])[0]
            n_observed = int(observed_indices.numel())

            if n_observed == 0:
                continue

            n_to_mask = max(
                self.min_masked_features,
                int(round(n_observed * self.masking_ratio)),
            )
            n_to_mask = min(n_to_mask, n_observed)

            row_scores = random_scores[row_idx, observed_indices]
            sorted_order = torch.argsort(row_scores)
            chosen = observed_indices[sorted_order[:n_to_mask]]

            prediction_mask[row_idx, chosen] = True

        return prediction_mask'''

    def _build_prediction_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Choose a subset of observed features to mask.
    
        If deterministic=False:
            use the running generator (training behavior)
        If deterministic=True:
            build the same mask for the same row position every call (validation behavior)
        """
        batch_size, num_features = observed_mask.shape
        prediction_mask = torch.zeros_like(observed_mask, dtype=torch.bool)
    
        if not self.deterministic:
            random_scores = torch.rand(
                (batch_size, num_features),
                generator=self.generator,
                device=observed_mask.device,
            )
    
            candidate_mask = observed_mask.clone()
    
            for row_idx in range(batch_size):
                observed_indices = torch.where(candidate_mask[row_idx])[0]
                n_observed = int(observed_indices.numel())
    
                if n_observed == 0:
                    continue
    
                n_to_mask = max(
                    self.min_masked_features,
                    int(round(n_observed * self.masking_ratio)),
                )
                n_to_mask = min(n_to_mask, n_observed)
    
                row_scores = random_scores[row_idx, observed_indices]
                sorted_order = torch.argsort(row_scores)
                chosen = observed_indices[sorted_order[:n_to_mask]]
    
                prediction_mask[row_idx, chosen] = True
    
            return prediction_mask
    
        # deterministic validation masking
        for row_idx in range(batch_size):
            observed_indices = torch.where(observed_mask[row_idx])[0]
            n_observed = int(observed_indices.numel())
    
            if n_observed == 0:
                continue
    
            n_to_mask = max(
                self.min_masked_features,
                int(round(n_observed * self.masking_ratio)),
            )
            n_to_mask = min(n_to_mask, n_observed)
    
            row_generator = torch.Generator(device=observed_mask.device)
            row_generator.manual_seed(self.seed + row_idx)
    
            row_scores = torch.rand(
                (n_observed,),
                generator=row_generator,
                device=observed_mask.device,
            )
    
            sorted_order = torch.argsort(row_scores)
            chosen = observed_indices[sorted_order[:n_to_mask]]
    
            prediction_mask[row_idx, chosen] = True
    
        return prediction_mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Batch is empty.")

        feature_names = batch[0]["feature_names"]
        if feature_names != self.feature_names:
            raise ValueError("Batch feature_names do not match collator feature_names.")

        values = self._stack_tensor_field(batch, "values", torch.float32)              # [B, F]
        observed_mask = self._stack_tensor_field(batch, "observed_mask", torch.bool)   # [B, F]

        # Replace naturally missing values with 0.0 for safe model input.
        # observed_mask still keeps track of which values were actually observed.
        values = values.clone()
        values[~observed_mask] = 0.0

        batch_size, num_features = values.shape

        feature_ids_1d = self.vocab.get_feature_ids()                                   # [F]
        feature_ids = feature_ids_1d.unsqueeze(0).repeat(batch_size, 1)                # [B, F]

        prediction_mask = self._build_prediction_mask(observed_mask)                    # [B, F]

        input_mask = observed_mask & (~prediction_mask)                                 # [B, F]

        masked_values = values.clone()
        masked_values[prediction_mask] = 0.0

        masked_targets = torch.full_like(values, float("nan"))
        masked_targets[prediction_mask] = values[prediction_mask]

        sample_ids = [item.get("sample_id") for item in batch]
        cohort_names = [item.get("cohort_name") for item in batch]

        return {
            "feature_ids": feature_ids,             # [B, F]
            "values": masked_values,                # [B, F] model input values
            "original_values": values,              # [B, F] original values
            "observed_mask": observed_mask,         # [B, F]
            "input_mask": input_mask,               # [B, F]
            "prediction_mask": prediction_mask,     # [B, F]
            "masked_targets": masked_targets,       # [B, F]
            "feature_names": self.feature_names,
            "sample_ids": sample_ids,
            "cohort_names": cohort_names,
        }
