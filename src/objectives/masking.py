from __future__ import annotations

import torch


def get_prediction_mask(batch: dict) -> torch.Tensor:
    """
    Returns the mask of positions the model should predict.
    Shape: [B, F]
    """
    prediction_mask = batch.get("prediction_mask")
    if prediction_mask is None:
        raise KeyError("Batch is missing 'prediction_mask'.")
    return prediction_mask.bool()


def get_masked_targets(batch: dict) -> torch.Tensor:
    """
    Returns masked targets.
    Shape: [B, F]
    """
    masked_targets = batch.get("masked_targets")
    if masked_targets is None:
        raise KeyError("Batch is missing 'masked_targets'.")
    return masked_targets


def get_reconstruction_inputs(batch: dict) -> dict[str, torch.Tensor]:
    """
    Extract the model inputs needed for masked feature modeling.
    """
    required = ["feature_ids", "values", "observed_mask", "input_mask"]
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing required reconstruction inputs: {missing}")

    return {
        "feature_ids": batch["feature_ids"],
        "values": batch["values"],
        "observed_mask": batch["observed_mask"],
        "input_mask": batch["input_mask"],
    }
