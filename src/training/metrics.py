from __future__ import annotations

from typing import Dict, List

import torch


class RunningAverage:
    """
    Utility to keep running average of a scalar metric.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class MetricTracker:
    """
    Tracks multiple metrics over an epoch.
    """

    def __init__(self, metric_names: List[str]) -> None:
        self.metrics = {name: RunningAverage() for name in metric_names}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            if key in self.metrics:
                self.metrics[key].update(value, n=n)

    def compute(self) -> Dict[str, float]:
        return {name: meter.compute() for name, meter in self.metrics.items()}

    def reset(self) -> None:
        for meter in self.metrics.values():
            meter.reset()


def compute_masked_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> float:
    """
    Compute MAE over masked positions.
    Returns a Python float.
    """
    prediction_mask = prediction_mask.bool()
    valid_mask = prediction_mask & torch.isfinite(targets)

    if valid_mask.sum() == 0:
        return 0.0

    diff = (predictions[valid_mask] - targets[valid_mask]).abs()
    return float(diff.mean().item())


def compute_masked_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> float:
    """
    Compute RMSE over masked positions.
    """
    prediction_mask = prediction_mask.bool()
    valid_mask = prediction_mask & torch.isfinite(targets)

    if valid_mask.sum() == 0:
        return 0.0

    diff = predictions[valid_mask] - targets[valid_mask]
    mse = diff.pow(2).mean()
    rmse = torch.sqrt(mse)

    return float(rmse.item())
