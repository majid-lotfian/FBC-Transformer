from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch


@dataclass
class ColumnStats:
    mean: float
    std: float
    min_value: float
    max_value: float
    n_observed: int


# ============================================================
# Existing DataFrame-based utilities
# ============================================================
def convert_columns_to_numeric(
    df: pd.DataFrame,
    *,
    exclude_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convert dataframe columns to numeric where intended.
    Non-convertible values become NaN.
    """
    numeric_df = df.copy()
    exclude = set(exclude_columns or [])

    for column in numeric_df.columns:
        if column in exclude:
            continue
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    return numeric_df


def fit_standardization_stats(
    df: pd.DataFrame,
    *,
    exclude_columns: Optional[list[str]] = None,
) -> Dict[str, ColumnStats]:
    """
    Compute per-column statistics for standardization.
    Ignores missing values.
    """
    exclude = set(exclude_columns or [])
    stats: Dict[str, ColumnStats] = {}

    for column in df.columns:
        if column in exclude:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        mean = float(series.mean())
        std = float(series.std(ddof=0))
        min_value = float(series.min())
        max_value = float(series.max())
        n_observed = int(series.shape[0])

        stats[column] = ColumnStats(
            mean=mean,
            std=std,
            min_value=min_value,
            max_value=max_value,
            n_observed=n_observed,
        )

    return stats


def apply_standardization(
    df: pd.DataFrame,
    stats: Dict[str, ColumnStats],
    *,
    exclude_columns: Optional[list[str]] = None,
    clip_value: float = 10.0,
) -> pd.DataFrame:
    """
    Apply z-score standardization using precomputed stats.
    If std == 0, column is centered only.
    """
    standardized_df = df.copy()
    exclude = set(exclude_columns or [])

    for column in standardized_df.columns:
        if column in exclude:
            continue
        if column not in stats:
            continue

        series = pd.to_numeric(standardized_df[column], errors="coerce")
        column_stats = stats[column]

        if column_stats.std > 0:
            standardized = (series - column_stats.mean) / column_stats.std
        else:
            standardized = series - column_stats.mean

        standardized = standardized.clip(-clip_value, clip_value)
        standardized_df[column] = standardized

    return standardized_df


def fit_and_apply_standardization(
    df: pd.DataFrame,
    *,
    exclude_columns: Optional[list[str]] = None,
    clip_value: float = 10.0,
) -> tuple[pd.DataFrame, Dict[str, ColumnStats]]:
    """
    Convenience function:
    1. convert to numeric
    2. fit stats
    3. apply z-score standardization
    """
    numeric_df = convert_columns_to_numeric(df, exclude_columns=exclude_columns)
    stats = fit_standardization_stats(numeric_df, exclude_columns=exclude_columns)
    standardized_df = apply_standardization(
        numeric_df,
        stats,
        exclude_columns=exclude_columns,
        clip_value=clip_value,
    )
    return standardized_df, stats


# ============================================================
# New shard-based utilities
# ============================================================
def save_column_stats(
    stats: Dict[str, ColumnStats],
    output_path: str | Path,
) -> Path:
    """
    Save normalization stats to JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        column_name: asdict(column_stats)
        for column_name, column_stats in stats.items()
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    return output_path


def load_column_stats(
    input_path: str | Path,
) -> Dict[str, ColumnStats]:
    """
    Load normalization stats from JSON.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    stats = {
        column_name: ColumnStats(**column_stats_dict)
        for column_name, column_stats_dict in raw.items()
    }
    return stats


def fit_standardization_stats_from_tensor_shards(
    shard_paths: list[str | Path],
) -> Dict[str, ColumnStats]:
    """
    Compute per-feature normalization stats from tensor shard files.

    Assumes each shard payload contains:
    - values: torch.Tensor [N, F]
    - observed_mask: torch.Tensor [N, F]
    - feature_names: list[str]

    Stats are fitted only on observed values.
    """
    if not shard_paths:
        raise ValueError("No shard paths were provided.")

    feature_names: list[str] | None = None

    running_sum: torch.Tensor | None = None
    running_sumsq: torch.Tensor | None = None
    running_count: torch.Tensor | None = None
    running_min: torch.Tensor | None = None
    running_max: torch.Tensor | None = None

    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        payload = torch.load(shard_path, map_location="cpu", weights_only=True)

        values = payload["values"].to(dtype=torch.float64)           # [N, F]
        observed_mask = payload["observed_mask"].bool()              # [N, F]
        shard_feature_names = payload["feature_names"]

        if feature_names is None:
            feature_names = list(shard_feature_names)
            num_features = len(feature_names)

            running_sum = torch.zeros(num_features, dtype=torch.float64)
            running_sumsq = torch.zeros(num_features, dtype=torch.float64)
            running_count = torch.zeros(num_features, dtype=torch.float64)
            running_min = torch.full((num_features,), float("inf"), dtype=torch.float64)
            running_max = torch.full((num_features,), float("-inf"), dtype=torch.float64)
        else:
            if list(shard_feature_names) != feature_names:
                raise ValueError(
                    f"Feature name mismatch in shard {shard_path}. "
                    "All shards must share the same feature_names ordering."
                )

        assert running_sum is not None
        assert running_sumsq is not None
        assert running_count is not None
        assert running_min is not None
        assert running_max is not None

        observed_values = torch.where(observed_mask, values, torch.zeros_like(values))

        running_sum += observed_values.sum(dim=0)
        running_sumsq += (observed_values ** 2).sum(dim=0)
        running_count += observed_mask.sum(dim=0).to(dtype=torch.float64)

        # per-feature min/max over observed entries only
        # use inf/-inf for unobserved positions so reduction ignores them
        min_candidates = torch.where(
            observed_mask,
            values,
            torch.full_like(values, float("inf")),
        )
        max_candidates = torch.where(
            observed_mask,
            values,
            torch.full_like(values, float("-inf")),
        )

        shard_min = min_candidates.min(dim=0).values
        shard_max = max_candidates.max(dim=0).values

        running_min = torch.minimum(running_min, shard_min)
        running_max = torch.maximum(running_max, shard_max)

    assert feature_names is not None
    assert running_sum is not None
    assert running_sumsq is not None
    assert running_count is not None
    assert running_min is not None
    assert running_max is not None

    stats: Dict[str, ColumnStats] = {}

    for idx, feature_name in enumerate(feature_names):
        count = int(running_count[idx].item())

        if count == 0:
            # no observed data for this feature across all shards
            continue

        mean = float((running_sum[idx] / running_count[idx]).item())

        variance = (running_sumsq[idx] / running_count[idx]) - (mean ** 2)
        variance = max(float(variance.item()), 0.0)
        std = variance ** 0.5

        min_value = float(running_min[idx].item())
        max_value = float(running_max[idx].item())

        stats[feature_name] = ColumnStats(
            mean=mean,
            std=std,
            min_value=min_value,
            max_value=max_value,
            n_observed=count,
        )

    return stats


def apply_standardization_to_tensor_values(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    feature_names: list[str],
    stats: Dict[str, ColumnStats],
    *,
    clip_value: float = 10.0,
) -> torch.Tensor:
    """
    Apply z-score standardization directly to a tensor of values.

    Parameters
    ----------
    values:
        Tensor of shape [N, F] or [F].
    observed_mask:
        Bool tensor with same shape as values. True means the original value was observed.
    feature_names:
        Feature order matching the tensor columns.
    stats:
        Per-feature normalization stats.
    clip_value:
        Final clipping range applied after normalization.

    Returns
    -------
    torch.Tensor
        Standardized tensor with the same shape as input values.
        Unobserved entries are left unchanged.
    """
    if values.shape != observed_mask.shape:
        raise ValueError(
            f"Shape mismatch: values {values.shape} != observed_mask {observed_mask.shape}"
        )

    output = values.clone().to(dtype=torch.float32)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    for feature_name, column_stats in stats.items():
        if feature_name not in feature_index:
            continue

        col_idx = feature_index[feature_name]
        col_values = output[..., col_idx]
        col_mask = observed_mask[..., col_idx].bool()

        if column_stats.std > 0:
            standardized = (col_values - column_stats.mean) / column_stats.std
        else:
            standardized = col_values - column_stats.mean

        standardized = torch.clamp(standardized, -clip_value, clip_value)

        output[..., col_idx] = torch.where(col_mask, standardized, col_values)

    return output
