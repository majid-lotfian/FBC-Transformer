from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class ColumnStats:
    mean: float
    std: float
    min_value: float
    max_value: float
    n_observed: int


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
        
        # clipping
        standardized = standardized.clip(-10.0, 10.0)
        
        standardized_df[column] = standardized
        
        '''if column_stats.std > 0:
            standardized_df[column] = (series - column_stats.mean) / column_stats.std
        else:
            standardized_df[column] = series - column_stats.mean'''

    return standardized_df


def fit_and_apply_standardization(
    df: pd.DataFrame,
    *,
    exclude_columns: Optional[list[str]] = None,
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
    )
    return standardized_df, stats
