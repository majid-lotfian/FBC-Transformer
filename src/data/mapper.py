from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .schema import MasterSchema


def normalize_name(value: object) -> str:
    """
    Normalize dataset column names the same way as schema names:
    - strip
    - lowercase
    - replace one-or-more spaces with '_'
    - keep punctuation
    """
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [normalize_name(col) for col in normalized_df.columns]
    return normalized_df


def get_cohort_rename_dict(schema: MasterSchema, cohort_name: str) -> Dict[str, str]:
    """
    Return mapping:
        cohort_feature_name -> canonical_name
    for all features present in the given cohort.
    """
    if cohort_name not in schema.cohort_mappings:
        available = schema.get_cohort_names()
        raise ValueError(
            f"Unknown cohort '{cohort_name}'. Available cohorts: {available}"
        )

    present_features = schema.get_present_features_for_cohort(cohort_name)

    rename_dict = {
        cohort_feature_name: canonical_name
        for canonical_name, cohort_feature_name in present_features.items()
    }

    return rename_dict


def get_expected_source_columns(schema: MasterSchema, cohort_name: str) -> List[str]:
    """
    Return the normalized source column names expected in the raw cohort dataset
    for features that are present in this cohort.
    """
    return sorted(get_cohort_rename_dict(schema, cohort_name).keys())


def get_expected_canonical_columns(schema: MasterSchema, cohort_name: str) -> List[str]:
    """
    Return the canonical feature names expected after mapping for features
    that are present in this cohort.
    """
    return sorted(schema.get_present_features_for_cohort(cohort_name).keys())


def map_cohort_dataframe_to_canonical(
    df: pd.DataFrame,
    schema: MasterSchema,
    cohort_name: str,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Normalize raw dataframe headers and map cohort-specific feature names
    into canonical feature names.

    Parameters
    ----------
    df:
        Raw cohort dataframe.
    schema:
        Loaded master schema.
    cohort_name:
        Cohort to use for mapping.
    strict:
        If True, raise an error when any expected mapped source columns are missing.
        If False, map only columns that are actually present.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names only.
    """
    normalized_df = normalize_dataframe_columns(df)
    rename_dict = get_cohort_rename_dict(schema, cohort_name)

    available_source_columns = [col for col in normalized_df.columns if col in rename_dict]
    expected_source_columns = set(rename_dict.keys())
    missing_source_columns = sorted(expected_source_columns - set(available_source_columns))

    if strict and missing_source_columns:
        raise ValueError(
            f"Cohort '{cohort_name}' is missing expected source columns after normalization: "
            f"{missing_source_columns}"
        )

    mapped_df = normalized_df[available_source_columns].copy()
    mapped_df = mapped_df.rename(columns=rename_dict)

    # Final safety check: no duplicate canonical columns
    if mapped_df.columns.duplicated().any():
        duplicated = mapped_df.columns[mapped_df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate canonical columns produced after mapping for cohort '{cohort_name}': "
            f"{duplicated}"
        )

    return mapped_df


def build_canonical_dataframe(
    df: pd.DataFrame,
    schema: MasterSchema,
    cohort_name: str,
    *,
    fill_missing_features: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Map a cohort dataframe to canonical names and optionally add absent canonical columns
    as missing values.

    This is useful if later parts of the pipeline expect the full canonical feature space.
    """
    mapped_df = map_cohort_dataframe_to_canonical(
        df=df,
        schema=schema,
        cohort_name=cohort_name,
        strict=strict,
    )

    if not fill_missing_features:
        return mapped_df

    canonical_order = schema.get_canonical_names()
    result_df = mapped_df.copy()

    for canonical_name in canonical_order:
        if canonical_name not in result_df.columns:
            result_df[canonical_name] = pd.NA

    result_df = result_df[canonical_order]
    return result_df
