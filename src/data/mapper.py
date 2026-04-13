from __future__ import annotations

import pandas as pd

from .schema import MasterSchema


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [normalize_name(col) for col in normalized_df.columns]
    return normalized_df


def get_cohort_rename_dict(schema: MasterSchema, cohort_name: str) -> dict[str, str]:
    present_features = schema.get_present_features_for_cohort(cohort_name)

    # cohort column name -> canonical feature name
    return {
        cohort_feature_name: canonical_name
        for canonical_name, cohort_feature_name in present_features.items()
    }


def map_cohort_dataframe_to_canonical(
    df: pd.DataFrame,
    schema: MasterSchema,
    cohort_name: str,
) -> pd.DataFrame:
    normalized_df = normalize_dataframe_columns(df)
    rename_dict = get_cohort_rename_dict(schema, cohort_name)

    available_columns = [col for col in normalized_df.columns if col in rename_dict]
    mapped_df = normalized_df[available_columns].copy()
    mapped_df = mapped_df.rename(columns=rename_dict)

    return mapped_df
