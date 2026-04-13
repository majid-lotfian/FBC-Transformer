from __future__ import annotations

import pandas as pd

from .schema import MasterSchema


def get_cohort_rename_dict(schema: MasterSchema, cohort_name: str) -> dict[str, str]:
    present_features = schema.get_present_features_for_cohort(cohort_name)

    # reverse mapping: cohort column -> canonical column
    return {
        cohort_feature_name: canonical_name
        for canonical_name, cohort_feature_name in present_features.items()
    }


def map_cohort_dataframe_to_canonical(
    df: pd.DataFrame,
    schema: MasterSchema,
    cohort_name: str,
) -> pd.DataFrame:
    rename_dict = get_cohort_rename_dict(schema, cohort_name)

    available_columns = [col for col in df.columns if col in rename_dict]
    mapped_df = df[available_columns].copy()
    mapped_df = mapped_df.rename(columns=rename_dict)

    return mapped_df
