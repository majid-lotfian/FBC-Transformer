from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import CanonicalFeature, CohortMappingEntry, MasterSchema


def normalize_feature_name(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()
    if text == "":
        return None

    return text.replace(" ", "_")


def normalize_unit(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if text == "" or text == "-":
        return None

    return text


def load_master_schema(schema_path: str | Path) -> MasterSchema:
    df = pd.read_excel(schema_path)

    required_columns = [
        "canonical_name",
        "amsterdam_name",
        "cambridge_name",
        "subgroup",
        "unit",
        "channel_profile",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required schema columns: {missing}")

    canonical_features = []
    cohort_mappings = {
        "amsterdam": {},
        "cambridge": {},
    }

    for _, row in df.iterrows():
        canonical_name = normalize_feature_name(row["canonical_name"])
        if canonical_name is None:
            continue

        feature = CanonicalFeature(
            canonical_name=canonical_name,
            subgroup=normalize_feature_name(row["subgroup"]),
            unit=normalize_unit(row["unit"]),
            channel_profile=normalize_feature_name(row["channel_profile"]),
        )
        canonical_features.append(feature)

        for cohort_name, column_name in {
            "amsterdam": "amsterdam_name",
            "cambridge": "cambridge_name",
        }.items():
            cohort_feature_name = normalize_feature_name(row[column_name])

            cohort_mappings[cohort_name][canonical_name] = CohortMappingEntry(
                cohort_name=cohort_name,
                canonical_name=canonical_name,
                cohort_feature_name=cohort_feature_name,
            )

    return MasterSchema(
        canonical_features=canonical_features,
        cohort_mappings=cohort_mappings,
    )
