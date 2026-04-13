from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import CanonicalFeature, CohortMappingEntry, MasterSchema


# -----------------------------
# Normalization helpers
# -----------------------------
def normalize_name(value: object) -> Optional[str]:
    """
    Normalize feature-like names:
    - convert to string
    - strip leading/trailing whitespace
    - lowercase
    - replace one-or-more whitespace with underscore
    - keep punctuation such as ., ?, /, ^
    - return None for empty / missing cells
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    text = text.lower()
    text = re.sub(r"\s+", "_", text)
    return text


def normalize_text(value: object) -> Optional[str]:
    """
    Normalize general metadata text:
    - convert to string
    - strip leading/trailing whitespace
    - return None for empty / missing cells
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    return text


def normalize_unit(value: object) -> Optional[str]:
    """
    Units are kept as raw text.
    Rules:
    - empty / missing -> None
    - '-' -> None (not applicable)
    - otherwise keep exact text
    """
    text = normalize_text(value)
    if text is None:
        return None
    if text == "-":
        return None
    return text


def normalize_header(value: object) -> str:
    """
    Normalize Excel column headers into internal stable names.
    Example:
    'Channel/Profile' -> 'channel/profile'
    'Exact Name Amsterdam' -> 'exact_name_amsterdam'
    """
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


# -----------------------------
# Column handling
# -----------------------------
def normalize_dataframe_headers(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [normalize_header(col) for col in normalized_df.columns]
    return normalized_df


def infer_cohort_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Detect cohort mapping columns dynamically.

    Expected pattern:
    - canonical_name   -> reserved, not a cohort
    - amsterdam_name   -> cohort 'amsterdam'
    - cambridge_name   -> cohort 'cambridge'
    - any future cohort should follow '<cohort>_name'
    """
    cohort_columns: dict[str, str] = {}

    for column in df.columns:
        if column == "canonical_name":
            continue
        if column.endswith("_name"):
            cohort_name = column[: -len("_name")]
            if cohort_name:
                cohort_columns[cohort_name] = column

    return cohort_columns


# -----------------------------
# Main loader
# -----------------------------
def load_master_schema(schema_path: str | Path) -> MasterSchema:
    """
    Load the master schema from an Excel file.

    Expected minimum columns after header normalization:
    - canonical_name
    - one or more cohort columns ending with '_name'

    Optional metadata columns:
    - subgroup
    - unit
    - channel_profile
    - channel/profile  (will normalize to 'channel/profile', so user should prefer channel_profile)
    """
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    df = pd.read_excel(schema_path, dtype=object)
    if df.empty:
        raise ValueError(f"Schema file is empty: {schema_path}")

    df = normalize_dataframe_headers(df)

    # Backward-compatible support if someone still uses channel/profile
    if "channel/profile" in df.columns and "channel_profile" not in df.columns:
        df = df.rename(columns={"channel/profile": "channel_profile"})

    if "canonical_name" not in df.columns:
        raise ValueError(
            "Schema file must contain a 'canonical_name' column "
            "(or a header that normalizes to it)."
        )

    cohort_columns = infer_cohort_columns(df)
    if not cohort_columns:
        raise ValueError(
            "No cohort mapping columns found. Expected one or more columns like "
            "'amsterdam_name', 'cambridge_name', or '<cohort>_name'."
        )

    canonical_features: list[CanonicalFeature] = []
    cohort_mappings: dict[str, dict[str, CohortMappingEntry]] = {
        cohort_name: {} for cohort_name in cohort_columns
    }

    seen_canonical_names: set[str] = set()

    for row_idx, row in df.iterrows():
        excel_row_number = row_idx + 2  # +2 because Excel starts at 1 and row 1 is header

        canonical_name = normalize_name(row.get("canonical_name"))
        if canonical_name is None:
            # Skip blank canonical rows safely
            continue

        if canonical_name in seen_canonical_names:
            raise ValueError(
                f"Duplicate canonical_name '{canonical_name}' found in schema file "
                f"at Excel row {excel_row_number}."
            )
        seen_canonical_names.add(canonical_name)

        subgroup = normalize_name(row.get("subgroup")) if "subgroup" in df.columns else None
        unit = normalize_unit(row.get("unit")) if "unit" in df.columns else None
        channel_profile = (
            normalize_name(row.get("channel_profile"))
            if "channel_profile" in df.columns
            else None
        )

        canonical_feature = CanonicalFeature(
            canonical_name=canonical_name,
            subgroup=subgroup,
            unit=unit,
            channel_profile=channel_profile,
        )
        canonical_features.append(canonical_feature)

        for cohort_name, cohort_column in cohort_columns.items():
            cohort_feature_name = normalize_name(row.get(cohort_column))

            cohort_mappings[cohort_name][canonical_name] = CohortMappingEntry(
                cohort_name=cohort_name,
                canonical_name=canonical_name,
                cohort_feature_name=cohort_feature_name,  # None => absent in that cohort
            )

    if not canonical_features:
        raise ValueError("No canonical features were loaded from the schema file.")

    return MasterSchema(
        canonical_features=canonical_features,
        cohort_mappings=cohort_mappings,
    )
