from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


DEFAULT_MISSING_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "not_available",
}


def _normalize_cell_to_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def clean_missing_values(
    df: pd.DataFrame,
    *,
    extra_missing_tokens: Optional[Iterable[str]] = None,
    treat_dash_as_missing: bool = False,
) -> pd.DataFrame:
    """
    Replace common textual missing-value placeholders with pd.NA.

    Notes
    -----
    - By default, '-' is NOT treated as missing in data values.
      This is intentional because '-' was defined as 'not applicable' for schema metadata,
      but in raw datasets you may want to decide separately.
    """
    cleaned_df = df.copy()

    missing_tokens = set(DEFAULT_MISSING_TOKENS)
    if extra_missing_tokens is not None:
        missing_tokens.update(str(token).strip().lower() for token in extra_missing_tokens)

    if treat_dash_as_missing:
        missing_tokens.add("-")

    def replace_value(value: object) -> object:
        text = _normalize_cell_to_text(value)
        if text is None:
            return pd.NA
        if text.lower() in missing_tokens:
            return pd.NA
        return value

    for column in cleaned_df.columns:
        cleaned_df[column] = cleaned_df[column].map(replace_value)

    return cleaned_df


def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from string-like cells.
    Leaves non-string values unchanged.
    """
    stripped_df = df.copy()

    def strip_value(value: object) -> object:
        if value is None or pd.isna(value):
            return value
        if isinstance(value, str):
            return value.strip()
        return value

    for column in stripped_df.columns:
        stripped_df[column] = stripped_df[column].map(strip_value)

    return stripped_df


def basic_preprocess_dataframe(
    df: pd.DataFrame,
    *,
    extra_missing_tokens: Optional[Iterable[str]] = None,
    treat_dash_as_missing: bool = False,
) -> pd.DataFrame:
    """
    Basic safe preprocessing for v1:
    1. strip surrounding whitespace from string cells
    2. convert common missing-value placeholders to pd.NA
    """
    processed_df = strip_string_values(df)
    processed_df = clean_missing_values(
        processed_df,
        extra_missing_tokens=extra_missing_tokens,
        treat_dash_as_missing=treat_dash_as_missing,
    )
    return processed_df
