from __future__ import annotations

import pandas as pd


def preprocess_dataframe(df: pd.DataFrame, canonical_features: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    valid_features = set(canonical_features["canonical_name"].astype(str))

    for column in list(processed.columns):
        if column not in valid_features:
            continue
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    return processed
