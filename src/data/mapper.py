from __future__ import annotations

import pandas as pd


def map_to_canonical(raw_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        row["local_name"]: row["canonical_name"]
        for _, row in mapping_df.iterrows()
        if pd.notna(row["canonical_name"])
    }
    mapped_df = raw_df.rename(columns=rename_map).copy()
    return mapped_df
