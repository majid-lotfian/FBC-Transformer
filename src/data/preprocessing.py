from __future__ import annotations

import pandas as pd


def basic_preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = df.copy()

    missing_tokens = {
        "",
        " ",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "-",
    }

    processed_df = processed_df.replace(list(missing_tokens), pd.NA)

    return processed_df
