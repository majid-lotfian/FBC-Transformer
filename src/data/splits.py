from __future__ import annotations

import pandas as pd


def make_train_val_dataframes(df: pd.DataFrame, train_fraction: float, seed: int):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(shuffled) * train_fraction)
    train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    val_df = shuffled.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df
