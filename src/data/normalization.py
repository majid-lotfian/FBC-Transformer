from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class Normalizer:
    enabled: bool = True
    mode: str = "zscore"
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> None:
        if not self.enabled:
            return
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            self.means[col] = float(df[col].mean())
            std = float(df[col].std()) if float(df[col].std()) > 0 else 1.0
            self.stds[col] = std

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return df.copy()
        out = df.copy()
        for col, mean in self.means.items():
            if col in out.columns:
                std = self.stds[col]
                out[col] = (out[col] - mean) / std
        return out
