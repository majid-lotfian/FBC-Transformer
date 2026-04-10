from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_dataframe(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format for raw data: {file_path.suffix}")
