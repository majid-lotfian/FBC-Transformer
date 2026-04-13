from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TabularSample:
    values: torch.Tensor
    observed_mask: torch.Tensor
    feature_names: list[str]
    sample_id: Optional[str] = None
    cohort_name: Optional[str] = None


class TabularFoundationDataset(Dataset):
    """
    Row-wise dataset for canonical tabular data.

    Expected input dataframe:
    - one row = one sample
    - columns = canonical feature names (plus optional excluded columns not passed in)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_names: Optional[list[str]] = None,
        sample_id_column: Optional[str] = None,
        cohort_name: Optional[str] = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.sample_id_column = sample_id_column
        self.cohort_name = cohort_name

        if feature_names is None:
            feature_names = list(self.df.columns)

        if sample_id_column is not None and sample_id_column in feature_names:
            raise ValueError(
                f"sample_id_column '{sample_id_column}' must not be included in feature_names."
            )

        missing_features = [col for col in feature_names if col not in self.df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset is missing requested feature columns: {missing_features}"
            )

        self.feature_names = feature_names
        self.feature_df = self.df[self.feature_names]

        self.values = torch.tensor(
            self.feature_df.to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        self.observed_mask = ~torch.isnan(self.values)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row_values = self.values[index]
        row_observed_mask = self.observed_mask[index]

        sample_id = None
        if self.sample_id_column is not None:
            sample_id = str(self.df.iloc[index][self.sample_id_column])

        sample = TabularSample(
            values=row_values,
            observed_mask=row_observed_mask,
            feature_names=self.feature_names,
            sample_id=sample_id,
            cohort_name=self.cohort_name,
        )

        return {
            "values": sample.values,
            "observed_mask": sample.observed_mask,
            "feature_names": sample.feature_names,
            "sample_id": sample.sample_id,
            "cohort_name": sample.cohort_name,
        }
