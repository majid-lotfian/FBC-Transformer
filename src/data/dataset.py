from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SampleItem:
    values: torch.Tensor
    observed_mask: torch.Tensor
    cohort_id: int


class TabularFeatureDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_names: List[str], cohort_id: int) -> None:
        self.feature_names = feature_names
        self.cohort_id = cohort_id
        self.dataframe = dataframe.copy()

        for feature in feature_names:
            if feature not in self.dataframe.columns:
                self.dataframe[feature] = np.nan

        self.dataframe = self.dataframe[feature_names]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> SampleItem:
        row = self.dataframe.iloc[idx]
        values = torch.tensor(row.fillna(0.0).to_numpy(dtype=float), dtype=torch.float32)
        observed_mask = torch.tensor((~row.isna()).to_numpy(dtype=bool), dtype=torch.bool)
        return SampleItem(values=values, observed_mask=observed_mask, cohort_id=self.cohort_id)

    def make_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        collate_fn: Callable,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
