from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import (
    ColumnStats,
    apply_standardization_to_tensor_values,
    load_column_stats,
)
from .sharding import list_tensor_shards


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

    Supports two modes:

    1) DataFrame mode
       - input: df=...
       - keeps backward compatibility with the original pipeline

    2) Shard mode
       - input: shard_base_dir=..., split_name=...
       - reads `.pt` tensor shards lazily
       - optionally applies normalization
       - supports shard-local row shuffling for fast training
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        feature_names: Optional[list[str]] = None,
        sample_id_column: Optional[str] = None,
        cohort_name: Optional[str] = None,
        shard_base_dir: Optional[str | Path] = None,
        split_name: Optional[str] = None,
        normalization_stats_path: Optional[str | Path] = None,
        clip_value: float = 10.0,
        shuffle_within_shard: bool = False,
        shard_shuffle_seed: int = 0,
    ) -> None:
        self.sample_id_column = sample_id_column
        self.cohort_name = cohort_name
        self.clip_value = clip_value
        self.shuffle_within_shard = shuffle_within_shard
        self.shard_shuffle_seed = shard_shuffle_seed

        # shared output metadata
        self.feature_names: list[str]
        self.normalization_stats: Optional[dict[str, ColumnStats]] = None

        # mode flags
        self._mode: str
        self._current_shard_idx: Optional[int] = None
        self._current_shard_payload: Optional[dict] = None

        if normalization_stats_path is not None:
            self.normalization_stats = load_column_stats(normalization_stats_path)

        if df is not None:
            if shard_base_dir is not None or split_name is not None:
                raise ValueError(
                    "Use either DataFrame mode (df=...) or shard mode "
                    "(shard_base_dir=..., split_name=...), not both."
                )
            self._init_from_dataframe(
                df=df,
                feature_names=feature_names,
                sample_id_column=sample_id_column,
                cohort_name=cohort_name,
            )
        else:
            if shard_base_dir is None or split_name is None:
                raise ValueError(
                    "When df is not provided, shard_base_dir and split_name are required."
                )
            self._init_from_shards(
                shard_base_dir=shard_base_dir,
                split_name=split_name,
                feature_names=feature_names,
                cohort_name=cohort_name,
            )

    # ============================================================
    # DataFrame mode
    # ============================================================
    def _init_from_dataframe(
        self,
        *,
        df: pd.DataFrame,
        feature_names: Optional[list[str]],
        sample_id_column: Optional[str],
        cohort_name: Optional[str],
    ) -> None:
        self._mode = "dataframe"

        self.df = df.reset_index(drop=True).copy()

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

        if self.normalization_stats is not None:
            self.values = apply_standardization_to_tensor_values(
                self.values,
                self.observed_mask,
                self.feature_names,
                self.normalization_stats,
                clip_value=self.clip_value,
            )

        self._length = len(self.df)

    # ============================================================
    # Shard mode
    # ============================================================
    def _init_from_shards(
        self,
        *,
        shard_base_dir: str | Path,
        split_name: str,
        feature_names: Optional[list[str]],
        cohort_name: Optional[str],
    ) -> None:
        self._mode = "shard"

        self.shard_base_dir = Path(shard_base_dir)
        self.split_name = split_name

        self.shard_paths = list_tensor_shards(self.shard_base_dir, split_name)
        if not self.shard_paths:
            raise ValueError(
                f"No shard files found for split '{split_name}' under "
                f"{self.shard_base_dir / split_name}"
            )

        self.shard_row_counts: list[int] = []
        self.shard_cumulative_ends: list[int] = []

        running_total = 0
        inferred_feature_names: Optional[list[str]] = None

        for shard_path in self.shard_paths:
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)

            shard_feature_names = payload["feature_names"]
            shard_num_rows = int(payload["num_rows"])

            if inferred_feature_names is None:
                inferred_feature_names = list(shard_feature_names)
            else:
                if list(shard_feature_names) != inferred_feature_names:
                    raise ValueError(
                        f"Feature name mismatch across shards. Problematic shard: {shard_path}"
                    )

            self.shard_row_counts.append(shard_num_rows)
            running_total += shard_num_rows
            self.shard_cumulative_ends.append(running_total)

        if inferred_feature_names is None:
            raise ValueError("Unable to infer feature_names from shard files.")

        if feature_names is not None and feature_names != inferred_feature_names:
            raise ValueError(
                "Provided feature_names do not match shard feature_names."
            )

        self.feature_names = inferred_feature_names
        self._length = running_total

        if cohort_name is None:
            first_payload = torch.load(
                self.shard_paths[0],
                map_location="cpu",
                weights_only=True,
            )
            self.cohort_name = first_payload.get("cohort_name")
        else:
            self.cohort_name = cohort_name

    def _load_shard_payload(self, shard_idx: int) -> dict:
        """
        Load one shard, optionally normalize the whole shard once,
        and optionally shuffle row order within the shard once.

        This is much faster than:
        - global random access across shards
        - per-sample normalization inside __getitem__
        """
        if self._current_shard_idx == shard_idx and self._current_shard_payload is not None:
            return self._current_shard_payload

        payload = torch.load(
            self.shard_paths[shard_idx],
            map_location="cpu",
            weights_only=True,
        )

        values = payload["values"].clone().to(dtype=torch.float32)
        observed_mask = payload["observed_mask"].clone().bool()
        sample_ids = payload.get("sample_ids")

        # Normalize the whole shard once, not one row at a time
        if self.normalization_stats is not None:
            values = apply_standardization_to_tensor_values(
                values,
                observed_mask,
                self.feature_names,
                self.normalization_stats,
                clip_value=self.clip_value,
            )

        # Shuffle row order within shard once (useful for training while keeping
        # DataLoader shuffle=False to avoid slow cross-shard random access)
        if self.shuffle_within_shard and self.split_name == "train":
            generator = torch.Generator()
            generator.manual_seed(self.shard_shuffle_seed + shard_idx)

            perm = torch.randperm(values.shape[0], generator=generator)
            values = values[perm]
            observed_mask = observed_mask[perm]

            if sample_ids is not None:
                sample_ids = [sample_ids[i] for i in perm.tolist()]

        prepared_payload = {
            "values": values,
            "observed_mask": observed_mask,
            "feature_names": payload["feature_names"],
            "cohort_name": payload.get("cohort_name", self.cohort_name),
            "sample_ids": sample_ids,
            "num_rows": payload["num_rows"],
            "num_features": payload["num_features"],
        }

        self._current_shard_idx = shard_idx
        self._current_shard_payload = prepared_payload
        return prepared_payload

    def _global_index_to_shard_position(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._length:
            raise IndexError(
                f"Index {index} out of range for dataset of length {self._length}"
            )

        shard_idx = bisect_right(self.shard_cumulative_ends, index)
        shard_start = 0 if shard_idx == 0 else self.shard_cumulative_ends[shard_idx - 1]
        local_idx = index - shard_start
        return shard_idx, local_idx

    # ============================================================
    # PyTorch Dataset API
    # ============================================================
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict:
        if self._mode == "dataframe":
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

        if self._mode == "shard":
            shard_idx, local_idx = self._global_index_to_shard_position(index)
            payload = self._load_shard_payload(shard_idx)

            row_values = payload["values"][local_idx].clone()
            row_observed_mask = payload["observed_mask"][local_idx].clone()

            sample_ids = payload.get("sample_ids")
            sample_id = None
            if sample_ids is not None:
                sample_id = sample_ids[local_idx]

            sample = TabularSample(
                values=row_values,
                observed_mask=row_observed_mask,
                feature_names=self.feature_names,
                sample_id=sample_id,
                cohort_name=payload.get("cohort_name", self.cohort_name),
            )

            return {
                "values": sample.values,
                "observed_mask": sample.observed_mask,
                "feature_names": sample.feature_names,
                "sample_id": sample.sample_id,
                "cohort_name": sample.cohort_name,
            }

        raise RuntimeError(f"Unknown dataset mode: {self._mode}")
