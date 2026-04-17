from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


@dataclass
class ShardWriteResult:
    shard_path: Path
    split_name: str
    shard_index: int
    num_rows: int
    num_features: int


def _validate_split_name(split_name: str) -> None:
    if split_name not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported split_name '{split_name}'. "
            "Expected one of: 'train', 'val', 'test'."
        )


def _make_shard_filename(split_name: str, shard_index: int) -> str:
    return f"{split_name}_shard_{shard_index:05d}.pt"


def ensure_shard_dir(base_dir: str | Path, split_name: str) -> Path:
    """
    Ensure the output directory for a given split exists.

    Example:
        base_dir=artifacts/run_x/shards
        split_name=train
        -> artifacts/run_x/shards/train
    """
    _validate_split_name(split_name)

    base_dir = Path(base_dir)
    split_dir = base_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def dataframe_to_tensor_payload(
    df: pd.DataFrame,
    *,
    feature_names: Optional[list[str]] = None,
    cohort_name: Optional[str] = None,
    sample_ids: Optional[list[str]] = None,
) -> dict:
    """
    Convert a processed dataframe into a tensor shard payload.

    Assumptions:
    - df is already canonicalized and preprocessed
    - numeric conversion / normalization may or may not already be done
    - all columns in feature_names are intended model features

    The payload structure is intentionally simple and future-proof for lazy loading.
    """
    if df.empty:
        raise ValueError("Cannot create a tensor payload from an empty dataframe.")

    if feature_names is None:
        feature_names = list(df.columns)

    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataframe is missing requested feature columns: {missing_features}"
        )

    feature_df = df[feature_names].copy()

    values = torch.tensor(
        feature_df.to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    observed_mask = ~torch.isnan(values)

    num_rows, num_features = values.shape

    if sample_ids is not None and len(sample_ids) != num_rows:
        raise ValueError(
            f"Length mismatch: sample_ids has {len(sample_ids)} items but "
            f"dataframe has {num_rows} rows."
        )

    payload = {
        "values": values,  # [N, F]
        "observed_mask": observed_mask,  # [N, F]
        "feature_names": feature_names,
        "cohort_name": cohort_name,
        "sample_ids": sample_ids,
        "num_rows": num_rows,
        "num_features": num_features,
    }
    return payload


def write_tensor_shard(
    df: pd.DataFrame,
    *,
    base_dir: str | Path,
    split_name: str,
    shard_index: int,
    feature_names: Optional[list[str]] = None,
    cohort_name: Optional[str] = None,
    sample_ids: Optional[list[str]] = None,
) -> ShardWriteResult:
    """
    Save one processed dataframe chunk as a `.pt` tensor shard.

    Output layout:
        <base_dir>/<split_name>/<split_name>_shard_00001.pt
    """
    _validate_split_name(split_name)

    if shard_index < 0:
        raise ValueError("shard_index must be >= 0.")

    split_dir = ensure_shard_dir(base_dir, split_name)
    shard_path = split_dir / _make_shard_filename(split_name, shard_index)

    payload = dataframe_to_tensor_payload(
        df,
        feature_names=feature_names,
        cohort_name=cohort_name,
        sample_ids=sample_ids,
    )

    torch.save(payload, shard_path)

    return ShardWriteResult(
        shard_path=shard_path,
        split_name=split_name,
        shard_index=shard_index,
        num_rows=payload["num_rows"],
        num_features=payload["num_features"],
    )


def list_tensor_shards(
    base_dir: str | Path,
    split_name: str,
) -> list[Path]:
    """
    List shard files for a given split in sorted order.
    """
    _validate_split_name(split_name)

    split_dir = Path(base_dir) / split_name
    if not split_dir.exists():
        return []

    return sorted(split_dir.glob(f"{split_name}_shard_*.pt"))
