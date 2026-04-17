from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_experiment_config
from src.paths import PathManager
from src.outputs.logger import setup_logging, get_logger
from src.utils.seed import set_global_seed

from src.data.loader import load_master_schema, iter_raw_cohort_chunks
from src.data.validator import validate_master_schema
from src.data.mapper import build_canonical_dataframe
from src.data.preprocessing import basic_preprocess_dataframe
from src.data.normalization import (
    convert_columns_to_numeric,
    fit_standardization_stats_from_tensor_shards,
    save_column_stats,
)
from src.data.sharding import list_tensor_shards, write_tensor_shard


def maybe_subset_dataframe(df, subset_fraction, seed):
    if subset_fraction is None or subset_fraction >= 1.0:
        return df
    if subset_fraction <= 0:
        raise ValueError("subset_fraction must be > 0 and <= 1.")
    return df.sample(frac=subset_fraction, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood foundation model v1")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--cohort-config-name", type=str, required=True)
    return parser.parse_args()


def _make_train_val_split(
    df: pd.DataFrame,
    train_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.sample(frac=train_fraction, random_state=seed)
    val_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _canonical_feature_names_from_schema(schema) -> list[str]:
    return [feature.canonical_name for feature in schema.canonical_features]


def _process_and_write_tensor_shards(cfg, schema, paths, logger) -> Path:
    """
    Chunked preprocessing pipeline:

    raw CSV chunk
      -> optional subset
      -> canonical mapping
      -> basic preprocessing
      -> train/val split per chunk
      -> numeric conversion
      -> save train/val tensor shards
    """
    chunk_size = getattr(cfg.data, "chunk_size", None)
    if chunk_size is None:
        raise ValueError(
            "cfg.data.chunk_size is required for chunked loading. "
            "Add 'chunk_size' under the 'data' section in your config."
        )

    shard_base_dir = paths.run_dir / "tensor_shards"
    feature_names = _canonical_feature_names_from_schema(schema)
    exclude_columns = getattr(cfg.data, "exclude_columns", [])

    train_shard_index = 0
    val_shard_index = 0

    total_raw_rows = 0
    total_subset_rows = 0
    total_train_rows = 0
    total_val_rows = 0

    logger.info("Writing tensor shards under %s", shard_base_dir)

    for chunk_idx, raw_chunk in enumerate(
        iter_raw_cohort_chunks(
            cfg.data.raw_data_path,
            chunk_size=chunk_size,
        ),
        start=1,
    ):
        logger.info("Processing raw chunk %d with shape %s", chunk_idx, raw_chunk.shape)
        total_raw_rows += len(raw_chunk)

        chunk_df = maybe_subset_dataframe(
            raw_chunk,
            subset_fraction=cfg.data.subset_fraction,
            seed=cfg.experiment.seed + chunk_idx,
        )

        if chunk_df.empty:
            logger.info("Chunk %d became empty after subsetting. Skipping.", chunk_idx)
            continue

        total_subset_rows += len(chunk_df)

        canonical_chunk = build_canonical_dataframe(
            chunk_df,
            schema=schema,
            cohort_name=cfg.cohort.name,
            fill_missing_features=True,
            strict=cfg.data.strict_mapping,
        )

        canonical_chunk = basic_preprocess_dataframe(
            canonical_chunk,
            treat_dash_as_missing=cfg.data.treat_dash_as_missing,
        )

        canonical_chunk = canonical_chunk[feature_names]

        train_chunk, val_chunk = _make_train_val_split(
            canonical_chunk,
            train_fraction=cfg.data.train_fraction,
            seed=cfg.experiment.seed + chunk_idx,
        )

        train_chunk = convert_columns_to_numeric(
            train_chunk,
            exclude_columns=exclude_columns,
        )
        val_chunk = convert_columns_to_numeric(
            val_chunk,
            exclude_columns=exclude_columns,
        )

        if not train_chunk.empty:
            train_result = write_tensor_shard(
                train_chunk,
                base_dir=shard_base_dir,
                split_name="train",
                shard_index=train_shard_index,
                feature_names=feature_names,
                cohort_name=cfg.cohort.name,
            )
            total_train_rows += train_result.num_rows
            logger.info(
                "Saved train shard %d -> %s (rows=%d, features=%d)",
                train_shard_index,
                train_result.shard_path,
                train_result.num_rows,
                train_result.num_features,
            )
            train_shard_index += 1

        if not val_chunk.empty:
            val_result = write_tensor_shard(
                val_chunk,
                base_dir=shard_base_dir,
                split_name="val",
                shard_index=val_shard_index,
                feature_names=feature_names,
                cohort_name=cfg.cohort.name,
            )
            total_val_rows += val_result.num_rows
            logger.info(
                "Saved val shard %d -> %s (rows=%d, features=%d)",
                val_shard_index,
                val_result.shard_path,
                val_result.num_rows,
                val_result.num_features,
            )
            val_shard_index += 1

    logger.info(
        "Finished shard writing: raw_rows=%d, subset_rows=%d, train_rows=%d, val_rows=%d, "
        "train_shards=%d, val_shards=%d",
        total_raw_rows,
        total_subset_rows,
        total_train_rows,
        total_val_rows,
        train_shard_index,
        val_shard_index,
    )

    return shard_base_dir


def _fit_and_save_normalization_stats(
    shard_base_dir: Path,
    paths,
    logger,
) -> Path:
    """
    Fit normalization stats from TRAIN shards only and save them to JSON.
    """
    train_shard_paths = list_tensor_shards(shard_base_dir, "train")
    if not train_shard_paths:
        raise ValueError(f"No train shards found under {shard_base_dir / 'train'}")

    logger.info("Found %d train shards for normalization fitting", len(train_shard_paths))

    stats = fit_standardization_stats_from_tensor_shards(train_shard_paths)

    stats_path = paths.run_dir / "normalization_stats.json"
    save_column_stats(stats, stats_path)

    logger.info("Saved normalization stats to %s", stats_path)
    return stats_path


def run() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        config_dir=Path(args.config_dir),
        cohort_config_name=args.cohort_config_name,
    )

    paths = PathManager.from_config(cfg)
    setup_logging(paths.log_file, to_console=cfg.output.log_to_console)
    logger = get_logger(__name__)

    set_global_seed(cfg.experiment.seed)
    logger.info("Loaded configuration for experiment '%s'", cfg.experiment.name)

    # 1. Load and validate schema
    schema = load_master_schema(cfg.data.schema_path)
    validate_master_schema(schema)

    # 2. Build tensor shards from chunked raw data
    shard_base_dir = _process_and_write_tensor_shards(
        cfg=cfg,
        schema=schema,
        paths=paths,
        logger=logger,
    )

    # 3. Fit normalization stats from train shards only
    stats_path = _fit_and_save_normalization_stats(
        shard_base_dir=shard_base_dir,
        paths=paths,
        logger=logger,
    )

    logger.info(
        "Shard preparation completed. Tensor shards saved in %s",
        shard_base_dir,
    )
    logger.info(
        "Normalization stats prepared at %s",
        stats_path,
    )
    logger.info("Run completed. Outputs saved in %s", paths.run_dir)


if __name__ == "__main__":
    run()
