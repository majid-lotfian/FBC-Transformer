from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.config import load_experiment_config
from src.paths import PathManager
from src.outputs.logger import setup_logging, get_logger
from src.utils.seed import set_global_seed

from src.data.loader import load_master_schema
from src.data.validator import validate_master_schema
from src.data.mapper import build_canonical_dataframe
from src.data.preprocessing import basic_preprocess_dataframe
from src.data.normalization import fit_and_apply_standardization
from src.data.dataset import TabularFoundationDataset
from src.data.collator import MaskedModelingCollator

from src.models.model import TabularFoundationModel
from src.objectives.objective_manager import ObjectiveManager
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer

from src.outputs.exporter import export_run_summary
from src.outputs.plots import plot_metric_history
from src.outputs.tables import write_metrics_table


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


def _make_train_val_split(df: pd.DataFrame, train_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.sample(frac=train_fraction, random_state=seed)
    val_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


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

    # 2. Load raw cohort data
    raw_df = pd.read_csv(cfg.data.raw_data_path)

    raw_df = maybe_subset_dataframe(
        raw_df,
        subset_fraction=cfg.data.subset_fraction,
        seed=cfg.experiment.seed,
    )

    # 3. Map to canonical feature space
    canonical_df = build_canonical_dataframe(
        raw_df,
        schema=schema,
        cohort_name=cfg.cohort.name,
        fill_missing_features=True,
        strict=cfg.data.strict_mapping,
    )

    # 4. Basic preprocessing
    canonical_df = basic_preprocess_dataframe(
        canonical_df,
        treat_dash_as_missing=cfg.data.treat_dash_as_missing,
    )

    # 5. Train/validation split
    train_df, val_df = _make_train_val_split(
        canonical_df,
        train_fraction=cfg.data.train_fraction,
        seed=cfg.experiment.seed,
    )

    # 6. Normalization (fit on train only)
    train_df, normalization_stats = fit_and_apply_standardization(
        train_df,
        exclude_columns=getattr(cfg.data, "exclude_columns", []),
    )

    # Reuse train stats on val
    from src.data.normalization import convert_columns_to_numeric, apply_standardization
    val_df = convert_columns_to_numeric(
        val_df,
        exclude_columns=getattr(cfg.data, "exclude_columns", []),
    )
    val_df = apply_standardization(
        val_df,
        normalization_stats,
        exclude_columns=getattr(cfg.data, "exclude_columns", []),
    )



    feature_names = list(train_df.columns)

    # 7. Dataset + loaders
    train_dataset = TabularFoundationDataset(
        df=train_df,
        feature_names=feature_names,
        cohort_name=cfg.cohort.name,
    )
    val_dataset = TabularFoundationDataset(
        df=val_df,
        feature_names=feature_names,
        cohort_name=cfg.cohort.name,
    )

    '''collator = MaskedModelingCollator(
        feature_names=feature_names,
        masking_ratio=cfg.objective.masking_ratio,
        min_masked_features=cfg.objective.min_masked_features,
        seed=cfg.experiment.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.run.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.run.num_workers,
    )'''
    train_collator = MaskedModelingCollator(
        feature_names=feature_names,
        masking_ratio=cfg.objective.masking_ratio,
        min_masked_features=cfg.objective.min_masked_features,
        seed=cfg.experiment.seed,
    )
    
    val_collator = MaskedModelingCollator(
        feature_names=feature_names,
        masking_ratio=cfg.objective.masking_ratio,
        min_masked_features=cfg.objective.min_masked_features,
        seed=cfg.experiment.seed + 1,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=cfg.run.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=cfg.run.num_workers,
    )

    # 8. Model
    model = TabularFoundationModel(
        num_features=len(feature_names),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        pooling_type=cfg.model.pooling_type,
        regression_head_hidden_dim=cfg.model.regression_head_hidden_dim,
        projection_dim=cfg.model.projection_dim,
        projection_hidden_dim=cfg.model.projection_hidden_dim,
    ).to(paths.device)

    # 9. Objectives + optimizer + scheduler
    objective_manager = ObjectiveManager(
        reconstruction_loss_weight=cfg.objective.reconstruction_loss_weight,
    )

    optimizer = build_optimizer(
        model=model,
        optimizer_name=cfg.train.optimizer_name,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=cfg.train.scheduler_name,
        num_epochs=cfg.train.num_epochs,
        step_size=cfg.train.scheduler_step_size,
        gamma=cfg.train.scheduler_gamma,
        t_max=cfg.train.scheduler_t_max,
        eta_min=cfg.train.scheduler_eta_min,
    )

    # 10. Trainer
    trainer = Trainer(
        model=model,
        objective_manager=objective_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        device=paths.device,
        grad_clip_norm=cfg.train.grad_clip_norm,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.train.num_epochs,
    )

    # 11. Outputs
    if cfg.output.write_metrics_csv:
        write_metrics_table(history, paths.metrics_file)

    if cfg.output.save_plots:
        plot_metric_history(history, paths.plots_dir / "loss_curve.png")

    if cfg.output.write_summary_json:
        export_run_summary(
            cfg=cfg,
            history=history,
            output_path=paths.summary_file,
        )

    logger.info("Run completed. Outputs saved in %s", paths.run_dir)


if __name__ == "__main__":
    run()
