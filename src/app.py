from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_experiment_config
from src.paths import PathManager
from src.outputs.logger import setup_logging, get_logger
from src.utils.seed import set_global_seed
from src.data.schema import load_metadata_bundle
from src.data.validator import validate_metadata_bundle
from src.data.loader import load_raw_dataframe
from src.data.mapper import map_to_canonical
from src.data.preprocessing import preprocess_dataframe
from src.data.normalization import Normalizer
from src.data.splits import make_train_val_dataframes
from src.data.dataset import TabularFeatureDataset
from src.data.collator import MaskedModelingCollator
from src.models.model import BloodFoundationModel
from src.objectives.objective_manager import ObjectiveManager
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.outputs.exporter import export_run_summary
from src.outputs.plots import plot_metric_history
from src.outputs.tables import write_metrics_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood foundation model v1")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing YAML configs")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    cfg = load_experiment_config(Path(args.config_dir))
    paths = PathManager.from_config(cfg)
    setup_logging(paths.log_file, to_console=cfg.output.log_to_console)
    logger = get_logger(__name__)

    set_global_seed(cfg.experiment.seed)
    logger.info("Loaded configuration for experiment '%s'", cfg.experiment.name)

    metadata = load_metadata_bundle(
        canonical_path=cfg.data.canonical_features_path,
        cohort_features_path=cfg.data.cohort_features_path,
        cohort_mapping_path=cfg.data.cohort_mapping_path,
    )
    validate_metadata_bundle(metadata)

    raw_df = load_raw_dataframe(cfg.data.raw_data_path)
    canonical_df = map_to_canonical(raw_df, metadata.mapping)
    canonical_df = preprocess_dataframe(canonical_df, metadata.canonical_features)

    train_df, val_df = make_train_val_dataframes(
        canonical_df,
        train_fraction=cfg.data.train_fraction,
        seed=cfg.experiment.seed,
    )

    normalizer = Normalizer(enabled=cfg.data.normalization.enabled, mode=cfg.data.normalization.mode)
    normalizer.fit(train_df)
    train_df = normalizer.transform(train_df)
    val_df = normalizer.transform(val_df)

    feature_names = metadata.model_input_feature_names
    train_dataset = TabularFeatureDataset(train_df, feature_names, cohort_id=cfg.cohort.id)
    val_dataset = TabularFeatureDataset(val_df, feature_names, cohort_id=cfg.cohort.id)
    collator = MaskedModelingCollator(feature_names=feature_names, mask_ratio=cfg.objective.mask_ratio)

    train_loader = train_dataset.make_dataloader(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = val_dataset.make_dataloader(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=collator,
    )

    model = BloodFoundationModel(
        num_features=len(feature_names),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        use_cohort_embedding=cfg.model.use_cohort_embedding,
    ).to(paths.device)

    objective_manager = ObjectiveManager()
    optimizer = build_optimizer(model=model, learning_rate=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=cfg.train.scheduler,
        epochs=cfg.train.epochs,
        steps_per_epoch=max(1, len(train_loader)),
    )

    trainer = Trainer(
        model=model,
        objective_manager=objective_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        device=paths.device,
        checkpoint_dir=paths.checkpoints_dir,
        gradient_clip_norm=cfg.train.gradient_clip_norm,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.train.epochs,
        checkpoint_every=cfg.output.checkpoint_every,
    )

    if cfg.output.write_metrics_csv:
        write_metrics_table(history, paths.tables_dir / "metrics.csv")
    if cfg.run.save_plots:
        plot_metric_history(history, paths.plots_dir / "loss_curve.png")
    if cfg.output.write_summary_json:
        export_run_summary(cfg=cfg, history=history, output_path=paths.run_dir / "summary.json")

    logger.info("Run completed. Outputs saved in %s", paths.run_dir)
