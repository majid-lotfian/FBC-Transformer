from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.loader import load_master_schema
from src.data.mapper import build_canonical_dataframe
from src.data.preprocessing import basic_preprocess_dataframe
from src.data.normalization import fit_and_apply_standardization
from src.data.dataset import TabularFoundationDataset
from src.data.collator import MaskedModelingCollator

from src.objectives.objective_manager import ObjectiveManager

from src.models.model import TabularFoundationModel

from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--schema_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cohort_name", type=str, required=True)

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 1. Load schema
    # =========================
    schema = load_master_schema(args.schema_path)

    # =========================
    # 2. Load raw data
    # =========================
    df = pd.read_csv(args.data_path)

    # =========================
    # 3. Map to canonical
    # =========================
    df = build_canonical_dataframe(
        df,
        schema=schema,
        cohort_name=args.cohort_name,
    )

    # =========================
    # 4. Preprocess
    # =========================
    df = basic_preprocess_dataframe(df)

    # =========================
    # 5. Normalize
    # =========================
    df, stats = fit_and_apply_standardization(df)

    # =========================
    # 6. Dataset
    # =========================
    feature_names = list(df.columns)

    dataset = TabularFoundationDataset(
        df=df,
        feature_names=feature_names,
        cohort_name=args.cohort_name,
    )

    collator = MaskedModelingCollator(
        feature_names=feature_names,
        masking_ratio=0.15,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # =========================
    # 7. Model
    # =========================
    model = TabularFoundationModel(
        num_features=len(feature_names),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )

    # =========================
    # 8. Training setup
    # =========================
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    objective_manager = ObjectiveManager()

    trainer = Trainer(
        model=model,
        objective_manager=objective_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # =========================
    # 9. Train
    # =========================
    trainer.fit(
        train_loader=dataloader,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
