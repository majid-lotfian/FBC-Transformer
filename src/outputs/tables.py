from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metrics_table(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = len(history.train_loss)

    data = {
        "epoch": list(range(1, num_epochs + 1)),
        "train_loss": history.train_loss,
        "train_reconstruction_loss": history.train_reconstruction_loss,
        "train_reconstruction_mae": history.train_reconstruction_mae,
        "val_loss": history.val_loss,
        "val_reconstruction_loss": history.val_reconstruction_loss,
        "val_reconstruction_mae": history.val_reconstruction_mae,
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
