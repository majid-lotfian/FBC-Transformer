from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def plot_metric_history(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
