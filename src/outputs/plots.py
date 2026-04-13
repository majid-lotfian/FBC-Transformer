from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_history(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.train_loss) + 1))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if history.train_loss:
        ax.plot(epochs, history.train_loss, label="train_loss")

    if history.val_loss:
        ax.plot(epochs, history.val_loss, label="val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
