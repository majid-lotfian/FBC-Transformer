from __future__ import annotations

from typing import Dict


def summarize_history(history) -> Dict[str, float]:
    summary = {}

    if history.train_loss:
        summary["final_train_loss"] = float(history.train_loss[-1])

    if history.train_reconstruction_loss:
        summary["final_train_reconstruction_loss"] = float(history.train_reconstruction_loss[-1])

    if history.train_reconstruction_mae:
        summary["final_train_reconstruction_mae"] = float(history.train_reconstruction_mae[-1])

    if history.val_loss:
        summary["final_val_loss"] = float(history.val_loss[-1])

    if history.val_reconstruction_loss:
        summary["final_val_reconstruction_loss"] = float(history.val_reconstruction_loss[-1])

    if history.val_reconstruction_mae:
        summary["final_val_reconstruction_mae"] = float(history.val_reconstruction_mae[-1])

    return summary
