from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _ns_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _ns_to_dict(v) for k, v in obj.items()}
    return obj


def _history_to_dict(history) -> dict:
    return {
        "train_loss": history.train_loss,
        "train_reconstruction_loss": history.train_reconstruction_loss,
        "train_reconstruction_mae": history.train_reconstruction_mae,
        "val_loss": history.val_loss,
        "val_reconstruction_loss": history.val_reconstruction_loss,
        "val_reconstruction_mae": history.val_reconstruction_mae,
    }


def export_run_summary(cfg, history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": _ns_to_dict(cfg),
        "history": _history_to_dict(history),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
