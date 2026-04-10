from __future__ import annotations

from typing import Dict, List


def summarize_history(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    last = history[-1]
    return {
        "final_train_loss": float(last["train_loss"]),
        "final_val_loss": float(last["val_loss"]),
    }
