from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(checkpoint_dir: Path, epoch: int, model, optimizer, scheduler) -> Path:
    path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": getattr(scheduler, "__dict__", None),
    }
    torch.save(payload, path)
    return path
