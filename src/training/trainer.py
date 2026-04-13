from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .engine import train_step, validation_step


@dataclass
class EpochHistory:
    train_loss: List[float] = field(default_factory=list)
    train_reconstruction_loss: List[float] = field(default_factory=list)
    train_reconstruction_mae: List[float] = field(default_factory=list)

    val_loss: List[float] = field(default_factory=list)
    val_reconstruction_loss: List[float] = field(default_factory=list)
    val_reconstruction_mae: List[float] = field(default_factory=list)


def _average_metrics(step_outputs: List[Dict[str, float]]) -> Dict[str, float]:
    if not step_outputs:
        return {}

    keys = step_outputs[0].keys()
    return {
        key: float(sum(step[key] for step in step_outputs) / len(step_outputs))
        for key in keys
    }


class Trainer:
    """
    High-level training loop manager for v1 masked-feature-modeling pretraining.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        objective_manager,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip_norm: Optional[float] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.objective_manager = objective_manager
        self.optimizer = optimizer
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler

        self.model.to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        step_outputs: List[Dict[str, float]] = []

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            metrics = train_step(
                model=self.model,
                batch=batch,
                objective_manager=self.objective_manager,
                optimizer=self.optimizer,
                device=self.device,
                grad_clip_norm=self.grad_clip_norm,
            )
            step_outputs.append(metrics)

            progress_bar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                recon=f"{metrics['reconstruction_loss']:.4f}",
                mae=f"{metrics['reconstruction_mae']:.4f}",
            )

        epoch_metrics = _average_metrics(step_outputs)

        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        step_outputs: List[Dict[str, float]] = []

        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for batch in progress_bar:
            metrics = validation_step(
                model=self.model,
                batch=batch,
                objective_manager=self.objective_manager,
                device=self.device,
            )
            step_outputs.append(metrics)

            progress_bar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                recon=f"{metrics['reconstruction_loss']:.4f}",
                mae=f"{metrics['reconstruction_mae']:.4f}",
            )

        return _average_metrics(step_outputs)

    def fit(
        self,
        *,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int,
    ) -> EpochHistory:
        history = EpochHistory()

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            train_metrics = self.train_epoch(train_loader)
            print(
                f"Train | loss={train_metrics.get('loss', float('nan')):.4f} | "
                f"recon={train_metrics.get('reconstruction_loss', float('nan')):.4f} | "
                f"mae={train_metrics.get('reconstruction_mae', float('nan')):.4f}"
            )

            history.train_loss.append(train_metrics.get("loss", float("nan")))
            history.train_reconstruction_loss.append(
                train_metrics.get("reconstruction_loss", float("nan"))
            )
            history.train_reconstruction_mae.append(
                train_metrics.get("reconstruction_mae", float("nan"))
            )

            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                print(
                    f"Val   | loss={val_metrics.get('loss', float('nan')):.4f} | "
                    f"recon={val_metrics.get('reconstruction_loss', float('nan')):.4f} | "
                    f"mae={val_metrics.get('reconstruction_mae', float('nan')):.4f}"
                )

                history.val_loss.append(val_metrics.get("loss", float("nan")))
                history.val_reconstruction_loss.append(
                    val_metrics.get("reconstruction_loss", float("nan"))
                )
                history.val_reconstruction_mae.append(
                    val_metrics.get("reconstruction_mae", float("nan"))
                )

        return history
