from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from src.training.engine import move_batch_to_device
from src.training.checkpoint import save_checkpoint
from src.training.metrics import average_metric_dict


class Trainer:
    def __init__(
        self,
        model,
        objective_manager,
        optimizer,
        scheduler,
        device: torch.device,
        checkpoint_dir: Path,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.objective_manager = objective_manager
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_clip_norm = gradient_clip_norm

    def fit(self, train_loader, val_loader, epochs: int, checkpoint_every: int = 1) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            epoch_metrics = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
            history.append(epoch_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
        return history

    def _run_epoch(self, data_loader, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        collected: List[Dict[str, float]] = []

        for batch in data_loader:
            batch = move_batch_to_device(batch, self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(batch)
                metrics = self.objective_manager.compute(outputs, batch)
                loss = metrics["loss"]

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()

            collected.append({
                "loss": float(loss.detach().cpu().item())
            })

        return average_metric_dict(collected)
