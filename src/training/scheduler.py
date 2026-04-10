from __future__ import annotations

from math import pi, cos


class SimpleCosineScheduler:
    def __init__(self, optimizer, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = max(1, total_epochs)
        self.epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.epoch += 1
        factor = 0.5 * (1 + cos(pi * min(self.epoch, self.total_epochs) / self.total_epochs))
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor


def build_scheduler(optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int):
    del steps_per_epoch
    if scheduler_name == "cosine":
        return SimpleCosineScheduler(optimizer=optimizer, total_epochs=epochs)
    if scheduler_name in {"none", "", None}:
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
