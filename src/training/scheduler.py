from __future__ import annotations

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str | None = None,
    num_epochs: int | None = None,
    step_size: int = 10,
    gamma: float = 0.1,
    t_max: int | None = None,
    eta_min: float = 0.0,
):
    """
    Build an optional learning-rate scheduler.

    Supported:
    - None
    - "step"
    - "cosine"
    """
    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    if scheduler_name == "cosine":
        if t_max is None:
            if num_epochs is None:
                raise ValueError("For cosine scheduler, provide either t_max or num_epochs.")
            t_max = num_epochs

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")
