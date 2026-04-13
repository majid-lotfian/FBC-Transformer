from __future__ import annotations

import torch


def build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
