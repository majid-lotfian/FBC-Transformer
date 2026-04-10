from __future__ import annotations

import torch


def build_optimizer(model, learning_rate: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
