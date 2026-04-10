from __future__ import annotations

import torch


def mean_pool(token_embeddings: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    weights = observed_mask.float().unsqueeze(-1)
    summed = (token_embeddings * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return summed / denom


def pool(token_embeddings: torch.Tensor, observed_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return mean_pool(token_embeddings, observed_mask)
    raise ValueError(f"Unsupported pooling mode: {mode}")
