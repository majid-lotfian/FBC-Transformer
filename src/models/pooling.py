from __future__ import annotations

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    Mean pooling over valid tokens only.
    """

    def forward(
        self,
        x: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, F, d_model]
        input_mask: [B, F] bool, True = valid token
        returns: [B, d_model]
        """
        if input_mask is None:
            return x.mean(dim=1)

        mask = input_mask.unsqueeze(-1).float()          # [B, F, 1]
        masked_x = x * mask                              # [B, F, d_model]
        counts = mask.sum(dim=1).clamp(min=1.0)         # [B, 1]
        pooled = masked_x.sum(dim=1) / counts           # [B, d_model]
        return pooled


class CLSPooling(nn.Module):
    """
    Pool using the first token.
    Only use this if you intentionally place a CLS token at index 0.
    """

    def forward(
        self,
        x: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x[:, 0, :]  # [B, d_model]


def build_pooling(pooling_type: str) -> nn.Module:
    pooling_type = pooling_type.lower()

    if pooling_type == "mean":
        return MeanPooling()
    if pooling_type == "cls":
        return CLSPooling()

    raise ValueError(f"Unsupported pooling_type: {pooling_type}")
