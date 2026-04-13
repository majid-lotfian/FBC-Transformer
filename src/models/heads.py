from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Token-level regression head for masked feature prediction.
    Input:  [B, F, d_model]
    Output: [B, F]
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Linear(d_model, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)          # [B, F, 1]
        return out.squeeze(-1)     # [B, F]


class ProjectionHead(nn.Module):
    """
    Sample-level projection head for future contrastive learning.
    Not used yet in masked-only v1, but useful to keep ready.
    Input:  [B, d_model]
    Output: [B, projection_dim]
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
