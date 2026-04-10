from __future__ import annotations

from torch import nn


class ContinuousReconstructionHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, token_embeddings):
        return self.proj(token_embeddings).squeeze(-1)
