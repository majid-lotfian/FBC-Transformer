from __future__ import annotations

import torch
from torch import nn


class ValueEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.unsqueeze(-1))


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        use_cohort_embedding: bool = True,
        num_states: int = 4,
        num_cohorts: int = 16,
    ) -> None:
        super().__init__()
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.state_embedding = nn.Embedding(num_states, d_model)
        self.value_embedding = ValueEmbedding(d_model=d_model)
        self.use_cohort_embedding = use_cohort_embedding
        self.cohort_embedding = nn.Embedding(num_cohorts, d_model) if use_cohort_embedding else None

    def forward(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        state_ids: torch.Tensor,
        cohort_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.feature_embedding(feature_ids)
        x = x + self.state_embedding(state_ids)
        x = x + self.value_embedding(values)
        if self.use_cohort_embedding and self.cohort_embedding is not None:
            x = x + self.cohort_embedding(cohort_ids).unsqueeze(1)
        return x
