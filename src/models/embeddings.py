from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """
    Embeds feature IDs (which feature is this)
    """

    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_features, d_model)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        # feature_ids: [B, F]
        return self.embedding(feature_ids)  # [B, F, d_model]


class ValueEmbedding(nn.Module):
    """
    Embeds scalar feature values into vector space
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, F]
        values = values.unsqueeze(-1)  # [B, F, 1]
        return self.linear(values)    # [B, F, d_model]


class MaskEmbedding(nn.Module):
    """
    Embeds mask state (observed vs missing)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, d_model)

    def forward(self, observed_mask: torch.Tensor) -> torch.Tensor:
        # observed_mask: [B, F] bool
        mask_ids = observed_mask.long()  # 0 or 1
        return self.embedding(mask_ids)  # [B, F, d_model]


class TabularEmbedding(nn.Module):
    """
    Combines:
    - feature identity
    - feature value
    - observed/missing state
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
    ) -> None:
        super().__init__()

        self.feature_embedding = FeatureEmbedding(num_features, d_model)
        self.value_embedding = ValueEmbedding(d_model)
        self.mask_embedding = MaskEmbedding(d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
        - feature_ids: [B, F]
        - values: [B, F]
        - observed_mask: [B, F]

        Output:
        - embeddings: [B, F, d_model]
        """

        f_emb = self.feature_embedding(feature_ids)
        v_emb = self.value_embedding(values)
        m_emb = self.mask_embedding(observed_mask)

        x = f_emb + v_emb + m_emb
        x = self.layer_norm(x)

        return x
