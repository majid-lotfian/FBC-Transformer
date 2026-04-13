from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .embeddings import TabularEmbedding
from .heads import ProjectionHead, RegressionHead
from .pooling import build_pooling
from .transformer import TabularTransformerEncoder


class TabularFoundationModel(nn.Module):
    """
    Full v1 model for masked feature modeling on tabular blood-test data.

    Pipeline:
    - feature/value/mask embeddings
    - transformer encoder
    - token-level regression head for masked reconstruction
    - sample-level pooling
    - optional projection head for future contrastive learning
    """

    def __init__(
        self,
        *,
        num_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        pooling_type: str = "mean",
        regression_head_hidden_dim: int | None = None,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.embedding = TabularEmbedding(
            num_features=num_features,
            d_model=d_model,
        )

        self.encoder = TabularTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.pooling = build_pooling(pooling_type)

        self.regression_head = RegressionHead(
            d_model=d_model,
            hidden_dim=regression_head_hidden_dim,
        )

        self.projection_head = None
        if projection_dim is not None:
            self.projection_head = ProjectionHead(
                d_model=d_model,
                projection_dim=projection_dim,
                hidden_dim=projection_hidden_dim,
            )

    def encode(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns token-level contextual representations.

        Inputs:
        - feature_ids:   [B, F]
        - values:        [B, F]
        - observed_mask: [B, F]
        - input_mask:    [B, F]
        """
        x = self.embedding(
            feature_ids=feature_ids,
            values=values,
            observed_mask=observed_mask,
        )  # [B, F, d_model]

        encoded = self.encoder(
            x=x,
            input_mask=input_mask,
        )  # [B, F, d_model]

        return encoded

    def pool(
        self,
        encoded: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns sample-level pooled embedding.
        """
        return self.pooling(
            x=encoded,
            input_mask=input_mask,
        )  # [B, d_model]

    def forward(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Returns a dictionary with:
        - token_embeddings: [B, F, d_model]
        - pooled_embedding: [B, d_model]
        - reconstruction:   [B, F]
        - projection:       [B, P] or None
        """
        token_embeddings = self.encode(
            feature_ids=feature_ids,
            values=values,
            observed_mask=observed_mask,
            input_mask=input_mask,
        )

        pooled_embedding = self.pool(
            encoded=token_embeddings,
            input_mask=input_mask,
        )

        reconstruction = self.regression_head(token_embeddings)  # [B, F]

        projection = None
        if self.projection_head is not None:
            projection = self.projection_head(pooled_embedding)

        return {
            "token_embeddings": token_embeddings,
            "pooled_embedding": pooled_embedding,
            "reconstruction": reconstruction,
            "projection": projection,
        }
