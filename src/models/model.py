from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.models.embeddings import TokenEmbedding
from src.models.transformer import TransformerEncoderBackbone
from src.models.pooling import pool
from src.models.heads import ContinuousReconstructionHead


class BloodFoundationModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str,
        use_cohort_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.pooling_mode = pooling
        self.token_embedding = TokenEmbedding(
            num_features=num_features,
            d_model=d_model,
            use_cohort_embedding=use_cohort_embedding,
        )
        self.encoder = TransformerEncoderBackbone(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.reconstruction_head = ContinuousReconstructionHead(d_model=d_model)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature_ids = batch["feature_ids"]
        values = batch["values"]
        state_ids = batch["state_ids"]
        cohort_ids = batch["cohort_ids"]
        observed_mask = batch["observed_mask"]

        x = self.token_embedding(
            feature_ids=feature_ids,
            values=values,
            state_ids=state_ids,
            cohort_ids=cohort_ids,
        )
        key_padding_mask = ~observed_mask
        token_embeddings = self.encoder(x, key_padding_mask=key_padding_mask)
        pooled_embedding = pool(token_embeddings, observed_mask=observed_mask, mode=self.pooling_mode)
        reconstruction = self.reconstruction_head(token_embeddings)

        return {
            "token_embeddings": token_embeddings,
            "pooled_embedding": pooled_embedding,
            "reconstruction": reconstruction,
        }
