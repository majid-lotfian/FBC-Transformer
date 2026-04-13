from __future__ import annotations

import torch
import torch.nn as nn


class TabularTransformerEncoder(nn.Module):
    """
    Encoder-only transformer for tabular feature tokens.

    Input:
    - x: [B, F, d_model]
    - input_mask: [B, F] bool
        True  = feature is available to attend with
        False = feature is hidden / unavailable

    Output:
    - encoded: [B, F, d_model]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            [B, F, d_model]
        input_mask:
            [B, F] bool
            True  = valid/visible token
            False = masked or unavailable token

        Returns
        -------
        torch.Tensor
            [B, F, d_model]
        """
        src_key_padding_mask = None
        if input_mask is not None:
            # PyTorch expects True where tokens should be ignored
            src_key_padding_mask = ~input_mask.bool()

        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoded = self.final_norm(encoded)
        return encoded
