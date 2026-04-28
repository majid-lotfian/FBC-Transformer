from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss (InfoNCE / SimCLR).

    Given two views z1 and z2 of the same B samples, treats (z1[i], z2[i])
    as a positive pair and all other 2(B-1) combinations as negatives.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: [B, D] — raw (unnormalized) projection vectors.
        Returns scalar loss.
        """
        B = z1.shape[0]
        if B < 2:
            return z1.sum() * 0.0

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1, z2], dim=0)             # [2B, D]
        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pair for row i is row i+B (and vice versa)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        return F.cross_entropy(sim, labels)
