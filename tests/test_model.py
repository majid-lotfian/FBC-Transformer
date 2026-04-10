import torch

from src.models.model import BloodFoundationModel


def test_model_forward_runs():
    model = BloodFoundationModel(
        num_features=5,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        pooling="mean",
    )
    batch = {
        "feature_ids": torch.arange(5).unsqueeze(0).repeat(2, 1),
        "values": torch.randn(2, 5),
        "state_ids": torch.zeros(2, 5, dtype=torch.long),
        "cohort_ids": torch.zeros(2, dtype=torch.long),
        "observed_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    out = model(batch)
    assert out["reconstruction"].shape == (2, 5)
    assert out["pooled_embedding"].shape[0] == 2
