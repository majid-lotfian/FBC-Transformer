import pandas as pd

from src.data.dataset import TabularFeatureDataset


def test_dataset_item_shapes():
    df = pd.DataFrame({"hemoglobin": [1.0, 2.0], "wbc": [3.0, 4.0]})
    ds = TabularFeatureDataset(df, feature_names=["hemoglobin", "wbc"], cohort_id=0)
    item = ds[0]
    assert item.values.shape[0] == 2
    assert item.observed_mask.shape[0] == 2
