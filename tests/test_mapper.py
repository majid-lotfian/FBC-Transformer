import pandas as pd

from src.data.mapper import map_to_canonical


def test_mapping_renames_columns():
    raw = pd.DataFrame({"Hb": [1.0], "WBC": [2.0]})
    mapping = pd.DataFrame({
        "local_name": ["Hb", "WBC"],
        "canonical_name": ["hemoglobin", "wbc"],
        "mapping_status": ["exact", "exact"],
    })
    out = map_to_canonical(raw, mapping)
    assert "hemoglobin" in out.columns
    assert "wbc" in out.columns
