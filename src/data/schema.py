from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class MetadataBundle:
    canonical_features: pd.DataFrame
    cohort_features: pd.DataFrame
    mapping: pd.DataFrame

    @property
    def model_input_feature_names(self) -> List[str]:
        if "model_input" in self.canonical_features.columns:
            model_inputs = self.canonical_features[self.canonical_features["model_input"].astype(str).str.lower() == "yes"]
            return model_inputs["canonical_name"].tolist()
        return self.canonical_features["canonical_name"].tolist()


def load_metadata_bundle(
    canonical_path: str | Path,
    cohort_features_path: str | Path,
    cohort_mapping_path: str | Path,
) -> MetadataBundle:
    return MetadataBundle(
        canonical_features=pd.read_csv(canonical_path),
        cohort_features=pd.read_csv(cohort_features_path),
        mapping=pd.read_csv(cohort_mapping_path),
    )
