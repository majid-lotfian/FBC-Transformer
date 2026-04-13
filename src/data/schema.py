from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CanonicalFeature:
    canonical_name: str
    subgroup: Optional[str] = None
    unit: Optional[str] = None
    channel_profile: Optional[str] = None


@dataclass(frozen=True)
class CohortMappingEntry:
    cohort_name: str
    canonical_name: str
    cohort_feature_name: Optional[str] = None  # None = absent in this cohort


@dataclass
class MasterSchema:
    canonical_features: List[CanonicalFeature] = field(default_factory=list)
    cohort_mappings: Dict[str, Dict[str, CohortMappingEntry]] = field(default_factory=dict)
    _feature_index: Dict[str, CanonicalFeature] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._feature_index = {feature.canonical_name: feature for feature in self.canonical_features}

    def get_canonical_names(self) -> List[str]:
        return list(self._feature_index.keys())

    def get_feature(self, canonical_name: str) -> Optional[CanonicalFeature]:
        return self._feature_index.get(canonical_name)

    def has_feature(self, canonical_name: str) -> bool:
        return canonical_name in self._feature_index

    def get_mapping(
        self,
        cohort_name: str,
        canonical_name: str,
    ) -> Optional[CohortMappingEntry]:
        return self.cohort_mappings.get(cohort_name, {}).get(canonical_name)

    def get_present_features_for_cohort(self, cohort_name: str) -> Dict[str, str]:
        mappings = self.cohort_mappings.get(cohort_name, {})
        return {
            canonical_name: entry.cohort_feature_name
            for canonical_name, entry in mappings.items()
            if entry.cohort_feature_name is not None
        }

    def get_absent_features_for_cohort(self, cohort_name: str) -> List[str]:
        mappings = self.cohort_mappings.get(cohort_name, {})
        return [
            canonical_name
            for canonical_name, entry in mappings.items()
            if entry.cohort_feature_name is None
        ]

    def get_cohort_names(self) -> List[str]:
        return list(self.cohort_mappings.keys())
