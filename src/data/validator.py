from __future__ import annotations

from src.data.schema import MetadataBundle


REQUIRED_CANONICAL_COLUMNS = {"canonical_name", "data_type"}
REQUIRED_MAPPING_COLUMNS = {"local_name", "canonical_name", "mapping_status"}
REQUIRED_COHORT_FEATURE_COLUMNS = {"local_name", "data_type"}


def validate_metadata_bundle(metadata: MetadataBundle) -> None:
    missing = REQUIRED_CANONICAL_COLUMNS - set(metadata.canonical_features.columns)
    if missing:
        raise ValueError(f"Missing canonical feature columns: {sorted(missing)}")

    missing = REQUIRED_MAPPING_COLUMNS - set(metadata.mapping.columns)
    if missing:
        raise ValueError(f"Missing mapping columns: {sorted(missing)}")

    missing = REQUIRED_COHORT_FEATURE_COLUMNS - set(metadata.cohort_features.columns)
    if missing:
        raise ValueError(f"Missing cohort feature columns: {sorted(missing)}")

    canonical_names = set(metadata.canonical_features["canonical_name"].astype(str))
    mapped_names = set(metadata.mapping["canonical_name"].dropna().astype(str))
    unknown = mapped_names - canonical_names
    if unknown:
        raise ValueError(f"Mapping references unknown canonical names: {sorted(unknown)}")
