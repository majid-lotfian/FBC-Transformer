from src.data.schema import load_metadata_bundle
from src.data.validator import validate_metadata_bundle


def test_metadata_bundle_loads():
    metadata = load_metadata_bundle(
        "metadata/canonical_features.csv",
        "metadata/cohorts/cohort_a/cohort_a_features.csv",
        "metadata/cohorts/cohort_a/cohort_a_mapping.csv",
    )
    validate_metadata_bundle(metadata)
    assert len(metadata.model_input_feature_names) > 0
