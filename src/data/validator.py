from __future__ import annotations

from collections import Counter

from .schema import MasterSchema


def validate_master_schema(schema: MasterSchema) -> None:
    """
    Validate the loaded master schema.

    Checks:
    - schema is not empty
    - canonical feature names are unique
    - each cohort has a mapping entry for every canonical feature
    - mapping keys and mapping objects are internally consistent
    - each present cohort feature name is unique within that cohort
    """
    canonical_names = schema.get_canonical_names()

    if not canonical_names:
        raise ValueError("Schema contains no canonical features.")

    canonical_counts = Counter(canonical_names)
    duplicate_canonical_names = [
        name for name, count in canonical_counts.items() if count > 1
    ]
    if duplicate_canonical_names:
        raise ValueError(
            f"Duplicate canonical feature names found: {duplicate_canonical_names}"
        )

    for canonical_name in canonical_names:
        feature = schema.get_feature(canonical_name)
        if feature is None:
            raise ValueError(
                f"Canonical feature '{canonical_name}' is missing from feature index."
            )

    cohort_names = schema.get_cohort_names()
    if not cohort_names:
        raise ValueError("Schema contains no cohort mappings.")

    canonical_name_set = set(canonical_names)

    for cohort_name in cohort_names:
        mappings = schema.cohort_mappings.get(cohort_name)
        if mappings is None:
            raise ValueError(f"Cohort '{cohort_name}' is missing its mapping dictionary.")

        mapping_keys = set(mappings.keys())

        missing_canonical_names = canonical_name_set - mapping_keys
        if missing_canonical_names:
            raise ValueError(
                f"Cohort '{cohort_name}' is missing mapping entries for canonical features: "
                f"{sorted(missing_canonical_names)}"
            )

        extra_canonical_names = mapping_keys - canonical_name_set
        if extra_canonical_names:
            raise ValueError(
                f"Cohort '{cohort_name}' contains mappings for unknown canonical features: "
                f"{sorted(extra_canonical_names)}"
            )

        present_cohort_feature_names: list[str] = []

        for canonical_name, entry in mappings.items():
            if canonical_name != entry.canonical_name:
                raise ValueError(
                    f"Cohort '{cohort_name}' has inconsistent mapping entry: "
                    f"dict key '{canonical_name}' != entry.canonical_name '{entry.canonical_name}'."
                )

            if entry.cohort_name != cohort_name:
                raise ValueError(
                    f"Canonical feature '{canonical_name}' has inconsistent cohort name: "
                    f"entry.cohort_name '{entry.cohort_name}' != expected '{cohort_name}'."
                )

            if entry.cohort_feature_name is not None:
                present_cohort_feature_names.append(entry.cohort_feature_name)

        present_counts = Counter(present_cohort_feature_names)
        duplicate_present_names = [
            name for name, count in present_counts.items() if count > 1
        ]
        if duplicate_present_names:
            raise ValueError(
                f"Cohort '{cohort_name}' has duplicate present feature names mapped to multiple "
                f"canonical features: {sorted(duplicate_present_names)}"
            )
