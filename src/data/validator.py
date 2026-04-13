from __future__ import annotations

from collections import Counter

from .schema import MasterSchema


def validate_master_schema(schema: MasterSchema) -> None:
    canonical_names = schema.get_canonical_names()

    if not canonical_names:
        raise ValueError("Schema contains no canonical features.")

    canonical_counts = Counter(canonical_names)
    duplicate_canonical = [name for name, count in canonical_counts.items() if count > 1]
    if duplicate_canonical:
        raise ValueError(f"Duplicate canonical feature names found: {duplicate_canonical}")

    for cohort_name in schema.get_cohort_names():
        mappings = schema.cohort_mappings.get(cohort_name, {})

        if len(mappings) != len(canonical_names):
            raise ValueError(
                f"Cohort '{cohort_name}' does not have mappings for all canonical features."
            )

        present_feature_names = [
            entry.cohort_feature_name
            for entry in mappings.values()
            if entry.cohort_feature_name is not None
        ]

        cohort_counts = Counter(present_feature_names)
        duplicate_cohort_names = [name for name, count in cohort_counts.items() if count > 1]
        if duplicate_cohort_names:
            raise ValueError(
                f"Cohort '{cohort_name}' has duplicate mapped feature names: {duplicate_cohort_names}"
            )

        for canonical_name, entry in mappings.items():
            if canonical_name != entry.canonical_name:
                raise ValueError(
                    f"Mapping mismatch in cohort '{cohort_name}': "
                    f"dict key '{canonical_name}' != entry canonical_name '{entry.canonical_name}'"
                )

            if entry.cohort_name != cohort_name:
                raise ValueError(
                    f"Mapping mismatch for canonical feature '{canonical_name}': "
                    f"entry cohort_name '{entry.cohort_name}' != expected '{cohort_name}'"
                )
