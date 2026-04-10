# Blood Foundation Model (v1 Skeleton)

A modular research codebase for pretraining a transformer-based foundation model on large-scale blood-test tabular data.

## v1 scope

- One cohort at a time
- Canonical schema + cohort mapping
- Feature-token transformer encoder
- Masked feature modeling objective
- Modular training / evaluation / outputs
- Reproducible configs and run artifacts

## Project layout

- `configs/`: YAML experiment settings
- `metadata/`: canonical schema and cohort mapping files
- `src/`: package source code
- `scripts/`: convenience run scripts
- `tests/`: lightweight tests
- `artifacts/`: outputs produced by runs

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e .
python main.py --config-dir configs
```

## Notes

This repository is intentionally initialized as a **strong skeleton**:
- interfaces are defined
- module boundaries are in place
- implementations are minimal but runnable
- advanced logic can be filled in incrementally

## Next implementation priorities

1. finalize metadata files
2. replace dummy sample data with real cohort data
3. strengthen mapper / preprocessing / normalization logic
4. tune model and masking policy
5. add denoising and contrastive objectives later
