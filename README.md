# FBC Transformer

FBC Transformer is a research codebase for pretraining a transformer-based foundation model on full blood count (FBC) tabular data. The main idea is to treat each blood-test feature as a token, learn contextual representations across all available features, and use the learned backbone for downstream clinical prediction tasks.

This version focuses on scalable pretraining. Instead of loading the full raw cohort into memory, the pipeline reads the raw CSV in chunks, maps each chunk into a shared canonical feature space, writes tensor shards to disk, and trains from those shards with lazy loading.

## What this project does

The project supports three main steps:

1. Build a canonical FBC feature table from cohort-specific raw data.
2. Pretrain a transformer model with masked feature reconstruction.
3. Fine-tune or evaluate the pretrained backbone on downstream tasks such as ferritin regression.

The code is designed for multi-cohort blood count data where the same clinical concept may have different column names in different cohorts. A schema file defines the canonical feature names and the mapping from each cohort to that shared feature space.

## Repository structure

```text
FBC-Transformer/
├── configs/
│   ├── base.yaml              # experiment name, seed, device, artifact root
│   ├── data.yaml              # raw data path, schema path, chunk and shard settings
│   ├── model.yaml             # transformer architecture
│   ├── train.yaml             # optimizer, scheduler, masking, epochs
│   ├── output.yaml            # output controls
│   └── cohort/
│       ├── amsterdam.yaml     # Amsterdam cohort config
│       └── cambridge.yaml     # Cambridge cohort config
├── main.py                    # pretraining entry point
├── metadata/                  # example metadata and canonical feature files
├── scripts/                   # helper shell scripts
├── src/
│   ├── data/                  # schema loading, mapping, preprocessing, sharding, datasets
│   ├── models/                # tabular embeddings, transformer encoder, pooling, heads
│   ├── objectives/            # masked reconstruction objective
│   ├── training/              # training loop, optimizer, scheduler, checkpointing
│   ├── outputs/               # logging, plots, metrics, summaries
│   └── FineTune/              # downstream ferritin fine-tuning script
└── tests/                     # unit tests for schema, mapper, dataset, model, training step
```

## Model overview

The model is implemented in `src/models/model.py` as `TabularFoundationModel`.

Each row of a blood-test table is represented as a sequence of feature tokens. For every feature, the model combines:

- a feature identity embedding, which tells the model which blood-test feature it is seeing;
- a value embedding, which projects the scalar feature value into the model dimension;
- a missingness embedding, which tells the model whether the value was observed or missing.

These token embeddings are passed through an encoder-only transformer. The model returns:

- `token_embeddings`: contextual embedding for each feature token;
- `pooled_embedding`: sample-level embedding produced by mean pooling;
- `reconstruction`: token-level predictions used for masked feature reconstruction;
- `projection`: optional projection output for future contrastive objectives.

The current pretraining objective is masked feature modeling. A subset of observed features is hidden from the model, and the model learns to reconstruct the original standardized values.

## Pretraining pipeline

The pretraining flow is controlled by `main.py`.

At a high level, the pipeline does the following:

1. Load configuration files from `configs/`.
2. Load the master schema from the Excel schema file.
3. Read the raw cohort CSV in chunks.
4. Map cohort-specific columns to canonical feature names.
5. Add missing canonical features as empty columns when needed.
6. Convert feature values to numeric values.
7. Split each chunk into train and validation rows.
8. Buffer rows until the target shard size is reached.
9. Save train and validation tensor shards as `.pt` files.
10. Fit normalization statistics from train shards only.
11. Create lazy shard-backed datasets.
12. Train the transformer with masked reconstruction.
13. Save checkpoints, metrics, plots, logs, and a run summary.

This sharded design is the main change in the current version. It avoids holding the full dataset in memory and makes the pipeline more suitable for large FBC datasets.

## Data and schema

The raw data is expected to be a CSV file. The schema is expected to be an Excel file with at least:

- `canonical_name`
- one or more cohort-specific mapping columns ending in `_name`, for example:
  - `amsterdam_name`
  - `cambridge_name`

Optional schema columns include:

- `subgroup`
- `unit`
- `channel_profile` or `channel/profile`

Column names are normalized internally by lowercasing, trimming spaces, and replacing whitespace with underscores. Punctuation is preserved.

For example, if a raw Amsterdam column and a raw Cambridge column refer to the same clinical feature, both can be mapped to the same canonical feature name through the schema file.

## Configuration

The main configuration files are:

```text
configs/base.yaml
configs/data.yaml
configs/model.yaml
configs/train.yaml
configs/output.yaml
configs/cohort/amsterdam.yaml
configs/cohort/cambridge.yaml
```

The most important data settings are in `configs/data.yaml`:

```yaml
data:
  raw_data_path: "path/to/raw_fbc.csv"
  schema_path: "path/to/master_schema.xlsx"
  chunk_size: 100000
  target_shard_rows: 50000
  subset_fraction: 1
  train_fraction: 0.8
  strict_mapping: false
  treat_dash_as_missing: false
  exclude_columns: []
```

`chunk_size` controls how many rows are read from the raw CSV at once. `target_shard_rows` controls how many rows are saved into each tensor shard. For large datasets, these two values are important for balancing memory use and training speed.

The model settings are in `configs/model.yaml`:

```yaml
model:
  d_model: 128
  nhead: 4
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.1
  pooling_type: mean
```

The training and masking settings are in `configs/train.yaml`:

```yaml
train:
  batch_size: 64
  num_epochs: 20
  optimizer_name: adamw
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
  scheduler_name: cosine
  grad_clip_norm: 1.0

objective:
  masking_ratio: 0.15
  min_masked_features: 1
  reconstruction_loss_weight: 1.0
```

## Installation

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -e .
```

The core dependencies are listed in `pyproject.toml`:

- PyYAML
- pandas
- numpy
- torch
- matplotlib

For the ferritin fine-tuning script, install the additional downstream packages:

```bash
pip install scipy scikit-learn xgboost lightgbm catboost pyarrow openpyxl
```

`openpyxl` is needed when reading Excel schema files with pandas.

## Running pretraining

First, edit `configs/data.yaml` so that `raw_data_path` and `schema_path` point to your local files.

Then run pretraining for a cohort:

```bash
python main.py --config-dir configs --cohort-config-name amsterdam.yaml
```

or:

```bash
python main.py --config-dir configs --cohort-config-name cambridge.yaml
```

The cohort name must match one of the cohort mapping columns in the schema. For example, `configs/cohort/amsterdam.yaml` sets:

```yaml
cohort:
  id: 0
  name: amsterdam
  description: Amsterdam cohort
```

The schema must therefore contain an Amsterdam mapping column such as `amsterdam_name`.

## Outputs

Each run creates a timestamped folder under the artifact root, for example:

```text
artifacts/blood_foundation_v2_YYYYMMDD_HHMMSS/
```

Inside this folder, the pipeline writes:

```text
checkpoints/
  best_model.pt
  last_model.pt
logs/
  run.log
plots/
  loss_curve.png
tables/
  metrics.csv
tensor_shards/
  train/
    train_shard_00000.pt
    train_shard_00001.pt
  val/
    val_shard_00000.pt
normalization_stats.json
summary.json
```

The checkpoint files contain `model_state_dict`, and can be reused for downstream fine-tuning.

## Tensor shards

Tensor shards are saved as PyTorch `.pt` files. Each shard contains:

- `values`: numeric feature tensor with shape `[N, F]`;
- `observed_mask`: boolean tensor showing which values were originally observed;
- `feature_names`: canonical feature order;
- `cohort_name`: cohort name;
- `sample_ids`: optional sample identifiers;
- `num_rows` and `num_features`.

During training, `TabularFoundationDataset` loads shards lazily and keeps a small LRU cache of recently used shards. Normalization is applied when shards are loaded, using statistics fitted on the training shards only.

## Missing values and masking

The pipeline keeps two different concepts separate:

1. Naturally missing values in the original dataset.
2. Artificially masked values used for self-supervised training.

Naturally missing values are tracked with `observed_mask`. Before entering the model, missing values are filled with `0.0`, but the model still receives the missingness information through the mask embedding.

For masked feature modeling, the collator selects a subset of observed features as prediction targets. These values are hidden from the input and stored in `masked_targets`. The loss is computed only on the selected prediction positions.

## Fine-tuning on ferritin regression

The downstream script is:

```text
src/FineTune/finetune_ferritin_regression.py
```

This script compares several model families:

- transformer initialized from pretrained checkpoints;
- transformer trained from scratch;
- MLPRegressor;
- XGBoost, if installed;
- LightGBM, if installed;
- CatBoost, if installed.

It supports both raw ferritin targets and `log1p(ferritin)` targets. It also supports low-label regimes through different training-label fractions:

```python
label_fractions = [1.0, 0.5, 0.3, 0.15, 0.10, 0.05]
```

Before running the script, edit the `CONFIG` dictionary at the top of the file:

```python
CONFIG = {
    "train_csv": "PATH/TO/train.csv",
    "val_csv": "PATH/TO/val.csv",
    "test_csv": "PATH/TO/test.csv",
    "target_col": "ferritin",
    "apply_canonical_mapping": True,
    "cohort_name": "amsterdam",
    "schema_path": "PATH/TO/master_schema.xlsx",
    "checkpoint_dir": "PATH/TO/checkpoints",
    "output_dir": "PATH/TO/output_ferritin_low_label",
}
```

Then run:

```bash
python src/FineTune/finetune_ferritin_regression.py
```

The script writes result files such as:

```text
all_results_combined.csv
all_results_ranked_by_test_rmse.csv
results_for_plotting.csv
```

The main reported metrics are:

- MAE
- RMSE
- R2
- Spearman correlation
- Median absolute error

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover schema loading, cohort mapping, dataset behavior, model forward pass, and the training step.

## Current notes

This is a research codebase under active development. The main working path is the sharded pretraining pipeline in `main.py`. The fine-tuning script is currently configured through an in-file `CONFIG` dictionary rather than command-line arguments.

The helper scripts in `scripts/` may need to be updated to pass the required `--cohort-config-name` argument. The direct command shown above is the recommended way to run pretraining.

## Typical workflow

A typical experiment looks like this:

1. Prepare a raw cohort CSV.
2. Prepare or update the schema Excel file with canonical feature names and cohort-specific mappings.
3. Edit `configs/data.yaml` with the raw data and schema paths.
4. Choose the cohort config, for example `amsterdam.yaml`.
5. Run `main.py` to create tensor shards and pretrain the model.
6. Use `best_model.pt` or `last_model.pt` for downstream experiments.
7. Run the ferritin fine-tuning script or another downstream evaluation script.

## Project status

The current implementation provides a working foundation-model pipeline for FBC tabular data with:

- schema-based multi-cohort feature harmonization;
- chunked raw CSV processing;
- buffered tensor shard writing;
- train-only normalization statistics;
- lazy shard-backed training;
- transformer-based masked feature reconstruction;
- checkpointing and metric export;
- downstream ferritin regression evaluation with low-label regimes.
