# pip install pandas numpy scipy scikit-learn torch xgboost lightgbm catboost pyarrow openpyxl

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False


CONFIG = {
    "train_csv": "PATH/TO/train.csv",
    "val_csv": "PATH/TO/val.csv",
    "test_csv": "PATH/TO/test.csv",
    "target_col": "ferritin",

    "apply_canonical_mapping": True,
    "cohort_name": "amsterdam",
    "schema_path": "PATH/TO/master_schema.xlsx",

    "canonical_feature_list_path": None,

    "checkpoint_dir": "PATH/TO/checkpoints",
    "checkpoint_glob": "*.pt",

    "output_dir": "PATH/TO/output_ferritin_low_label",

    "model_hparams": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "pooling_type": "mean",
        "regression_head_hidden_dim": None,
        "projection_dim": None,
        "projection_hidden_dim": None,
    },

    "downstream_head_hidden_dim": 192,
    "downstream_head_dropout": 0.1,

    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "patience": 8,
    "min_delta": 0.0,
    "seed": 42,
    "num_workers": 0,

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "run_raw_target": True,
    "run_log1p_target": True,

    "label_fractions": [1.0, 0.5, 0.3, 0.15, 0.10, 0.05],
    "subsample_seed": 42,
}


try:
    from src.data.mapper import build_canonical_dataframe, normalize_dataframe_columns
except Exception:
    build_canonical_dataframe = None
    normalize_dataframe_columns = None

try:
    from src.data.loader import load_master_schema
except Exception:
    load_master_schema = None

try:
    from src.models.model import TabularFoundationModel
except Exception as e:
    print("IMPORT ERROR for TabularFoundationModel:", repr(e))
    raise


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_name(value: object) -> str:
    import re
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


def maybe_load_feature_order(path: Optional[str]) -> Optional[List[str]]:
    if path is None or str(path).strip() == "":
        return None

    path = Path(path)
    if not path.exists():
        print(f"[WARN] canonical_feature_list_path not found: {path}")
        return None

    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [normalize_name(x) for x in obj]

    if isinstance(obj, dict):
        if "amsterdam_features" in obj:
            return [normalize_name(x) for x in obj["amsterdam_features"]]
        for _, v in obj.items():
            if isinstance(v, list):
                return [normalize_name(x) for x in v]

    raise ValueError(
        f"Could not parse feature order file: {path}. "
        f"Expected either a list or a dict containing 'amsterdam_features'."
    )


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    sp = spearmanr(y_true, y_pred)
    spearman_value = float(sp.correlation) if sp.correlation is not None else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": safe_rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman_value,
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@dataclass
class StandardizationStats:
    mean: pd.Series
    std: pd.Series


def convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace(["-", " - ", ""], np.nan)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fit_standardization_stats(train_df: pd.DataFrame) -> StandardizationStats:
    mean = train_df.mean(axis=0, skipna=True)
    std = train_df.std(axis=0, skipna=True, ddof=0)
    return StandardizationStats(mean=mean, std=std)


def apply_standardization(
    df: pd.DataFrame,
    stats: StandardizationStats,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        col = out[c].astype(float)
        mean = stats.mean[c]
        std = stats.std[c]
        if pd.isna(std) or std == 0:
            standardized = col - mean
        else:
            standardized = (col - mean) / std
        standardized = standardized.clip(lower=clip_min, upper=clip_max)
        out[c] = standardized
    return out


def prepare_features_and_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], StandardizationStats]:
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df.")
    if target_col not in val_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in val_df.")
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test_df.")

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").values.astype(np.float32)
    y_val = pd.to_numeric(val_df[target_col], errors="coerce").values.astype(np.float32)
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").values.astype(np.float32)

    train_keep = ~np.isnan(y_train)
    val_keep = ~np.isnan(y_val)
    test_keep = ~np.isnan(y_test)

    train_df = train_df.loc[train_keep].reset_index(drop=True)
    val_df = val_df.loc[val_keep].reset_index(drop=True)
    test_df = test_df.loc[test_keep].reset_index(drop=True)

    y_train = y_train[train_keep]
    y_val = y_val[val_keep]
    y_test = y_test[test_keep]

    feature_cols = [c for c in train_df.columns if c != target_col]

    if feature_order is not None:
        missing_train = [c for c in feature_order if c not in train_df.columns]
        missing_val = [c for c in feature_order if c not in val_df.columns]
        missing_test = [c for c in feature_order if c not in test_df.columns]

        if missing_train or missing_val or missing_test:
            raise ValueError(
                "Feature-order file contains columns missing from one or more splits.\n"
                f"missing_train[:20]={missing_train[:20]}\n"
                f"missing_val[:20]={missing_val[:20]}\n"
                f"missing_test[:20]={missing_test[:20]}"
            )
        feature_cols = feature_order
    else:
        feature_cols = [c for c in feature_cols if c in val_df.columns and c in test_df.columns]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X_train = convert_columns_to_numeric(X_train)
    X_val = convert_columns_to_numeric(X_val)
    X_test = convert_columns_to_numeric(X_test)

    stats = fit_standardization_stats(X_train)
    X_train_std = apply_standardization(X_train, stats)
    X_val_std = apply_standardization(X_val, stats)
    X_test_std = apply_standardization(X_test, stats)

    return X_train_std, X_val_std, X_test_std, y_train, y_val, y_test, feature_cols, stats


def subsample_training_data(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    if fraction == 1.0:
        return X_train_df.reset_index(drop=True), y_train.copy()

    n = len(X_train_df)
    n_keep = max(1, int(round(n * fraction)))

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=n_keep, replace=False)
    indices = np.sort(indices)

    return (
        X_train_df.iloc[indices].reset_index(drop=True),
        y_train[indices].copy(),
    )


class FerritinTabularDataset(Dataset):
    def __init__(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        feature_columns: List[str],
        log1p_target: bool = False,
    ):
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)

        values = X_df[feature_columns].values.astype(np.float32)
        observed_mask = ~np.isnan(values)
        values_zero_filled = np.where(observed_mask, values, 0.0).astype(np.float32)

        feature_ids = np.arange(self.num_features, dtype=np.int64)
        feature_ids = np.tile(feature_ids, (len(X_df), 1))

        input_mask = np.ones_like(values_zero_filled, dtype=np.float32)

        target = y.astype(np.float32)
        if log1p_target:
            if np.any(target < 0):
                raise ValueError("log1p target requested but negative ferritin values found.")
            target = np.log1p(target).astype(np.float32)

        self.feature_ids = torch.from_numpy(feature_ids).long()
        self.values = torch.from_numpy(values_zero_filled).float()
        self.observed_mask = torch.from_numpy(observed_mask.astype(np.float32)).float()
        self.input_mask = torch.from_numpy(input_mask).float()
        self.target = torch.from_numpy(target).float()

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        return {
            "feature_ids": self.feature_ids[idx],
            "values": self.values[idx],
            "observed_mask": self.observed_mask[idx],
            "input_mask": self.input_mask[idx],
            "target": self.target[idx],
        }


def build_pretraining_model(num_features: int, model_hparams: dict):
    required_keys = ["d_model", "nhead", "num_layers", "dim_feedforward"]
    missing = [k for k in required_keys if k not in model_hparams]
    if missing:
        raise KeyError(
            f"Missing required model_hparams keys: {missing}. "
            f"Received keys: {list(model_hparams.keys())}"
        )

    try:
        model = TabularFoundationModel(
            num_features=num_features,
            d_model=model_hparams["d_model"],
            nhead=model_hparams["nhead"],
            num_layers=model_hparams["num_layers"],
            dim_feedforward=model_hparams["dim_feedforward"],
            dropout=model_hparams.get("dropout", 0.1),
            pooling_type=model_hparams.get("pooling_type", "mean"),
            regression_head_hidden_dim=model_hparams.get("regression_head_hidden_dim"),
            projection_dim=model_hparams.get("projection_dim"),
            projection_hidden_dim=model_hparams.get("projection_hidden_dim"),
        )
    except TypeError as e:
        raise TypeError(
            "TabularFoundationModel constructor does not match the arguments used in "
            "build_pretraining_model(). Check src/models/model.py __init__ signature."
        ) from e

    return model


class FerritinRegressorFromFoundation(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        head_hidden_dim: int = 192,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(
            feature_ids=feature_ids,
            values=values,
            observed_mask=observed_mask,
            input_mask=input_mask,
        )

        if not isinstance(out, dict):
            raise ValueError("Backbone forward() is expected to return a dict.")
        if "pooled_embedding" not in out:
            raise ValueError("Backbone output dict does not contain 'pooled_embedding'.")

        pooled = out["pooled_embedding"]
        pred = self.regression_head(pooled).squeeze(-1)
        return pred


def load_pretrained_checkpoint(backbone: nn.Module, checkpoint_path: str | Path) -> Dict[str, List[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'. "
            f"Available keys: {list(ckpt.keys())}"
        )

    state_dict = ckpt["model_state_dict"]
    incompatible = backbone.load_state_dict(state_dict, strict=False)

    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    min_delta: float
    device: str
    num_workers: int


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.best_state = None
        self.counter = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: str):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    all_losses = []
    all_preds = []
    all_targets = []

    for batch in loader:
        feature_ids = batch["feature_ids"].to(device)
        values = batch["values"].to(device)
        observed_mask = batch["observed_mask"].to(device)
        input_mask = batch["input_mask"].to(device)
        target = batch["target"].to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(
            feature_ids=feature_ids,
            values=values,
            observed_mask=observed_mask,
            input_mask=input_mask,
        )

        loss = criterion(pred, target)

        if is_train:
            loss.backward()
            optimizer.step()

        all_losses.append(loss.item())
        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return float(np.mean(all_losses)), all_preds, all_targets


def train_torch_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.device)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, cfg.device)

    stopper = EarlyStopper(cfg.patience, cfg.min_delta)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, cfg.device, criterion)
        val_loss, _, _ = run_epoch(model, val_loader, None, cfg.device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        should_stop = stopper.step(val_loss, model)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model, history


@torch.no_grad()
def predict_torch_model(model: nn.Module, dataset: Dataset, cfg: TrainConfig) -> np.ndarray:
    model.eval()
    model.to(cfg.device)
    loader = make_loader(dataset, cfg.batch_size, False, cfg.num_workers, cfg.device)

    preds = []
    for batch in loader:
        feature_ids = batch["feature_ids"].to(cfg.device)
        values = batch["values"].to(cfg.device)
        observed_mask = batch["observed_mask"].to(cfg.device)
        input_mask = batch["input_mask"].to(cfg.device)

        pred = model(
            feature_ids=feature_ids,
            values=values,
            observed_mask=observed_mask,
            input_mask=input_mask,
        )
        preds.append(pred.cpu().numpy())

    return np.concatenate(preds)


def invert_if_log(y: np.ndarray, use_log1p: bool) -> np.ndarray:
    return np.expm1(y) if use_log1p else y


def evaluate_target_space(y_true: np.ndarray, y_pred: np.ndarray, use_log1p: bool) -> Dict[str, float]:
    y_true_eval = invert_if_log(y_true, use_log1p)
    y_pred_eval = invert_if_log(y_pred, use_log1p)
    return compute_metrics(y_true_eval, y_pred_eval)


def train_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[MLPRegressor, Dict[str, float]]:
    model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return model, compute_metrics(y_val, val_pred)


def train_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
):
    if not HAS_XGB:
        raise ImportError("xgboost is not installed.")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict(X_val)
    return model, compute_metrics(y_val, val_pred)


def train_lightgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
):
    if not HAS_LGBM:
        raise ImportError("lightgbm is not installed.")

    model = LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=seed,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred = model.predict(X_val)
    return model, compute_metrics(y_val, val_pred)


def train_catboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
):
    if not HAS_CAT:
        raise ImportError("catboost is not installed.")

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        use_best_model=True,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    val_pred = model.predict(X_val)
    return model, compute_metrics(y_val, val_pred)


def maybe_apply_canonical_mapping(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    if not cfg["apply_canonical_mapping"]:
        out.columns = [normalize_name(c) for c in out.columns]
        return out

    if build_canonical_dataframe is None:
        raise ImportError(
            "apply_canonical_mapping=True but build_canonical_dataframe could not be imported."
        )

    if load_master_schema is None:
        raise ImportError(
            "apply_canonical_mapping=True but load_master_schema could not be imported "
            "from src.data.loader."
        )

    target_col = normalize_name(cfg["target_col"])
    out.columns = [normalize_name(c) for c in out.columns]

    target_series = None
    if target_col in out.columns:
        target_series = out[target_col].copy()
        feature_df = out.drop(columns=[target_col])
    else:
        feature_df = out

    schema = load_master_schema(cfg["schema_path"])

    mapped_df = build_canonical_dataframe(
        df=feature_df,
        schema=schema,
        cohort_name=cfg["cohort_name"],
        fill_missing_features=True,
        strict=False,
    )

    mapped_df.columns = [normalize_name(c) for c in mapped_df.columns]

    if target_series is not None:
        mapped_df[target_col] = target_series.values

    return mapped_df


def load_all_splits(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = read_csv_safe(cfg["train_csv"])
    val_df = read_csv_safe(cfg["val_csv"])
    test_df = read_csv_safe(cfg["test_csv"])

    train_df = maybe_apply_canonical_mapping(train_df, cfg)
    val_df = maybe_apply_canonical_mapping(val_df, cfg)
    test_df = maybe_apply_canonical_mapping(test_df, cfg)

    target_col = normalize_name(cfg["target_col"])
    train_df.columns = [normalize_name(c) for c in train_df.columns]
    val_df.columns = [normalize_name(c) for c in val_df.columns]
    test_df.columns = [normalize_name(c) for c in test_df.columns]

    if target_col != cfg["target_col"]:
        cfg["target_col"] = target_col

    print("Mapped train shape:", train_df.shape)
    print("Mapped val shape:", val_df.shape)
    print("Mapped test shape:", test_df.shape)

    return train_df, val_df, test_df


def run_one_transform_setting(
    transform_name: str,
    use_log1p_target: bool,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> List[Dict[str, object]]:
    results = []

    feature_order = maybe_load_feature_order(cfg.get("canonical_feature_list_path"))

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, feature_cols, stats = prepare_features_and_target(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=cfg["target_col"],
        feature_order=feature_order,
    )

    print("Number of features used:", len(feature_cols))
    print("First 20 features:", feature_cols[:20])

    train_cfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        device=cfg["device"],
        num_workers=cfg["num_workers"],
    )

    output_dir = Path(cfg["output_dir"]) / transform_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "transform_name": transform_name,
            "use_log1p_target": use_log1p_target,
            "num_features": len(feature_cols),
            "feature_cols": feature_cols,
            "config": cfg,
        },
        output_dir / "run_config.json",
    )

    checkpoint_paths = sorted(Path(cfg["checkpoint_dir"]).glob(cfg["checkpoint_glob"]))

    for label_fraction in cfg["label_fractions"]:
        print(f"\n===== [{transform_name}] label_fraction={label_fraction:.2f} =====")

        X_train_frac_df, y_train_frac = subsample_training_data(
            X_train_df=X_train_df,
            y_train=y_train,
            fraction=label_fraction,
            seed=cfg["subsample_seed"],
        )

        train_ds = FerritinTabularDataset(
            X_train_frac_df,
            y_train_frac,
            feature_cols,
            log1p_target=use_log1p_target,
        )
        val_ds = FerritinTabularDataset(
            X_val_df,
            y_val,
            feature_cols,
            log1p_target=use_log1p_target,
        )
        test_ds = FerritinTabularDataset(
            X_test_df,
            y_test,
            feature_cols,
            log1p_target=use_log1p_target,
        )

        print("Train rows used:", len(X_train_frac_df))
        print("Val rows:", len(X_val_df))
        print("Test rows:", len(X_test_df))

        fraction_tag = f"frac_{str(label_fraction).replace('.', 'p')}"
        fraction_output_dir = output_dir / fraction_tag
        fraction_output_dir.mkdir(parents=True, exist_ok=True)

        for ckpt_path in checkpoint_paths:
            print(f"\n=== [{transform_name}] [{fraction_tag}] Pretrained checkpoint: {ckpt_path.name} ===")

            try:
                backbone = build_pretraining_model(
                    num_features=len(feature_cols),
                    model_hparams=cfg["model_hparams"],
                )

                load_info = load_pretrained_checkpoint(backbone, ckpt_path)

                model = FerritinRegressorFromFoundation(
                    backbone=backbone,
                    d_model=cfg["model_hparams"]["d_model"],
                    head_hidden_dim=cfg["downstream_head_hidden_dim"],
                    head_dropout=cfg["downstream_head_dropout"],
                )

                model, history = train_torch_model(model, train_ds, val_ds, train_cfg)

                val_pred = predict_torch_model(model, val_ds, train_cfg)
                test_pred = predict_torch_model(model, test_ds, train_cfg)

                val_target_used = np.log1p(y_val) if use_log1p_target else y_val
                test_target_used = np.log1p(y_test) if use_log1p_target else y_test

                val_metrics = evaluate_target_space(val_target_used, val_pred, use_log1p_target)
                test_metrics = evaluate_target_space(test_target_used, test_pred, use_log1p_target)

                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "transformer_pretrained",
                    "model_name": ckpt_path.stem,
                    "checkpoint_path": str(ckpt_path),
                    "num_features": len(feature_cols),
                    "missing_keys_count": len(load_info["missing_keys"]),
                    "unexpected_keys_count": len(load_info["unexpected_keys"]),
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                })

                save_json(
                    {
                        "history": history,
                        "load_info": load_info,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    },
                    fraction_output_dir / f"{ckpt_path.stem}_details.json",
                )

            except Exception as e:
                print(f"[FAILED] {ckpt_path.name}: {e}")
                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "transformer_pretrained",
                    "model_name": ckpt_path.stem,
                    "checkpoint_path": str(ckpt_path),
                    "error": str(e),
                })

        print(f"\n=== [{transform_name}] [{fraction_tag}] Transformer from scratch ===")
        try:
            scratch_backbone = build_pretraining_model(
                num_features=len(feature_cols),
                model_hparams=cfg["model_hparams"],
            )

            scratch_model = FerritinRegressorFromFoundation(
                backbone=scratch_backbone,
                d_model=cfg["model_hparams"]["d_model"],
                head_hidden_dim=cfg["downstream_head_hidden_dim"],
                head_dropout=cfg["downstream_head_dropout"],
            )

            scratch_model, scratch_history = train_torch_model(scratch_model, train_ds, val_ds, train_cfg)

            val_pred = predict_torch_model(scratch_model, val_ds, train_cfg)
            test_pred = predict_torch_model(scratch_model, test_ds, train_cfg)

            val_target_used = np.log1p(y_val) if use_log1p_target else y_val
            test_target_used = np.log1p(y_test) if use_log1p_target else y_test

            val_metrics = evaluate_target_space(val_target_used, val_pred, use_log1p_target)
            test_metrics = evaluate_target_space(test_target_used, test_pred, use_log1p_target)

            results.append({
                "target_transform": transform_name,
                "label_fraction": label_fraction,
                "train_rows_used": len(X_train_frac_df),
                "val_rows": len(X_val_df),
                "test_rows": len(X_test_df),
                "model_family": "transformer_scratch",
                "model_name": "transformer_scratch",
                "checkpoint_path": "",
                "missing_keys_count": 0,
                "unexpected_keys_count": 0,
                "num_features": len(feature_cols),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })

            save_json(
                {
                    "history": scratch_history,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                fraction_output_dir / "transformer_scratch_details.json",
            )

        except Exception as e:
            print(f"[FAILED] transformer_scratch: {e}")
            results.append({
                "target_transform": transform_name,
                "label_fraction": label_fraction,
                "train_rows_used": len(X_train_frac_df),
                "val_rows": len(X_val_df),
                "test_rows": len(X_test_df),
                "model_family": "transformer_scratch",
                "model_name": "transformer_scratch",
                "checkpoint_path": "",
                "error": str(e),
            })

        X_train_np = np.where(np.isnan(X_train_frac_df.values), 0.0, X_train_frac_df.values).astype(np.float32)
        X_val_np = np.where(np.isnan(X_val_df.values), 0.0, X_val_df.values).astype(np.float32)
        X_test_np = np.where(np.isnan(X_test_df.values), 0.0, X_test_df.values).astype(np.float32)

        y_train_baseline = np.log1p(y_train_frac) if use_log1p_target else y_train_frac
        y_val_baseline = np.log1p(y_val) if use_log1p_target else y_val
        y_test_baseline = np.log1p(y_test) if use_log1p_target else y_test

        print(f"\n=== [{transform_name}] [{fraction_tag}] MLPRegressor ===")
        try:
            mlp_model, _ = train_mlp_regressor(
                X_train_np, y_train_baseline, X_val_np, y_val_baseline, cfg["seed"]
            )
            val_pred = mlp_model.predict(X_val_np)
            test_pred = mlp_model.predict(X_test_np)

            val_metrics = evaluate_target_space(y_val_baseline, val_pred, use_log1p_target)
            test_metrics = evaluate_target_space(y_test_baseline, test_pred, use_log1p_target)

            results.append({
                "target_transform": transform_name,
                "label_fraction": label_fraction,
                "train_rows_used": len(X_train_frac_df),
                "val_rows": len(X_val_df),
                "test_rows": len(X_test_df),
                "model_family": "baseline",
                "model_name": "mlp_regressor",
                "checkpoint_path": "",
                "num_features": len(feature_cols),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })
        except Exception as e:
            print(f"[FAILED] mlp_regressor: {e}")
            results.append({
                "target_transform": transform_name,
                "label_fraction": label_fraction,
                "train_rows_used": len(X_train_frac_df),
                "val_rows": len(X_val_df),
                "test_rows": len(X_test_df),
                "model_family": "baseline",
                "model_name": "mlp_regressor",
                "checkpoint_path": "",
                "error": str(e),
            })

        print(f"\n=== [{transform_name}] [{fraction_tag}] XGBoost ===")
        if HAS_XGB:
            try:
                xgb_model, _ = train_xgboost_regressor(
                    X_train_np, y_train_baseline, X_val_np, y_val_baseline, cfg["seed"]
                )
                val_pred = xgb_model.predict(X_val_np)
                test_pred = xgb_model.predict(X_test_np)

                val_metrics = evaluate_target_space(y_val_baseline, val_pred, use_log1p_target)
                test_metrics = evaluate_target_space(y_test_baseline, test_pred, use_log1p_target)

                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "baseline",
                    "model_name": "xgboost",
                    "checkpoint_path": "",
                    "num_features": len(feature_cols),
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                })
            except Exception as e:
                print(f"[FAILED] xgboost: {e}")
                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "baseline",
                    "model_name": "xgboost",
                    "checkpoint_path": "",
                    "error": str(e),
                })

        print(f"\n=== [{transform_name}] [{fraction_tag}] LightGBM ===")
        if HAS_LGBM:
            try:
                lgbm_model, _ = train_lightgbm_regressor(
                    X_train_np, y_train_baseline, X_val_np, y_val_baseline, cfg["seed"]
                )
                val_pred = lgbm_model.predict(X_val_np)
                test_pred = lgbm_model.predict(X_test_np)

                val_metrics = evaluate_target_space(y_val_baseline, val_pred, use_log1p_target)
                test_metrics = evaluate_target_space(y_test_baseline, test_pred, use_log1p_target)

                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "baseline",
                    "model_name": "lightgbm",
                    "checkpoint_path": "",
                    "num_features": len(feature_cols),
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                })
            except Exception as e:
                print(f"[FAILED] lightgbm: {e}")
                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "baseline",
                    "model_name": "lightgbm",
                    "checkpoint_path": "",
                    "error": str(e),
                })

        print(f"\n=== [{transform_name}] [{fraction_tag}] CatBoost ===")
        if HAS_CAT:
            try:
                cat_model, _ = train_catboost_regressor(
                    X_train_np, y_train_baseline, X_val_np, y_val_baseline, cfg["seed"]
                )
                val_pred = cat_model.predict(X_val_np)
                test_pred = cat_model.predict(X_test_np)

                val_metrics = evaluate_target_space(y_val_baseline, val_pred, use_log1p_target)
                test_metrics = evaluate_target_space(y_test_baseline, test_pred, use_log1p_target)

                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "baseline",
                    "model_name": "catboost",
                    "checkpoint_path": "",
                    "num_features": len(feature_cols),
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                })
            except Exception as e:
                print(f"[FAILED] catboost: {e}")
                results.append({
                    "target_transform": transform_name,
                    "label_fraction": label_fraction,
                    "train_rows_used": len(X_train_frac_df),
                    "val_rows": len(X_val_df),
                    "test_rows": len(X_test_df),
                    "model_family": "baseline",
                    "model_name": "catboost",
                    "checkpoint_path": "",
                    "error": str(e),
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    if "test_rmse" in results_df.columns:
        ranked = results_df.sort_values(
            by=["label_fraction", "test_rmse"],
            ascending=[False, True],
            na_position="last",
        )
        ranked.to_csv(output_dir / "results_sorted_by_test_rmse.csv", index=False)

    return results


def main():
    set_seed(CONFIG["seed"])

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, val_df, test_df = load_all_splits(CONFIG)

    print("train shape:", train_df.shape)
    print("val shape:", val_df.shape)
    print("test shape:", test_df.shape)

    all_results = []

    if CONFIG["run_raw_target"]:
        print("\n##############################")
        print("RUNNING RAW FERRITIN")
        print("##############################")
        all_results.extend(
            run_one_transform_setting(
                transform_name="raw_target",
                use_log1p_target=False,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                cfg=CONFIG,
            )
        )

    if CONFIG["run_log1p_target"]:
        print("\n##############################")
        print("RUNNING LOG1P(FERRITIN)")
        print("##############################")
        all_results.extend(
            run_one_transform_setting(
                transform_name="log1p_target",
                use_log1p_target=True,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                cfg=CONFIG,
            )
        )

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(out_dir / "all_results_combined.csv", index=False)

    if "test_rmse" in all_results_df.columns:
        all_ranked = all_results_df.sort_values(
            by=["target_transform", "label_fraction", "test_rmse"],
            ascending=[True, False, True],
            na_position="last",
        )
        all_ranked.to_csv(out_dir / "all_results_ranked_by_test_rmse.csv", index=False)

    plot_cols = [
        "target_transform",
        "label_fraction",
        "train_rows_used",
        "val_rows",
        "test_rows",
        "model_family",
        "model_name",
        "checkpoint_path",
        "num_features",
        "val_mae",
        "val_rmse",
        "val_r2",
        "val_spearman",
        "val_median_ae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
        "test_median_ae",
    ]
    available_plot_cols = [c for c in plot_cols if c in all_results_df.columns]
    all_results_df[available_plot_cols].to_csv(
        out_dir / "results_for_plotting.csv",
        index=False,
    )

    print("\nDone.")
    print("Saved:")
    print(out_dir / "all_results_combined.csv")
    if "test_rmse" in all_results_df.columns:
        print(out_dir / "all_results_ranked_by_test_rmse.csv")
    print(out_dir / "results_for_plotting.csv")


if __name__ == "__main__":
    main()
