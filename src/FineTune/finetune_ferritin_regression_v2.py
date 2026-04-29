# pip install pandas numpy scipy scikit-learn torch pyarrow openpyxl

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
from dataclasses import dataclass, field
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "train_csv": "Z:/Bloodcounts/Majid-sensitive/splits/with-ferritin/train_with_ferritin.csv",
    "val_csv":   "Z:/Bloodcounts/Majid-sensitive/splits/with-ferritin/val_with_ferritin.csv",
    "test_csv":  "Z:/Bloodcounts/Majid-sensitive/splits/with-ferritin/test_with_ferritin.csv",
    "target_col": "ferritin",

    "apply_canonical_mapping": True,
    "cohort_name": "amsterdam",
    "schema_path": "Z:/Bloodcounts/FBC-Transformer/data/schema.xlsx",

    # Optional: path to a JSON file listing the exact feature order used during
    # pretraining (a list of canonical feature names).  Set to None to fall back
    # on whatever columns the canonical mapper returns – they should be identical
    # as long as the schema has not changed.
    "canonical_feature_list_path": None,

    # ---- UPDATE to your v3 checkpoint directory ----
    "checkpoint_dir": "Z:/Bloodcounts/FBC-Transformer/artifacts/blood_foundation_v3_XXXXXXXX/checkpoints",
    "checkpoint_glob": "best_model.pt",

    "output_dir": "Z:/Bloodcounts/FBC-Transformer/finetunningOutput/output_ferritin_v2",

    # Must exactly match the pretraining model architecture so that the full
    # checkpoint (including projection_head) loads without missing or unexpected keys.
    "model_hparams": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "pooling_type": "mean",
        "regression_head_hidden_dim": None,
        "projection_dim": 64,          # must match pretraining
        "projection_hidden_dim": 128,  # must match pretraining
    },

    "downstream_head_hidden_dim": 192,
    "downstream_head_dropout": 0.2,

    # ---- Training ----
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 15,
    "min_delta": 1e-6,
    "seed": 42,
    "num_workers": 0,

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Two-phase fine-tuning for the pretrained model:
    #   Phase 1 – backbone frozen, only the downstream head is trained.
    #             This lets the randomly-initialised head converge to a
    #             reasonable starting point before gradients flow into the
    #             pretrained backbone.
    #   Phase 2 – backbone unfrozen with a much smaller LR so pretrained
    #             representations are refined rather than overwritten.
    "freeze_backbone_epochs": 5,
    "backbone_lr_multiplier": 0.1,   # backbone_lr = lr * this

    # Gradient clipping (applied to trainable params only)
    "grad_clip_norm": 1.0,

    # Cosine annealing LR scheduler, applied during phase 2 (and the single
    # phase used for the from-scratch model)
    "scheduler_t_max": 30,
    "scheduler_eta_min": 1e-6,

    # Huber loss is more robust to the heavy-tailed ferritin distribution than
    # plain MSE.  Set to False to use MSELoss instead.
    "use_huber_loss": True,
    "huber_delta": 1.0,

    "run_raw_target":   True,
    "run_log1p_target": True,

    "label_fractions": [1.0, 0.5, 0.3, 0.15, 0.10, 0.05],
    "subsample_seed": 42,
}


# ---------------------------------------------------------------------------
# Imports from project source
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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
        "Expected either a list or a dict containing 'amsterdam_features'."
    )


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    sp = spearmanr(y_true, y_pred)
    spearman_value = float(sp.correlation) if sp.correlation is not None else np.nan
    return {
        "mae":       float(mean_absolute_error(y_true, y_pred)),
        "rmse":      safe_rmse(y_true, y_pred),
        "r2":        float(r2_score(y_true, y_pred)),
        "spearman":  spearman_value,
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def invert_if_log(y: np.ndarray, use_log1p: bool) -> np.ndarray:
    return np.expm1(y) if use_log1p else y


def evaluate_target_space(
    y_true: np.ndarray, y_pred: np.ndarray, use_log1p: bool
) -> Dict[str, float]:
    return compute_metrics(invert_if_log(y_true, use_log1p), invert_if_log(y_pred, use_log1p))


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------
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
    return StandardizationStats(
        mean=train_df.mean(axis=0, skipna=True),
        std=train_df.std(axis=0, skipna=True, ddof=0),
    )


def apply_standardization(
    df: pd.DataFrame,
    stats: StandardizationStats,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        col  = out[c].astype(float)
        mean = stats.mean[c]
        std  = stats.std[c]
        standardized = (col - mean) / std if (not pd.isna(std) and std != 0) else (col - mean)
        out[c] = standardized.clip(lower=clip_min, upper=clip_max)
    return out


def prepare_features_and_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], StandardizationStats]:
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if target_col not in split_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {split_name}_df.")

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").values.astype(np.float32)
    y_val   = pd.to_numeric(val_df[target_col],   errors="coerce").values.astype(np.float32)
    y_test  = pd.to_numeric(test_df[target_col],  errors="coerce").values.astype(np.float32)

    train_df = train_df.loc[~np.isnan(y_train)].reset_index(drop=True)
    val_df   = val_df.loc[~np.isnan(y_val)].reset_index(drop=True)
    test_df  = test_df.loc[~np.isnan(y_test)].reset_index(drop=True)

    y_train = y_train[~np.isnan(y_train)]
    y_val   = y_val[~np.isnan(y_val)]
    y_test  = y_test[~np.isnan(y_test)]

    if feature_order is not None:
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            missing = [c for c in feature_order if c not in split_df.columns]
            if missing:
                raise ValueError(
                    f"Feature-order file contains columns missing from {split_name}_df.\n"
                    f"First 20 missing: {missing[:20]}"
                )
        feature_cols = feature_order
    else:
        feature_cols = [
            c for c in train_df.columns
            if c != target_col and c in val_df.columns and c in test_df.columns
        ]

    X_train = convert_columns_to_numeric(train_df[feature_cols].copy())
    X_val   = convert_columns_to_numeric(val_df[feature_cols].copy())
    X_test  = convert_columns_to_numeric(test_df[feature_cols].copy())

    stats = fit_standardization_stats(X_train)
    return (
        apply_standardization(X_train, stats),
        apply_standardization(X_val,   stats),
        apply_standardization(X_test,  stats),
        y_train, y_val, y_test,
        feature_cols, stats,
    )


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

    n_keep = max(1, int(round(len(X_train_df) * fraction)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(X_train_df), size=n_keep, replace=False))
    return X_train_df.iloc[indices].reset_index(drop=True), y_train[indices].copy()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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
        values_filled = np.where(observed_mask, values, 0.0).astype(np.float32)

        feature_ids = np.tile(np.arange(self.num_features, dtype=np.int64), (len(X_df), 1))
        input_mask  = np.ones_like(values_filled, dtype=np.float32)

        target = y.astype(np.float32)
        if log1p_target:
            if np.any(target < 0):
                raise ValueError("log1p target requested but negative ferritin values found.")
            target = np.log1p(target)

        self.feature_ids  = torch.from_numpy(feature_ids).long()
        self.values       = torch.from_numpy(values_filled).float()
        self.observed_mask = torch.from_numpy(observed_mask.astype(np.float32)).float()
        self.input_mask   = torch.from_numpy(input_mask).float()
        self.target       = torch.from_numpy(target).float()

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        return {
            "feature_ids":   self.feature_ids[idx],
            "values":        self.values[idx],
            "observed_mask": self.observed_mask[idx],
            "input_mask":    self.input_mask[idx],
            "target":        self.target[idx],
        }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_backbone(num_features: int, model_hparams: dict) -> nn.Module:
    required = ["d_model", "nhead", "num_layers", "dim_feedforward"]
    missing = [k for k in required if k not in model_hparams]
    if missing:
        raise KeyError(f"Missing required model_hparams keys: {missing}")

    return TabularFoundationModel(
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


class FerritinRegressorFromFoundation(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        head_hidden_dim: int = 192,
        head_dropout: float = 0.2,
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
        if not isinstance(out, dict) or "pooled_embedding" not in out:
            raise ValueError("Backbone must return a dict containing 'pooled_embedding'.")
        return self.regression_head(out["pooled_embedding"]).squeeze(-1)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def infer_num_features_from_checkpoint(ckpt_path: Path) -> int:
    """Read num_features from the feature-embedding weight in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    key = "embedding.feature_embedding.embedding.weight"
    if key not in state_dict:
        raise KeyError(
            f"Cannot infer num_features from checkpoint '{ckpt_path.name}': "
            f"key '{key}' not found."
        )
    return int(state_dict[key].shape[0])


def load_pretrained_checkpoint(
    backbone: nn.Module, checkpoint_path: Path
) -> Dict[str, List[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing 'model_state_dict'. "
            f"Keys present: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )
    incompatible = backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    return {
        "missing_keys":    list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def validate_checkpoint_load_info(load_info: Dict[str, List[str]]) -> None:
    """
    Warn if any backbone keys failed to load.

    For the pretrained→fine-tune scenario the only acceptable situation is
    zero missing keys and zero unexpected keys (because we build the backbone
    with the same architecture as pretraining, including the projection head).
    """
    missing    = load_info["missing_keys"]
    unexpected = load_info["unexpected_keys"]

    if not missing and not unexpected:
        print("  [CKPT] All backbone weights loaded cleanly (no missing / unexpected keys).")
        return

    if missing:
        print(
            f"  [CKPT WARN] {len(missing)} MISSING keys – these backbone weights were NOT "
            f"restored from the checkpoint and remain randomly initialised:\n"
            f"  {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            "  => Check that model_hparams exactly matches the pretraining config."
        )
    if unexpected:
        print(
            f"  [CKPT WARN] {len(unexpected)} UNEXPECTED keys in the checkpoint that were "
            f"not used:\n"
            f"  {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}\n"
            "  => Likely a projection_dim mismatch between pretraining and finetune config."
        )


# ---------------------------------------------------------------------------
# Training infrastructure
# ---------------------------------------------------------------------------
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
    freeze_backbone_epochs: int = 5
    backbone_lr_multiplier: float = 0.1
    grad_clip_norm: Optional[float] = 1.0
    scheduler_t_max: int = 30
    scheduler_eta_min: float = 1e-6
    use_huber_loss: bool = True
    huber_delta: float = 1.0


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = math.inf
        self.best_state = None
        self.counter    = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        """Call at the phase 1→2 boundary so phase 2 gets the full patience budget."""
        self.best_loss  = math.inf
        self.best_state = None
        self.counter    = 0


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: str
) -> DataLoader:
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
    grad_clip_norm: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    all_losses, all_preds, all_targets = [], [], []

    for batch in loader:
        feature_ids   = batch["feature_ids"].to(device)
        values        = batch["values"].to(device)
        observed_mask = batch["observed_mask"].to(device)
        input_mask    = batch["input_mask"].to(device)
        target        = batch["target"].to(device)

        if is_train:
            optimizer.zero_grad()
            pred = model(
                feature_ids=feature_ids, values=values,
                observed_mask=observed_mask, input_mask=input_mask,
            )
            loss = criterion(pred, target)
            loss.backward()
            if grad_clip_norm is not None:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(
                    feature_ids=feature_ids, values=values,
                    observed_mask=observed_mask, input_mask=input_mask,
                )
                loss = criterion(pred, target)

        all_losses.append(loss.item())
        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    return float(np.mean(all_losses)), np.concatenate(all_preds), np.concatenate(all_targets)


def _make_optimizer_and_scheduler(
    model: FerritinRegressorFromFoundation,
    cfg: TrainConfig,
    differential_lr: bool,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    if differential_lr:
        param_groups = [
            {"params": model.backbone.parameters(),        "lr": cfg.lr * cfg.backbone_lr_multiplier},
            {"params": model.regression_head.parameters(), "lr": cfg.lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": cfg.lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler_t_max > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler_t_max,
            eta_min=cfg.scheduler_eta_min,
        )
    return optimizer, scheduler


def train_torch_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
    is_pretrained: bool = False,
) -> Tuple[nn.Module, Dict]:
    """
    Two-phase training when is_pretrained=True:
      Phase 1 – backbone frozen, head trains for cfg.freeze_backbone_epochs epochs.
      Phase 2 – backbone unfrozen with differential LR + cosine scheduler.

    Single-phase (cosine LR from epoch 1) when is_pretrained=False (from scratch).
    """
    model = model.to(cfg.device)
    criterion = (
        nn.HuberLoss(delta=cfg.huber_delta) if cfg.use_huber_loss else nn.MSELoss()
    )
    train_loader = make_loader(train_ds, cfg.batch_size, True,  cfg.num_workers, cfg.device)
    val_loader   = make_loader(val_ds,   cfg.batch_size, False, cfg.num_workers, cfg.device)

    stopper = EarlyStopper(cfg.patience, cfg.min_delta)
    history: Dict = {"train_loss": [], "val_loss": [], "phase": []}

    do_freeze = is_pretrained and cfg.freeze_backbone_epochs > 0

    if do_freeze:
        # ---- Phase 1: head warm-up ----
        set_requires_grad(model.backbone, False)
        optimizer_p1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        print(f"  [Phase 1] Backbone frozen — warming up head for {cfg.freeze_backbone_epochs} epochs.")

        for epoch in range(1, cfg.freeze_backbone_epochs + 1):
            tr_loss, _, _ = run_epoch(model, train_loader, optimizer_p1, cfg.device, criterion, cfg.grad_clip_norm)
            va_loss, _, _ = run_epoch(model, val_loader,   None,          cfg.device, criterion)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["phase"].append(1)
            stopper.step(va_loss, model)  # track best but don't stop in phase 1
            print(f"  Epoch {epoch:03d} [phase 1] | train={tr_loss:.6f} | val={va_loss:.6f}")

        # ---- Transition to Phase 2 ----
        set_requires_grad(model.backbone, True)
        stopper.reset()  # full patience budget for phase 2
        optimizer, scheduler = _make_optimizer_and_scheduler(model, cfg, differential_lr=True)
        backbone_lr = cfg.lr * cfg.backbone_lr_multiplier
        print(
            f"  [Phase 2] Backbone unfrozen — backbone_lr={backbone_lr:.2e}, "
            f"head_lr={cfg.lr:.2e}."
        )
    else:
        # Single phase (from scratch or no freeze)
        optimizer, scheduler = _make_optimizer_and_scheduler(model, cfg, differential_lr=False)

    # ---- Main training loop (phase 2 for pretrained, single phase for scratch) ----
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer, cfg.device, criterion, cfg.grad_clip_norm)
        va_loss, _, _ = run_epoch(model, val_loader,   None,      cfg.device, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["phase"].append(2 if do_freeze else 1)

        if scheduler is not None:
            scheduler.step()

        should_stop = stopper.step(va_loss, model)
        phase_tag   = "phase 2" if do_freeze else "phase 1"
        print(f"  Epoch {epoch:03d} [{phase_tag}] | train={tr_loss:.6f} | val={va_loss:.6f}")

        if should_stop:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model, history


@torch.no_grad()
def predict_torch_model(
    model: nn.Module, dataset: Dataset, cfg: TrainConfig
) -> np.ndarray:
    model.eval()
    model.to(cfg.device)
    loader = make_loader(dataset, cfg.batch_size, False, cfg.num_workers, cfg.device)
    preds = []
    for batch in loader:
        pred = model(
            feature_ids   = batch["feature_ids"].to(cfg.device),
            values        = batch["values"].to(cfg.device),
            observed_mask = batch["observed_mask"].to(cfg.device),
            input_mask    = batch["input_mask"].to(cfg.device),
        )
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Data loading + canonical mapping
# ---------------------------------------------------------------------------
def maybe_apply_canonical_mapping(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    if not cfg["apply_canonical_mapping"]:
        out.columns = [normalize_name(c) for c in out.columns]
        return out

    if build_canonical_dataframe is None or load_master_schema is None:
        raise ImportError(
            "apply_canonical_mapping=True but build_canonical_dataframe / "
            "load_master_schema could not be imported."
        )

    target_col = normalize_name(cfg["target_col"])
    out.columns = [normalize_name(c) for c in out.columns]

    target_series = out.pop(target_col) if target_col in out.columns else None

    schema = load_master_schema(cfg["schema_path"])
    mapped = build_canonical_dataframe(
        df=out,
        schema=schema,
        cohort_name=cfg["cohort_name"],
        fill_missing_features=True,
        strict=False,
    )
    mapped.columns = [normalize_name(c) for c in mapped.columns]

    if target_series is not None:
        mapped[target_col] = target_series.values

    return mapped


def load_all_splits(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = maybe_apply_canonical_mapping(read_csv_safe(cfg["train_csv"]), cfg)
    val_df   = maybe_apply_canonical_mapping(read_csv_safe(cfg["val_csv"]),   cfg)
    test_df  = maybe_apply_canonical_mapping(read_csv_safe(cfg["test_csv"]),  cfg)

    target_col = normalize_name(cfg["target_col"])
    for df in [train_df, val_df, test_df]:
        df.columns = [normalize_name(c) for c in df.columns]
    cfg["target_col"] = target_col

    print(f"  train: {train_df.shape}  val: {val_df.shape}  test: {test_df.shape}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Experiment runner for one target-transform setting
# ---------------------------------------------------------------------------
def run_one_transform_setting(
    transform_name: str,
    use_log1p_target: bool,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> List[Dict]:
    results: List[Dict] = []

    feature_order = maybe_load_feature_order(cfg.get("canonical_feature_list_path"))

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, feature_cols, _ = (
        prepare_features_and_target(train_df, val_df, test_df, cfg["target_col"], feature_order)
    )

    print(f"  Features: {len(feature_cols)}  |  First 10: {feature_cols[:10]}")

    train_cfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        device=cfg["device"],
        num_workers=cfg["num_workers"],
        freeze_backbone_epochs=cfg.get("freeze_backbone_epochs", 5),
        backbone_lr_multiplier=cfg.get("backbone_lr_multiplier", 0.1),
        grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
        scheduler_t_max=cfg.get("scheduler_t_max", 30),
        scheduler_eta_min=cfg.get("scheduler_eta_min", 1e-6),
        use_huber_loss=cfg.get("use_huber_loss", True),
        huber_delta=cfg.get("huber_delta", 1.0),
    )

    output_dir = Path(cfg["output_dir"]) / transform_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "transform_name":  transform_name,
            "use_log1p_target": use_log1p_target,
            "num_features":    len(feature_cols),
            "feature_cols":    feature_cols,
            "config":          cfg,
        },
        output_dir / "run_config.json",
    )

    checkpoint_paths = sorted(Path(cfg["checkpoint_dir"]).glob(cfg["checkpoint_glob"]))
    if not checkpoint_paths:
        print(f"  [WARN] No checkpoints found in {cfg['checkpoint_dir']} "
              f"matching '{cfg['checkpoint_glob']}'. Skipping pretrained runs.")

    for label_fraction in cfg["label_fractions"]:
        print(f"\n===== [{transform_name}] label_fraction={label_fraction:.2f} =====")

        X_tr_frac, y_tr_frac = subsample_training_data(
            X_train_df, y_train, label_fraction, cfg["subsample_seed"]
        )

        train_ds = FerritinTabularDataset(X_tr_frac, y_tr_frac, feature_cols, log1p_target=use_log1p_target)
        val_ds   = FerritinTabularDataset(X_val_df,  y_val,     feature_cols, log1p_target=use_log1p_target)
        test_ds  = FerritinTabularDataset(X_test_df, y_test,    feature_cols, log1p_target=use_log1p_target)

        print(f"  train rows: {len(X_tr_frac)}  val: {len(X_val_df)}  test: {len(X_test_df)}")

        fraction_tag = f"frac_{str(label_fraction).replace('.', 'p')}"
        frac_dir = output_dir / fraction_tag
        frac_dir.mkdir(parents=True, exist_ok=True)

        y_val_eval  = np.log1p(y_val)  if use_log1p_target else y_val
        y_test_eval = np.log1p(y_test) if use_log1p_target else y_test

        # ------------------------------------------------------------------ #
        # Pretrained model(s)
        # ------------------------------------------------------------------ #
        for ckpt_path in checkpoint_paths:
            print(f"\n--- [{fraction_tag}] Pretrained: {ckpt_path.name} ---")
            try:
                # Infer num_features from checkpoint and verify match
                num_features_ckpt = infer_num_features_from_checkpoint(ckpt_path)
                if num_features_ckpt != len(feature_cols):
                    raise ValueError(
                        f"num_features mismatch: checkpoint has {num_features_ckpt} features "
                        f"but fine-tuning data has {len(feature_cols)} features.  "
                        "Set canonical_feature_list_path to the feature list saved during "
                        "pretraining to align the feature spaces."
                    )

                backbone = build_backbone(num_features_ckpt, cfg["model_hparams"])
                load_info = load_pretrained_checkpoint(backbone, ckpt_path)
                validate_checkpoint_load_info(load_info)

                model = FerritinRegressorFromFoundation(
                    backbone=backbone,
                    d_model=cfg["model_hparams"]["d_model"],
                    head_hidden_dim=cfg["downstream_head_hidden_dim"],
                    head_dropout=cfg["downstream_head_dropout"],
                )

                model, history = train_torch_model(
                    model, train_ds, val_ds, train_cfg, is_pretrained=True
                )

                val_pred  = predict_torch_model(model, val_ds,  train_cfg)
                test_pred = predict_torch_model(model, test_ds, train_cfg)

                val_metrics  = evaluate_target_space(y_val_eval,  val_pred,  use_log1p_target)
                test_metrics = evaluate_target_space(y_test_eval, test_pred, use_log1p_target)

                results.append({
                    "target_transform":         transform_name,
                    "label_fraction":           label_fraction,
                    "train_rows_used":          len(X_tr_frac),
                    "val_rows":                 len(X_val_df),
                    "test_rows":                len(X_test_df),
                    "model_family":             "transformer_pretrained",
                    "model_name":               ckpt_path.stem,
                    "checkpoint_path":          str(ckpt_path),
                    "num_features":             len(feature_cols),
                    "num_features_from_ckpt":   num_features_ckpt,
                    "missing_keys_count":       len(load_info["missing_keys"]),
                    "unexpected_keys_count":    len(load_info["unexpected_keys"]),
                    **{f"val_{k}":  v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                })

                save_json(
                    {
                        "history":      history,
                        "load_info":    load_info,
                        "val_metrics":  val_metrics,
                        "test_metrics": test_metrics,
                    },
                    frac_dir / f"{ckpt_path.stem}_pretrained_details.json",
                )

            except Exception as e:
                print(f"  [FAILED] pretrained {ckpt_path.name}: {e}")
                results.append({
                    "target_transform": transform_name,
                    "label_fraction":   label_fraction,
                    "train_rows_used":  len(X_tr_frac),
                    "model_family":     "transformer_pretrained",
                    "model_name":       ckpt_path.stem,
                    "checkpoint_path":  str(ckpt_path),
                    "error":            str(e),
                })

        # ------------------------------------------------------------------ #
        # From-scratch baseline
        # ------------------------------------------------------------------ #
        print(f"\n--- [{fraction_tag}] Transformer from scratch ---")
        try:
            scratch_backbone = build_backbone(len(feature_cols), cfg["model_hparams"])
            scratch_model = FerritinRegressorFromFoundation(
                backbone=scratch_backbone,
                d_model=cfg["model_hparams"]["d_model"],
                head_hidden_dim=cfg["downstream_head_hidden_dim"],
                head_dropout=cfg["downstream_head_dropout"],
            )

            scratch_model, scratch_history = train_torch_model(
                scratch_model, train_ds, val_ds, train_cfg, is_pretrained=False
            )

            val_pred  = predict_torch_model(scratch_model, val_ds,  train_cfg)
            test_pred = predict_torch_model(scratch_model, test_ds, train_cfg)

            val_metrics  = evaluate_target_space(y_val_eval,  val_pred,  use_log1p_target)
            test_metrics = evaluate_target_space(y_test_eval, test_pred, use_log1p_target)

            results.append({
                "target_transform":       transform_name,
                "label_fraction":         label_fraction,
                "train_rows_used":        len(X_tr_frac),
                "val_rows":               len(X_val_df),
                "test_rows":              len(X_test_df),
                "model_family":           "transformer_scratch",
                "model_name":             "transformer_scratch",
                "checkpoint_path":        "",
                "num_features":           len(feature_cols),
                "num_features_from_ckpt": 0,
                "missing_keys_count":     0,
                "unexpected_keys_count":  0,
                **{f"val_{k}":  v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })

            save_json(
                {
                    "history":      scratch_history,
                    "val_metrics":  val_metrics,
                    "test_metrics": test_metrics,
                },
                frac_dir / "transformer_scratch_details.json",
            )

        except Exception as e:
            print(f"  [FAILED] transformer_scratch: {e}")
            results.append({
                "target_transform": transform_name,
                "label_fraction":   label_fraction,
                "train_rows_used":  len(X_tr_frac),
                "model_family":     "transformer_scratch",
                "model_name":       "transformer_scratch",
                "checkpoint_path":  "",
                "error":            str(e),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    if "test_rmse" in results_df.columns:
        (
            results_df
            .sort_values(["label_fraction", "test_rmse"], ascending=[False, True], na_position="last")
            .to_csv(output_dir / "results_sorted_by_test_rmse.csv", index=False)
        )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    set_seed(CONFIG["seed"])

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data splits...")
    train_df, val_df, test_df = load_all_splits(CONFIG)

    all_results: List[Dict] = []

    if CONFIG["run_raw_target"]:
        print("\n" + "=" * 60)
        print("RUNNING RAW FERRITIN TARGET")
        print("=" * 60)
        all_results.extend(
            run_one_transform_setting(
                transform_name="raw_target",
                use_log1p_target=False,
                train_df=train_df, val_df=val_df, test_df=test_df,
                cfg=CONFIG,
            )
        )

    if CONFIG["run_log1p_target"]:
        print("\n" + "=" * 60)
        print("RUNNING LOG1P(FERRITIN) TARGET")
        print("=" * 60)
        all_results.extend(
            run_one_transform_setting(
                transform_name="log1p_target",
                use_log1p_target=True,
                train_df=train_df, val_df=val_df, test_df=test_df,
                cfg=CONFIG,
            )
        )

    all_df = pd.DataFrame(all_results)
    all_df.to_csv(out_dir / "all_results_combined.csv", index=False)

    if "test_rmse" in all_df.columns:
        (
            all_df
            .sort_values(
                ["target_transform", "label_fraction", "test_rmse"],
                ascending=[True, False, True], na_position="last",
            )
            .to_csv(out_dir / "all_results_ranked_by_test_rmse.csv", index=False)
        )

    plot_cols = [
        "target_transform", "label_fraction", "train_rows_used", "val_rows", "test_rows",
        "model_family", "model_name", "checkpoint_path", "num_features",
        "val_mae", "val_rmse", "val_r2", "val_spearman", "val_median_ae",
        "test_mae", "test_rmse", "test_r2", "test_spearman", "test_median_ae",
    ]
    available = [c for c in plot_cols if c in all_df.columns]
    all_df[available].to_csv(out_dir / "results_for_plotting.csv", index=False)

    print("\nDone. Saved to:", out_dir)


if __name__ == "__main__":
    main()
