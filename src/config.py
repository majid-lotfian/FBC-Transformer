from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    experiment: SimpleNamespace
    cohort: SimpleNamespace
    paths: SimpleNamespace
    run: SimpleNamespace
    data: SimpleNamespace
    model: SimpleNamespace
    train: SimpleNamespace
    objective: SimpleNamespace
    output: SimpleNamespace


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_experiment_config(config_dir: Path) -> ExperimentConfig:
    config_files = [
        config_dir / "base.yaml",
        config_dir / "data.yaml",
        config_dir / "model.yaml",
        config_dir / "train.yaml",
        config_dir / "output.yaml",
        config_dir / "cohort" / "cohort_a.yaml",
    ]
    merged: Dict[str, Any] = {}
    for path in config_files:
        merged = _deep_update(merged, load_yaml(path))

    return ExperimentConfig(
        experiment=_to_namespace(merged["experiment"]),
        cohort=_to_namespace(merged["cohort"]),
        paths=_to_namespace(merged["paths"]),
        run=_to_namespace(merged["run"]),
        data=_to_namespace(merged["data"]),
        model=_to_namespace(merged["model"]),
        train=_to_namespace(merged["train"]),
        objective=_to_namespace(merged["objective"]),
        output=_to_namespace(merged["output"]),
    )
