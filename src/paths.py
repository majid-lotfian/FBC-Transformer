from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class PathManager:
    root: Path
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    plots_dir: Path
    tables_dir: Path

    log_file: Path
    metrics_file: Path
    summary_file: Path
    config_snapshot_file: Path

    device: torch.device

    @classmethod
    def from_config(cls, cfg) -> "PathManager":
        root = Path(cfg.paths.artifacts_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{cfg.experiment.name}_{timestamp}"

        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        plots_dir = run_dir / "plots"
        tables_dir = run_dir / "tables"

        for path in [run_dir, checkpoints_dir, logs_dir, plots_dir, tables_dir]:
            path.mkdir(parents=True, exist_ok=True)

        device = cls._resolve_device(cfg.experiment.device)

        return cls(
            root=root,
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
            log_file=logs_dir / "run.log",
            metrics_file=tables_dir / "metrics.csv",
            summary_file=run_dir / "summary.json",
            config_snapshot_file=run_dir / "config_snapshot.yaml",
            device=device,
        )

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        device_name = str(device_name).strip().lower()

        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("Config requested 'cuda' but CUDA is not available.")

        return torch.device(device_name)
