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

        if cfg.experiment.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.experiment.device)

        return cls(
            root=root,
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
            log_file=logs_dir / "run.log",
            device=device,
        )
