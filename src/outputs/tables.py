from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def write_metrics_table(history: List[Dict[str, float]], output_path: Path) -> None:
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
