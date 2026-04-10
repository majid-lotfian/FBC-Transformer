from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _ns_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _ns_to_dict(v) for k, v in obj.items()}
    return obj


def export_run_summary(cfg, history: List[Dict[str, float]], output_path: Path) -> None:
    summary = {
        "config": _ns_to_dict(cfg),
        "history": history,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
