from __future__ import annotations

from typing import Dict, List


def average_metric_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {"loss": 0.0}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}
