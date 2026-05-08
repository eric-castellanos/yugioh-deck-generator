from __future__ import annotations

from typing import Any


def composite_weight(row: dict[str, Any], base: float = 1.0) -> float:
    probability = float(row.get("cluster_probability", 1.0) or 1.0)
    novelty_boost = float(row.get("novelty_weight", 1.0) or 1.0)
    outlier_penalty = 1.0 - min(max(float(row.get("outlier_score", 0.0) or 0.0), 0.0), 0.95)
    return max(0.0001, base * probability * novelty_boost * outlier_penalty)
