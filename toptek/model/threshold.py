"""Threshold optimisation for calibrated probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    hit_rate: float
    coverage: float
    expected_value: float


def _compute_expected_value(labels: np.ndarray) -> float:
    # Hit yields +1, miss yields -1.
    if labels.size == 0:
        return 0.0
    hit_rate = labels.mean()
    return 2 * hit_rate - 1


def opt_threshold(
    pred_probs: Iterable[float],
    labels: Iterable[int],
    *,
    min_coverage: float = 0.30,
    min_expected_value: float = 0.0,
    min_samples: int = 30,
    grid: Tuple[float, float, float] = (0.5, 0.95, 0.01),
) -> Tuple[float, List[Dict[str, float]]]:
    """Optimise confidence threshold subject to coverage/EV constraints."""

    probs = np.asarray(list(pred_probs), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if probs.shape[0] != y.shape[0]:
        raise ValueError("pred_probs and labels must share the same length")
    if probs.size == 0:
        raise ValueError("No predictions provided")

    start, stop, step = grid
    thresholds = np.arange(start, stop + 1e-9, step)
    curve: List[Dict[str, float]] = []
    best_tau = 0.5
    best_hit_rate = -np.inf

    for tau in thresholds:
        mask = probs >= tau
        coverage = float(mask.mean())
        if coverage < min_coverage:
            continue
        selected = y[mask]
        if selected.size < min_samples:
            continue
        hit_rate = float(selected.mean())
        ev = float(_compute_expected_value(selected))
        curve.append(
            {
                "threshold": float(tau),
                "hit_rate": hit_rate,
                "coverage": coverage,
                "ev": ev,
            }
        )
        if ev < min_expected_value:
            continue
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_tau = float(tau)

    if not curve:
        # Fallback to baseline threshold
        baseline_mask = probs >= 0.5
        coverage = float(baseline_mask.mean())
        selected = y[baseline_mask]
        curve.append(
            {
                "threshold": 0.5,
                "hit_rate": float(selected.mean()) if selected.size else 0.0,
                "coverage": coverage,
                "ev": float(_compute_expected_value(selected)) if selected.size else 0.0,
            }
        )
        return 0.5, curve

    return best_tau, curve


__all__ = ["opt_threshold", "ThresholdResult"]
