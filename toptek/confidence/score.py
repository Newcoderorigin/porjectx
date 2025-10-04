"""Confidence scoring helpers for probability forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class ConfidenceResult:
    probability: float
    confidence: float
    ci_low: float
    ci_high: float
    coverage: float
    expected_value: float


def _wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = phat + z * z / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def score_probabilities(
    probabilities: Iterable[float],
    *,
    method: Literal["fast", "beta"] = "fast",
) -> ConfidenceResult:
    probs = np.clip(np.fromiter(probabilities, dtype=float), 0.0, 1.0)
    if probs.size == 0:
        raise ValueError("probabilities must not be empty")
    p = float(np.mean(probs))
    coverage = float(np.mean(probs >= 0.5))
    ev = float(np.mean(2 * probs - 1))
    if method == "fast":
        conf = abs(p - 0.5) * 2
        low, high = _wilson_interval(int(np.round(p * probs.size)), probs.size)
    elif method == "beta":
        alpha = probs.sum() + 1
        beta_param = probs.size - probs.sum() + 1
        conf = float(abs(p - 0.5) * 2)
        low, high = beta.ppf([0.025, 0.975], alpha, beta_param)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported method: {method}")
    return ConfidenceResult(
        probability=p,
        confidence=conf,
        ci_low=float(low),
        ci_high=float(high),
        coverage=coverage,
        expected_value=ev,
    )


__all__ = ["ConfidenceResult", "score_probabilities"]
