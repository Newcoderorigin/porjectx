"""Walk-forward ranking engine enforcing risk constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankRequest:
    signals: pd.DataFrame
    min_coverage: float = 0.1
    min_ev: float = 0.0
    max_drawdown: float = 0.2
    folds: int = 3


@dataclass(frozen=True)
class StrategyScore:
    name: str
    hit_rate: float
    coverage: float
    expected_value: float
    max_drawdown: float


@dataclass(frozen=True)
class RankResult:
    scores: List[StrategyScore]

    def to_json(self, path: Path) -> None:
        path.write_text(
            pd.DataFrame([s.__dict__ for s in self.scores]).to_json(
                orient="records", indent=2
            ),
            encoding="utf-8",
        )

    def to_csv(self, path: Path) -> None:
        pd.DataFrame([s.__dict__ for s in self.scores]).to_csv(path, index=False)


def _max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / peaks
    return float(np.max(drawdowns))


def _walk_forward_chunks(length: int, folds: int) -> Iterable[tuple[int, int]]:
    fold_size = length // folds
    for i in range(folds):
        start = i * fold_size
        end = length if i == folds - 1 else (i + 1) * fold_size
        yield start, end


def rank_strategies(request: RankRequest) -> RankResult:
    if request.folds < 2:
        raise ValueError("folds must be >=2")
    df = request.signals.copy()
    if df.empty:
        raise ValueError("signals frame must not be empty")
    scores: list[StrategyScore] = []
    for column in df.columns:
        series = df[column].astype(float)
        pnl: list[float] = []
        hits: list[float] = []
        coverages: list[float] = []
        for start, end in _walk_forward_chunks(len(series), request.folds):
            window = series.iloc[start:end]
            if window.empty:
                continue
            decisions = window >= 0.0
            coverage = float(decisions.mean())
            if coverage < request.min_coverage:
                continue
            returns = window.where(decisions, 0.0)
            pnl.append(float(returns.mean()))
            hits.append(float((returns > 0).mean()))
            coverages.append(coverage)
        if not pnl:
            continue
        equity = np.cumsum(pnl)
        max_dd = _max_drawdown(np.array(equity) + 1.0)
        hit_rate = float(np.mean(hits))
        coverage = float(np.mean(coverages))
        expected_value = float(np.mean(pnl))
        if (
            coverage < request.min_coverage
            or expected_value < request.min_ev
            or max_dd > request.max_drawdown
        ):
            continue
        scores.append(
            StrategyScore(
                name=column,
                hit_rate=hit_rate,
                coverage=coverage,
                expected_value=expected_value,
                max_drawdown=max_dd,
            )
        )
    scores.sort(key=lambda s: (s.expected_value, s.hit_rate), reverse=True)
    return RankResult(scores)


__all__ = ["RankRequest", "RankResult", "rank_strategies", "StrategyScore"]
