"""Rolling performance metrics for live model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import pandas as pd
import sqlite3


@dataclass(frozen=True)
class ContractMetrics:
    """Snapshot of rolling metrics for a single contract."""

    symbol: str
    hit_rate: float
    expectancy: float
    observations: int
    confidence: float
    entry: float
    target: float
    stop: float
    updated: str

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "hit_rate": self.hit_rate,
            "expectancy": self.expectancy,
            "observations": self.observations,
            "confidence": self.confidence,
            "entry": self.entry,
            "target": self.target,
            "stop": self.stop,
            "updated": self.updated,
        }


class RollingMetrics:
    """Compute rolling hit-rate and expectancy for configured contracts."""

    def __init__(self, conn: sqlite3.Connection, *, window: int = 100) -> None:
        self._conn = conn
        self._window = max(1, int(window))

    def snapshot(self, symbols: Sequence[str]) -> Dict[str, ContractMetrics]:
        return {symbol: self.compute(symbol) for symbol in symbols}

    def compute(self, symbol: str) -> ContractMetrics:
        predictions = pd.read_sql_query(
            (
                "SELECT ts, prob_up, realized_hit, realized_return "
                "FROM model_predictions WHERE symbol = ? "
                "ORDER BY ts DESC LIMIT ?"
            ),
            self._conn,
            params=(symbol, self._window),
        )
        predictions["ts"] = pd.to_datetime(predictions["ts"], errors="coerce")
        resolved = predictions.dropna(subset=["realized_hit", "realized_return"])
        observations = int(len(resolved))
        hit_rate = float(resolved["realized_hit"].mean()) if observations else 0.0
        expectancy = float(resolved["realized_return"].mean()) if observations else 0.0
        confidence = float(predictions["prob_up"].iloc[0]) if not predictions.empty else 0.0
        updated = (
            predictions["ts"].iloc[0].isoformat()
            if not predictions.empty and pd.notna(predictions["ts"].iloc[0])
            else ""
        )

        bars = pd.read_sql_query(
            "SELECT ts, close FROM market_bars WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            self._conn,
            params=(symbol,),
        )
        entry = float(bars["close"].iloc[0]) if not bars.empty else 0.0
        if entry and observations:
            target = entry * (1.0 + expectancy)
            stop = entry * (1.0 - abs(expectancy))
        else:
            target = entry
            stop = entry

        return ContractMetrics(
            symbol=symbol,
            hit_rate=hit_rate,
            expectancy=expectancy,
            observations=observations,
            confidence=confidence,
            entry=entry,
            target=target,
            stop=stop,
            updated=updated,
        )


class MetricsAPI:
    """Facade exposing polling-friendly metrics responses."""

    def __init__(self, conn: sqlite3.Connection, *, window: int = 100) -> None:
        self._engine = RollingMetrics(conn, window=window)

    def payload(self, symbols: Sequence[str]) -> Dict[str, Mapping[str, float | str]]:
        snapshots = self._engine.snapshot(symbols)
        return {symbol: metrics.as_dict() for symbol, metrics in snapshots.items()}


__all__ = ["ContractMetrics", "RollingMetrics", "MetricsAPI"]
