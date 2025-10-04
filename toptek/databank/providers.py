"""Market data providers for the offline-first data bank."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yf = None  # type: ignore


class BarsProvider(Protocol):
    """Protocol defining a market data source."""

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Retrieve OHLCV bars for *symbol* between *start* and *end* inclusive."""


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    unit = timeframe[-1].lower()
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Invalid timeframe: {timeframe}") from exc
    if value <= 0:
        raise ValueError(f"Timeframe must be positive: {timeframe}")
    mapping = {
        "s": timedelta(seconds=value),
        "m": timedelta(minutes=value),
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
    }
    if unit not in mapping:
        raise ValueError(f"Unsupported timeframe unit: {timeframe}")
    return mapping[unit]


@dataclass(slots=True)
class SyntheticBars:
    """Deterministic geometric Brownian motion provider for offline workflows."""

    seed: int | None = None

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        step = _timeframe_to_timedelta(timeframe)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end < start:
            raise ValueError("End must be after start")
        index = pd.date_range(start=start, end=end, freq=step)
        if index.empty:
            index = pd.DatetimeIndex([start, end])
        base_seed = self.seed
        if base_seed is None:
            digest = hashlib.sha256(f"{symbol}:{timeframe}".encode("utf-8")).digest()
            base_seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(base_seed)
        drift = 0.0002
        volatility = 0.01
        prices = np.empty(len(index), dtype=np.float64)
        prices[0] = 100.0 + (hash(symbol) % 100) * 0.1
        for i in range(1, len(index)):
            shock = rng.normal(drift, volatility)
            prices[i] = prices[i - 1] * math.exp(shock)
        highs = prices * (1 + rng.uniform(0.0005, 0.002, size=len(index)))
        lows = prices * (1 - rng.uniform(0.0005, 0.002, size=len(index)))
        opens = np.concatenate(([prices[0]], prices[:-1]))
        volumes = rng.integers(1_000, 10_000, size=len(index))
        df = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum.reduce([opens, highs, prices]),
                "low": np.minimum.reduce([opens, lows, prices]),
                "close": prices,
                "volume": volumes.astype(np.int64),
            },
            index=index,
        )
        df.index.name = "timestamp"
        return df


@dataclass(slots=True)
class YFinanceBars:
    """Yahoo Finance powered provider with graceful fallback when unavailable."""

    auto_adjust: bool = True

    def __post_init__(self) -> None:
        if yf is None:  # pragma: no cover - runtime guard
            raise ImportError("yfinance is not installed; SyntheticBars is the default")

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        interval = timeframe
        data = yf.download(  # type: ignore[attr-defined]
            symbol,
            start=start,
            end=end + timedelta(days=1),
            interval=interval,
            progress=False,
            auto_adjust=self.auto_adjust,
        )
        if data.empty:
            raise RuntimeError(f"No data returned from yfinance for {symbol}")
        data.index = data.index.tz_localize(timezone.utc)
        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "close",
                "Volume": "volume",
            }
        )
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(data.columns):
            raise RuntimeError("Unexpected columns from yfinance download")
        return data[list(required)].sort_index()


__all__ = ["BarsProvider", "SyntheticBars", "YFinanceBars"]
