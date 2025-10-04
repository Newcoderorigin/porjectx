"""Data retrieval and local caching helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .gateway import ProjectXGateway
from .utils import build_logger


logger = build_logger(__name__)


def _cache_file(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return cache_dir / f"{safe_symbol}_{timeframe}.json"


def load_cached_bars(
    cache_dir: Path, symbol: str, timeframe: str
) -> List[Dict[str, Any]]:
    """Load cached bar data if available."""

    path = _cache_file(cache_dir, symbol, timeframe)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cached_bars(
    cache_dir: Path, symbol: str, timeframe: str, bars: Iterable[Dict[str, Any]]
) -> None:
    """Persist bar data to disk for reuse."""

    path = _cache_file(cache_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(bars), handle)


def fetch_bars(
    gateway: ProjectXGateway,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
) -> List[Dict[str, Any]]:
    """Fetch bars from ProjectX or local cache."""

    cached = load_cached_bars(cache_dir, symbol, timeframe)
    if cached:
        return cached
    payload = {
        "contractSymbol": symbol,
        "timeFrame": timeframe,
        "startTime": start.isoformat(),
        "endTime": end.isoformat(),
    }
    response = gateway.retrieve_bars(payload)
    bars = response.get("bars", [])
    save_cached_bars(cache_dir, symbol, timeframe, bars)
    return bars


def resample_ohlc(bars: List[Dict[str, Any]], *, field: str = "close") -> np.ndarray:
    """Return a numpy array of a given bar field."""

    return np.array([float(bar.get(field, 0.0)) for bar in bars], dtype=float)


__all__ = [
    "fetch_bars",
    "resample_ohlc",
    "load_cached_bars",
    "save_cached_bars",
    "sample_dataframe",
]


def sample_dataframe(rows: int = 500) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for offline workflows."""

    index = pd.date_range(end=datetime.utcnow(), periods=rows, freq="5min")
    base = np.cumsum(np.random.randn(rows)) + 4500
    high = base + np.random.rand(rows) * 2
    low = base - np.random.rand(rows) * 2
    close = base + np.random.randn(rows) * 0.5
    open_ = close + np.random.randn(rows) * 0.3
    volume = np.random.randint(100, 1000, size=rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
