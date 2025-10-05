"""Parquet cache utilities for OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CacheKey:
    """Uniquely identifies a cached OHLCV slice."""

    source: str
    symbol: str
    timeframe: str
    start: datetime | None = None
    end: datetime | None = None

    def to_parts(self) -> Iterable[str]:
        start_part = self.start.isoformat().replace(":", "-") if self.start else "none"
        end_part = self.end.isoformat().replace(":", "-") if self.end else "none"
        return (self.source, self.symbol, self.timeframe, start_part, end_part)

    def to_filename(self) -> str:
        parts = "__".join(self.to_parts())
        return f"{parts}.parquet"


class OHLCVCache:
    """Filesystem-backed cache storing OHLCV frames as parquet."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: CacheKey) -> Path:
        return self._root / key.to_filename()

    def _resolve_pickle(self, key: CacheKey) -> Path:
        return self._root / key.to_filename().replace(".parquet", ".pickle")

    def load(self, key: CacheKey) -> pd.DataFrame | None:
        path = self._resolve(key)
        pickle_path = self._resolve_pickle(key)
        if path.exists():
            df = pd.read_parquet(path)
        elif pickle_path.exists():
            df = pd.read_pickle(pickle_path)
        else:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Cached frame must use DatetimeIndex")
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()

    def store(self, key: CacheKey, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot cache empty dataframe")
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_to_store = df.copy()
        if not isinstance(df_to_store.index, pd.DatetimeIndex):
            raise TypeError("OHLCV dataframe must have DatetimeIndex")
        if df_to_store.index.tzinfo is None:
            df_to_store.index = df_to_store.index.tz_localize("UTC")
        else:
            df_to_store.index = df_to_store.index.tz_convert("UTC")
        df_to_store.sort_index(inplace=True)
        try:
            df_to_store.to_parquet(path)
        except (ImportError, ValueError):
            pickle_path = self._resolve_pickle(key)
            df_to_store.to_pickle(pickle_path)


__all__ = ["CacheKey", "OHLCVCache"]
