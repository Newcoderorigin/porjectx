"""Base interfaces for OHLCV data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd

from toptek.data.cache import CacheKey, OHLCVCache


@dataclass(frozen=True)
class AdapterContext:
    """Configuration shared across adapters."""

    cache: OHLCVCache
    options: Mapping[str, str] | None = None


class OHLCVDataAdapter(ABC):
    """Abstract adapter that returns OHLCV price data."""

    def __init__(self, context: AdapterContext | None = None) -> None:
        self._context = context

    @property
    def context(self) -> AdapterContext | None:
        return self._context

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        source: str,
        start: datetime | None = None,
        end: datetime | None = None,
        params: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV bars indexed by UTC timestamps."""

    def _maybe_cache(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str,
        start: datetime | None,
        end: datetime | None,
    ) -> None:
        if self._context is None:
            return
        cache = self._context.cache
        start_utc = self._normalize_boundary(start)
        end_utc = self._normalize_boundary(end)
        key = CacheKey(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start=start_utc,
            end=end_utc,
        )
        cache.store(key, df)

    def _lookup_cache(
        self,
        *,
        symbol: str,
        timeframe: str,
        source: str,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame | None:
        if self._context is None:
            return None
        cache = self._context.cache
        start_utc = self._normalize_boundary(start)
        end_utc = self._normalize_boundary(end)
        key = CacheKey(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start=start_utc,
            end=end_utc,
        )
        return cache.load(key)

    @staticmethod
    def _normalize_boundary(bound: datetime | None) -> datetime | None:
        """Return a UTC-aware datetime for cache keys and filtering."""

        if bound is None:
            return None
        if bound.tzinfo is None:
            return bound.replace(tzinfo=timezone.utc)
        return bound.astimezone(timezone.utc)


def ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV schema and return a copy sorted by timestamp."""

    required_columns = ("open", "high", "low", "close", "volume")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataframe index must be a DatetimeIndex")

    if df.index.tzinfo is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")

    return df.sort_index()


__all__ = [
    "AdapterContext",
    "OHLCVDataAdapter",
    "ensure_ohlcv_schema",
]
