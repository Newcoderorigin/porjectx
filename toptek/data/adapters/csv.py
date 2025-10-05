"""CSV-backed OHLCV adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from .base import AdapterContext, OHLCVDataAdapter, ensure_ohlcv_schema


class CSVAdapter(OHLCVDataAdapter):
    """Load OHLCV data from local CSV files."""

    def __init__(self, context: AdapterContext | None = None, *, root: str | Path | None = None) -> None:
        super().__init__(context)
        self._root = Path(root) if root is not None else None

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        source: str = "csv",
        start: datetime | None = None,
        end: datetime | None = None,
        params: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        cached = self._lookup_cache(
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            start=start,
            end=end,
        )
        if cached is not None:
            return cached

        path = self._resolve_path(symbol, timeframe, params)
        df = pd.read_csv(
            path,
            parse_dates=["ts"],
            index_col="ts",
        )
        df = ensure_ohlcv_schema(df)
        if start is not None:
            df = df.loc[df.index >= start]
        if end is not None:
            df = df.loc[df.index <= end]
        self._maybe_cache(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            start=start,
            end=end,
        )
        return df

    def _resolve_path(
        self,
        symbol: str,
        timeframe: str,
        params: Mapping[str, object] | None,
    ) -> Path:
        if params and "path" in params:
            return Path(str(params["path"]))
        if self._root is None:
            raise FileNotFoundError("CSV root directory not configured")
        filename = f"{symbol}_{timeframe}.csv"
        candidate = self._root / filename
        if not candidate.exists():
            raise FileNotFoundError(f"CSV file not found: {candidate}")
        return candidate


__all__ = ["CSVAdapter"]
