"""CCXT adapter stub."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

import pandas as pd

from .base import AdapterContext, OHLCVDataAdapter


class CCXTAdapter(OHLCVDataAdapter):
    """Load OHLCV data via CCXT.

    This stub supports cache lookups and defers the live exchange call to a
    follow-up layer.
    """

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        source: str = "ccxt",
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
        raise NotImplementedError("CCXT network fetch not yet implemented")


__all__ = ["CCXTAdapter"]
