"""Alpha Vantage adapter stub."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

import pandas as pd

from .base import AdapterContext, OHLCVDataAdapter


class AlphaVantageAdapter(OHLCVDataAdapter):
    """Fetch OHLCV data from the Alpha Vantage API.

    The concrete HTTP integration will be implemented in a later layer; this
    stub validates parameters and consults the cache when available.
    """

    def __init__(self, context: AdapterContext | None = None) -> None:
        super().__init__(context)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        source: str = "alpha_vantage",
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
        raise NotImplementedError("Alpha Vantage network fetch not yet implemented")


__all__ = ["AlphaVantageAdapter"]
