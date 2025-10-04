"""Advisor data providers for research mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yf = None  # type: ignore


class AdvisorProvider(Protocol):
    """Interface for fetching advisor context."""

    def quotes(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def headlines(self, symbol: str) -> Iterable[str]:
        raise NotImplementedError


@dataclass(slots=True)
class SyntheticAdvisorProvider:
    seed: int | None = None

    def quotes(self, symbol: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed or abs(hash(symbol)) % (2**32))
        base_price = 100 + (abs(hash(symbol)) % 50)
        idx = pd.date_range(end=datetime.now(timezone.utc), periods=20, freq="1H")
        prices = base_price * np.cumprod(1 + rng.normal(0, 0.002, size=len(idx)))
        return pd.DataFrame({"close": prices}, index=idx)

    def headlines(self, symbol: str) -> Iterable[str]:
        topics = [
            f"{symbol} consolidates ahead of earnings",
            f"Analysts eye upside skew for {symbol}",
            f"{symbol} implied vol signals event risk",
            f"Macro flows shift toward {symbol} sector leaders",
        ]
        return topics[:3]


@dataclass(slots=True)
class YFinanceAdvisorProvider:
    def __post_init__(self) -> None:
        if yf is None:  # pragma: no cover - runtime guard
            raise ImportError("yfinance is required for YFinanceAdvisorProvider")

    def quotes(self, symbol: str) -> pd.DataFrame:
        data = yf.download(symbol, period="5d", interval="1h", progress=False)  # type: ignore[attr-defined]
        if data.empty:
            raise RuntimeError("No advisor quotes returned")
        return data[["Close"]].rename(columns={"Close": "close"})

    def headlines(self, symbol: str) -> Iterable[str]:
        return [
            f"{symbol} sentiment steady per options skew",
            f"{symbol} desk highlights tactical range trades",
            f"{symbol} liquidity remains supportive",
        ]


__all__ = [
    "AdvisorProvider",
    "SyntheticAdvisorProvider",
    "YFinanceAdvisorProvider",
]
