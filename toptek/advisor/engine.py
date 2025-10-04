"""Advisor engine generating contextual guidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .providers import AdvisorProvider, SyntheticAdvisorProvider


@dataclass(frozen=True)
class AdvisorResponse:
    symbol: str
    risk_bucket: str
    atr_percent: float
    bullets: tuple[str, str, str]
    recommendation: str


class AdvisorEngine:
    def __init__(self, provider: AdvisorProvider | None = None) -> None:
        self._provider = provider or SyntheticAdvisorProvider()

    def advise(self, symbol: str) -> AdvisorResponse:
        quotes = self._provider.quotes(symbol)
        if quotes.empty:
            raise RuntimeError("No advisor quotes available")
        closes = quotes["close"].astype(float)
        atr = float(np.abs(np.diff(closes)).mean()) if len(closes) > 1 else 0.0
        atr_percent = float(atr / closes.iloc[-1]) if closes.iloc[-1] else 0.0
        risk_bucket = self._risk_bucket(atr_percent)
        bullets = tuple(self._provider.headlines(symbol))
        if len(bullets) < 3:
            bullets = bullets + ("Liquidity normalising",) * (3 - len(bullets))
        bullet_triplet = (bullets[0], bullets[1], bullets[2])
        recommendation = self._recommendation(risk_bucket, atr_percent)
        return AdvisorResponse(
            symbol=symbol.upper(),
            risk_bucket=risk_bucket,
            atr_percent=atr_percent,
            bullets=bullet_triplet,
            recommendation=recommendation,
        )

    @staticmethod
    def _risk_bucket(atr_percent: float) -> str:
        if atr_percent < 0.01:
            return "Low"
        if atr_percent < 0.02:
            return "Moderate"
        return "High"

    @staticmethod
    def _recommendation(bucket: str, atr_percent: float) -> str:
        if bucket == "Low":
            return "Consider scaling into the trend with tight risk."
        if bucket == "Moderate":
            return "Deploy balanced positioning; watch catalysts."
        return "Trade defensive; volatility regime elevated."


__all__ = ["AdvisorEngine", "AdvisorResponse"]
