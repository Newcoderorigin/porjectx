"""Deterministic factories used across the test suite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List


@dataclass
class SyntheticBar:
    """Simple container mirroring the ProjectX bar schema."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_dict(self) -> Dict[str, float | str]:
        """Return a serialisable mapping."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


def synthetic_bars(
    *, count: int = 5, start: datetime | None = None, spacing: timedelta | None = None
) -> List[Dict[str, float | str]]:
    """Return deterministic OHLCV rows suitable for caching tests."""

    origin = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    stride = spacing or timedelta(minutes=5)
    bars: List[Dict[str, float | str]] = []
    price = 4300.0
    for idx in range(count):
        ts = origin + stride * idx
        bar = SyntheticBar(
            timestamp=ts,
            open=price + 0.25,
            high=price + 0.75,
            low=price - 0.50,
            close=price + 0.10,
            volume=1500.0 + idx,
        )
        bars.append(bar.as_dict())
        price += 1.0
    return bars


@dataclass
class StubGateway:
    """Gateway double that records payloads while returning static bars."""

    bars: Iterable[Dict[str, float | str]]

    def __post_init__(self) -> None:
        self.calls: int = 0
        self.payloads: List[Dict[str, object]] = []

    def retrieve_bars(self, payload: Dict[str, object]) -> Dict[str, object]:
        self.calls += 1
        self.payloads.append(dict(payload))
        return {"bars": list(self.bars)}


__all__ = ["synthetic_bars", "StubGateway"]
