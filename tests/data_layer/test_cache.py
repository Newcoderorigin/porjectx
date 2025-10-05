from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pathlib import Path

pytest.importorskip("pandas")
import pandas as pd

from toptek.data.cache import CacheKey, OHLCVCache


def _make_frame() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [100, 200, 300],
        },
        index=idx,
    )


def test_store_and_load_roundtrip(tmp_path: Path) -> None:
    cache = OHLCVCache(tmp_path)
    key = CacheKey(
        source="csv",
        symbol="ES=F",
        timeframe="1d",
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end=datetime(2023, 1, 3, tzinfo=timezone.utc),
    )
    df = _make_frame()
    cache.store(key, df)
    loaded = cache.load(key)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_empty_frame_rejected(tmp_path: Path) -> None:
    cache = OHLCVCache(tmp_path)
    key = CacheKey(source="csv", symbol="ES=F", timeframe="1d")
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    try:
        cache.store(key, empty)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty frame")
