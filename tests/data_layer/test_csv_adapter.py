from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

from toptek.data.adapters import AdapterContext, CSVAdapter
from toptek.data.cache import CacheKey, OHLCVCache


def _write_sample_csv(path: Path) -> Path:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "open": [10, 11, 12, 13],
            "high": [11, 12, 13, 14],
            "low": [9, 10, 11, 12],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1000, 1100, 1200, 1300],
        }
    )
    file_path = path / "ES=F_1d.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_fetch_reads_and_caches(tmp_path: Path) -> None:
    cache = OHLCVCache(tmp_path / "cache")
    context = AdapterContext(cache=cache)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_sample_csv(data_root)
    adapter = CSVAdapter(context, root=data_root)
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = adapter.fetch("ES=F", "1d", start=start)
    assert len(df) == 3
    assert df.index.min() >= start
    key = CacheKey(source="csv", symbol="ES=F", timeframe="1d", start=start, end=None)
    cached = cache.load(key)
    pd.testing.assert_frame_equal(cached, df)


def test_fetch_accepts_naive_bounds(tmp_path: Path) -> None:
    cache = OHLCVCache(tmp_path / "cache")
    context = AdapterContext(cache=cache)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_sample_csv(data_root)
    adapter = CSVAdapter(context, root=data_root)
    start = datetime(2024, 1, 3)
    df = adapter.fetch("ES=F", "1d", start=start)
    assert len(df) == 2
    assert df.index[0].tzinfo is not None
    assert df.index[0] >= start.replace(tzinfo=timezone.utc)
    # second call should hit cache even with naive boundary
    again = adapter.fetch("ES=F", "1d", start=start)
    pd.testing.assert_frame_equal(df, again)


def test_fetch_with_custom_path(tmp_path: Path) -> None:
    cache = OHLCVCache(tmp_path / "cache")
    context = AdapterContext(cache=cache)
    file_path = tmp_path / "custom.csv"
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-02-01", periods=2, freq="D"),
            "open": [1, 2],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 200],
        }
    )
    df.to_csv(file_path, index=False)
    adapter = CSVAdapter(context)
    result = adapter.fetch("ES=F", "1d", params={"path": file_path})
    assert len(result) == 2
    assert result.index.tzinfo is not None
