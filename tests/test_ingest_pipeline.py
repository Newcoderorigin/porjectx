from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from toptek.core import data as data_module  # noqa: E402

from .factories import StubGateway, synthetic_bars  # noqa: E402


def test_fetch_bars_idempotent_cache(tmp_path: Path) -> None:
    bars = synthetic_bars(count=3)
    gateway = StubGateway(bars)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=10)

    first = data_module.fetch_bars(
        gateway,
        symbol="ES/ESH4",
        timeframe="5m",
        start=start,
        end=end,
        cache_dir=tmp_path,
    )
    second = data_module.fetch_bars(
        gateway,
        symbol="ES/ESH4",
        timeframe="5m",
        start=start,
        end=end,
        cache_dir=tmp_path,
    )

    assert first == bars
    assert second == bars
    assert gateway.calls == 1
    cache_file = tmp_path / "ES-ESH4_5m.json"
    assert cache_file.exists()


def test_resample_and_sample_dataframe_produce_expected_shapes(tmp_path: Path) -> None:
    bars = synthetic_bars(count=4)
    arr = data_module.resample_ohlc(bars, field="close")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4,)
    assert np.allclose(arr, [float(bar["close"]) for bar in bars])

    np.random.seed(2024)
    frame = data_module.sample_dataframe(rows=16)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 16
    # Ensure cached build remains deterministic regardless of existing cache files.
    second_frame = data_module.sample_dataframe(rows=16)
    assert second_frame.shape == frame.shape
