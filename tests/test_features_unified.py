"""Regression tests for the unified feature pipeline."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from toptek.features import build_features  # noqa: E402


def _sample_dataframe(rows: int = 128) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    trend = np.linspace(100, 120, rows)
    data = {
        "open": trend + rng.normal(scale=0.5, size=rows),
        "high": trend + rng.uniform(0.5, 1.5, size=rows),
        "low": trend - rng.uniform(0.5, 1.5, size=rows),
        "close": trend + rng.normal(scale=0.3, size=rows),
        "volume": rng.integers(1000, 2000, size=rows),
    }
    return pd.DataFrame(data, index=idx)


def test_build_features_idempotent(tmp_path) -> None:
    df = _sample_dataframe()
    bundle_first = build_features(df, cache_dir=tmp_path)
    bundle_second = build_features(df, cache_dir=tmp_path)

    assert np.array_equal(bundle_first.X, bundle_second.X)
    assert np.array_equal(bundle_first.y, bundle_second.y)
    assert bundle_first.meta["cache_key"] == bundle_second.meta["cache_key"]


def test_psar_warning_suppressed(tmp_path) -> None:
    df = _sample_dataframe()
    with pytest.warns(None) as record:
        build_features(df, cache_dir=tmp_path)
    assert not record.list
