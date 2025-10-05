"""Indicator regression tests for ``toptek.core.features``."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from toptek.core import features as core_features  # noqa: E402


def _sample_dataframe(rows: int = 256) -> pd.DataFrame:
    rng = np.random.default_rng(202401)
    idx = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    trend = np.linspace(100.0, 120.0, rows)
    data = {
        "open": trend + rng.normal(scale=0.3, size=rows),
        "high": trend + rng.uniform(0.5, 1.5, size=rows),
        "low": trend - rng.uniform(0.5, 1.5, size=rows),
        "close": trend + rng.normal(scale=0.25, size=rows),
        "volume": rng.integers(1_000, 2_000, size=rows),
    }
    return pd.DataFrame(data, index=idx)


def test_compute_features_produces_expected_columns() -> None:
    df = _sample_dataframe()
    result = core_features.compute_features(df)

    expected = {
        "sma_10",
        "ema_26",
        "ema_200",
        "macd",
        "macd_signal",
        "rsi_14",
        "bb_high",
        "bb_low",
        "donchian_width",
        "adx_14",
        "di_plus",
        "di_minus",
        "psar",
        "return_20",
        "volatility_parkinson",
        "volume_zscore",
    }

    assert expected.issubset(result.keys())
    for name, values in result.items():
        assert values.shape == (len(df),), f"Unexpected shape for {name}"


def test_indicator_outputs_are_finite_post_warmup() -> None:
    df = _sample_dataframe()
    result = core_features.compute_features(df)

    warmup = 210
    for name, values in result.items():
        tail = values[warmup:]
        if not len(tail):
            continue
        assert np.isfinite(tail).all(), f"Non-finite values detected in {name}"


def test_donchian_width_matches_band_difference() -> None:
    df = _sample_dataframe()
    result = core_features.compute_features(df)
    high = result["donchian_high"]
    low = result["donchian_low"]
    width = result["donchian_width"]

    assert np.allclose(width, high - low, equal_nan=True)
