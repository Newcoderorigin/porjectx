"""Unit tests for internal backtest metrics helpers."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from toptek.backtest import run as backtest_run  # noqa: E402


def test_performance_metrics_expected_values() -> None:
    returns = np.array([0.10, -0.05, 0.02, 0.0], dtype=float)
    metrics = backtest_run._performance_metrics(returns)

    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["expectancy"] == pytest.approx(0.0175)
    assert metrics["max_drawdown"] == pytest.approx(0.05)
    assert metrics["sharpe"] == pytest.approx(5.143, rel=1e-3)


def test_apply_policy_filters_and_reports() -> None:
    probs = np.array([0.9, 0.8, 0.4, 0.6], dtype=float)
    returns = np.array([0.05, -0.02, 0.01, 0.03], dtype=float)
    report = backtest_run._apply_policy(probs, returns, tau=0.6)

    assert report["coverage"] == pytest.approx(0.75)
    assert report["threshold"] == pytest.approx(0.6)
    assert report["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert report["expectancy"] == pytest.approx((0.05 - 0.02 + 0.03) / 3)
