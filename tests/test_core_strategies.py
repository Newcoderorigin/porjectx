"""Strategy regression tests for the GUI sample backtest."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from toptek.core import backtest  # noqa: E402
from toptek.core.data import sample_dataframe  # noqa: E402


@pytest.fixture(scope="module")
def sample_returns() -> np.ndarray:
    np.random.seed(1337)
    df = sample_dataframe(480)
    return np.log(df["close"]).diff().fillna(0).to_numpy()


def _strategy_result(returns: np.ndarray, kind: str) -> backtest.BacktestResult:
    if kind == "momentum":
        signals = (returns > 0).astype(int)
    elif kind == "mean_reversion":
        signals = (returns < 0).astype(int)
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown strategy: {kind}")
    return backtest.run_backtest(returns, signals)


def test_momentum_strategy_remains_profitable(sample_returns: np.ndarray) -> None:
    result = _strategy_result(sample_returns, "momentum")
    assert result.expectancy >= 0.0
    assert result.hit_rate >= 0.45
    assert result.max_drawdown >= 0.0


def test_mean_reversion_strategy_is_conservative(sample_returns: np.ndarray) -> None:
    momentum = _strategy_result(sample_returns, "momentum")
    mean_rev = _strategy_result(sample_returns, "mean_reversion")

    assert mean_rev.expectancy <= 0.0
    assert mean_rev.hit_rate <= momentum.hit_rate
    assert mean_rev.max_drawdown >= momentum.max_drawdown
