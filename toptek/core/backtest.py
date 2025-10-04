"""Vectorised backtesting utilities for evaluating strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    """Summary statistics for a backtest run."""

    hit_rate: float
    sharpe: float
    max_drawdown: float
    expectancy: float
    equity_curve: np.ndarray


def run_backtest(
    returns: np.ndarray, signals: np.ndarray, *, fee_per_trade: float = 0.0
) -> BacktestResult:
    """Run a simple long/flat backtest."""

    trade_returns = returns * signals - fee_per_trade
    equity_curve = np.cumsum(trade_returns)
    wins = trade_returns > 0
    hit_rate = float(wins.mean()) if len(trade_returns) else 0.0
    sharpe = float(
        np.mean(trade_returns) / (np.std(trade_returns) + 1e-9) * np.sqrt(252)
    )
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0
    expectancy = float(np.mean(trade_returns))
    return BacktestResult(
        hit_rate=hit_rate,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        expectancy=expectancy,
        equity_curve=equity_curve,
    )


__all__ = ["run_backtest", "BacktestResult"]
