from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from toptek.rank import RankRequest, rank_strategies


def test_rank_strategies_filters_constraints() -> None:
    index = pd.RangeIndex(90)
    signals = pd.DataFrame(
        {
            "strategy_a": np.linspace(-0.1, 0.2, len(index)),
            "strategy_b": np.linspace(0.05, 0.15, len(index)),
        },
        index=index,
    )
    result = rank_strategies(
        RankRequest(
            signals=signals, min_coverage=0.2, min_ev=0.01, max_drawdown=0.5, folds=3
        )
    )
    assert result.scores
    names = [score.name for score in result.scores]
    assert "strategy_b" in names
