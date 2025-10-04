from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from toptek.core import risk  # noqa: E402
from toptek.model import threshold  # noqa: E402


def test_opt_threshold_respects_constraints() -> None:
    probs = np.array([0.95, 0.92, 0.80, 0.60, 0.55])
    labels = np.array([1, 1, 0, 1, 0])
    tau, curve = threshold.opt_threshold(
        probs,
        labels,
        min_coverage=0.20,
        min_expected_value=0.0,
        min_samples=1,
        grid=(0.6, 0.95, 0.05),
    )

    assert tau == pytest.approx(0.9, abs=0.05)
    assert curve
    assert all(point["coverage"] >= 0.20 for point in curve)


def test_opt_threshold_falls_back_to_baseline() -> None:
    probs = np.array([0.51, 0.49, 0.48])
    labels = np.array([0, 0, 0])
    tau, curve = threshold.opt_threshold(
        probs,
        labels,
        min_coverage=0.8,
        min_expected_value=0.5,
        min_samples=5,
        grid=(0.6, 0.9, 0.1),
    )

    assert tau == pytest.approx(0.5)
    assert curve[0]["threshold"] == 0.5


def test_position_size_enforces_rank_constraints() -> None:
    profile = risk.RiskProfile(
        max_position_size=3,
        max_daily_loss=2500,
        restricted_hours=[],
        atr_multiplier_stop=2.0,
        cooldown_losses=2,
        cooldown_minutes=30,
    )
    size = risk.position_size(
        account_balance=100_000,
        risk_profile=profile,
        atr=1.0,
        tick_value=50.0,
        risk_per_trade=0.02,
    )
    assert size == 3

    zero_atr = risk.position_size(
        account_balance=10_000,
        risk_profile=profile,
        atr=0.0,
        tick_value=50.0,
    )
    assert zero_atr == 0
