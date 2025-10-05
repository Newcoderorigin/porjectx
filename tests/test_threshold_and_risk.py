from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
try:  # pragma: no cover - optional dependency gate
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover - missing hypothesis
    pytest.skip("hypothesis is required for property tests", allow_module_level=True)

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


@settings(max_examples=75, deadline=None, seed=1337)
@given(
    balance=st.floats(
        min_value=10_000.0,
        max_value=500_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    balance_scale=st.floats(
        min_value=1.0,
        max_value=3.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    atr=st.floats(
        min_value=0.25,
        max_value=15.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    tick=st.floats(
        min_value=0.25,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    risk_fraction=st.floats(
        min_value=0.001,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    risk_scale=st.floats(
        min_value=1.0,
        max_value=4.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_position_size_monotonic_in_balance_and_risk(
    balance: float,
    balance_scale: float,
    atr: float,
    tick: float,
    risk_fraction: float,
    risk_scale: float,
) -> None:
    profile = risk.RiskProfile(
        max_position_size=15,
        max_daily_loss=5_000.0,
        restricted_hours=[],
        atr_multiplier_stop=2.5,
        cooldown_losses=3,
        cooldown_minutes=15,
    )
    bigger_balance = balance * balance_scale
    higher_risk = min(risk_fraction * risk_scale, 0.25)

    base = risk.position_size(
        account_balance=balance,
        risk_profile=profile,
        atr=atr,
        tick_value=tick,
        risk_per_trade=risk_fraction,
    )
    scaled_balance = risk.position_size(
        account_balance=bigger_balance,
        risk_profile=profile,
        atr=atr,
        tick_value=tick,
        risk_per_trade=risk_fraction,
    )
    scaled_risk = risk.position_size(
        account_balance=balance,
        risk_profile=profile,
        atr=atr,
        tick_value=tick,
        risk_per_trade=higher_risk,
    )

    assert scaled_balance >= base
    assert scaled_risk >= base


@settings(max_examples=75, deadline=None, seed=2024)
@given(
    balance=st.floats(
        min_value=25_000.0,
        max_value=250_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    atr_values=st.tuples(
        st.floats(
            min_value=0.5,
            max_value=8.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.floats(
            min_value=0.5,
            max_value=8.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    tick=st.floats(
        min_value=0.5,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    risk_fraction=st.floats(
        min_value=0.005,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_position_size_decreases_with_higher_atr(
    balance: float,
    atr_values: tuple[float, float],
    tick: float,
    risk_fraction: float,
) -> None:
    low_atr, high_atr = sorted(atr_values)
    if low_atr == high_atr:
        high_atr = low_atr * 1.1
    profile = risk.RiskProfile(
        max_position_size=20,
        max_daily_loss=10_000.0,
        restricted_hours=[],
        atr_multiplier_stop=2.0,
        cooldown_losses=2,
        cooldown_minutes=10,
    )

    low = risk.position_size(
        account_balance=balance,
        risk_profile=profile,
        atr=low_atr,
        tick_value=tick,
        risk_per_trade=risk_fraction,
    )
    high = risk.position_size(
        account_balance=balance,
        risk_profile=profile,
        atr=high_atr,
        tick_value=tick,
        risk_per_trade=risk_fraction,
    )

    assert high <= low
