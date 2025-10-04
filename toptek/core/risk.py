"""Risk management helpers aligned with Topstep guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Iterable

import numpy as np


@dataclass
class RiskProfile:
    """Risk controls for trading sessions."""

    max_position_size: int
    max_daily_loss: float
    restricted_hours: Iterable[Dict[str, str]]
    atr_multiplier_stop: float
    cooldown_losses: int
    cooldown_minutes: int


def can_trade(current_time: datetime, risk_profile: RiskProfile) -> bool:
    """Return ``True`` if trading is allowed at *current_time*."""

    t = current_time.time()
    for window in risk_profile.restricted_hours:
        start = _parse_time(window.get("start", "00:00"))
        end = _parse_time(window.get("end", "00:00"))
        if start <= t <= end:
            return False
    return True


def position_size(
    account_balance: float,
    risk_profile: RiskProfile,
    atr: float,
    tick_value: float,
    *,
    risk_per_trade: float = 0.01,
) -> int:
    """Return an integer contract size respecting risk limits."""

    dollar_risk = account_balance * risk_per_trade
    stop_risk = atr * risk_profile.atr_multiplier_stop * tick_value
    if stop_risk == 0:
        return 0
    size = int(np.floor(dollar_risk / stop_risk))
    return max(0, min(size, risk_profile.max_position_size))


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


__all__ = ["RiskProfile", "can_trade", "position_size"]
