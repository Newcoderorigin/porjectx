"""Latency badge tests covering severity transitions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toptek.monitor import LatencyBadge, Severity, build_latency_badge


def _now() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_latency_badge_live_severity():
    badge = build_latency_badge(
        _now() - timedelta(seconds=10),
        now=_now(),
        warning_threshold=30,
        alert_threshold=90,
    )

    assert isinstance(badge, LatencyBadge)
    assert badge.severity == Severity.STABLE
    assert badge.label == "Live"
    assert "healthy" in badge.message


def test_latency_badge_watch_severity():
    badge = build_latency_badge(
        _now() - timedelta(seconds=45),
        now=_now(),
        warning_threshold=30,
        alert_threshold=90,
    )

    assert badge.severity == Severity.WATCH
    assert badge.label == "Lagging"
    assert badge.latency_seconds == pytest.approx(45.0)


def test_latency_badge_alert_severity():
    badge = build_latency_badge(
        _now() - timedelta(seconds=120),
        now=_now(),
        warning_threshold=30,
        alert_threshold=90,
    )

    assert badge.severity == Severity.ALERT
    assert badge.label == "Stalled"
    assert "stalled" in badge.message.lower()


def test_latency_badge_unknown_without_timestamp():
    badge = build_latency_badge(None, now=_now())

    assert badge.severity == Severity.UNKNOWN
    assert badge.label == "No signal"
    assert badge.message == "No bars received yet."


def test_latency_badge_validates_threshold_order():
    with pytest.raises(ValueError):
        build_latency_badge(
            _now(), now=_now(), warning_threshold=90, alert_threshold=30
        )

    with pytest.raises(ValueError):
        build_latency_badge(
            _now(), now=_now(), warning_threshold=-1, alert_threshold=30
        )
