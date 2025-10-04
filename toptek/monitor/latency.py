"""Latency utilities for rendering last-bar freshness badges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .drift import Severity


@dataclass(frozen=True)
class LatencyBadge:
    """Badge metadata describing feed latency for UI surfaces."""

    severity: Severity
    latency_seconds: float
    label: str
    message: str


def build_latency_badge(
    last_bar_timestamp: Optional[datetime],
    *,
    now: Optional[datetime] = None,
    warning_threshold: float = 30.0,
    alert_threshold: float = 90.0,
) -> LatencyBadge:
    """Compute a latency badge for the latest market bar.

    Parameters
    ----------
    last_bar_timestamp:
        Timestamp of the last ingested bar. When ``None`` the badge will fall
        back to ``UNKNOWN`` severity.
    now:
        Optional override for the current timestamp. Defaults to
        ``datetime.now(timezone.utc)`` to keep the computation deterministic
        during testing.
    warning_threshold:
        Boundary in seconds where the badge escalates to ``WATCH`` severity.
    alert_threshold:
        Boundary in seconds where the badge escalates to ``ALERT`` severity.

    Returns
    -------
    LatencyBadge
        Structured badge metadata with severity and copy for the UI layer.
    """

    if warning_threshold <= 0 or alert_threshold <= 0:
        raise ValueError("Thresholds must be positive numbers.")
    if alert_threshold <= warning_threshold:
        raise ValueError("`alert_threshold` must exceed `warning_threshold`.")

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if last_bar_timestamp is None:
        return LatencyBadge(
            severity=Severity.UNKNOWN,
            latency_seconds=float("nan"),
            label="No signal",
            message="No bars received yet.",
        )

    if last_bar_timestamp.tzinfo is None:
        last_bar_timestamp = last_bar_timestamp.replace(tzinfo=timezone.utc)

    latency_seconds = (now - last_bar_timestamp).total_seconds()
    latency_seconds = max(latency_seconds, 0.0)

    if latency_seconds < warning_threshold:
        severity = Severity.STABLE
        label = "Live"
        message = f"Feed healthy ({latency_seconds:.0f}s latency)."
    elif latency_seconds < alert_threshold:
        severity = Severity.WATCH
        label = "Lagging"
        message = f"Feed delayed ({latency_seconds:.0f}s latency)."
    else:
        severity = Severity.ALERT
        label = "Stalled"
        message = f"Feed stalled ({latency_seconds:.0f}s latency)."

    return LatencyBadge(
        severity=severity,
        latency_seconds=latency_seconds,
        label=label,
        message=message,
    )
