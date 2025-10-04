from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")
if (
    getattr(pd.DataFrame, "__module__", "builtins") == "builtins"
):  # pragma: no cover - env specific
    pytest.skip(
        "pandas DataFrame unavailable in test environment", allow_module_level=True
    )

from toptek.replay import ReplayBar, ReplaySimulator  # noqa: E402


class VirtualClock:
    """Deterministic clock used to capture simulated sleep durations."""

    def __init__(self) -> None:
        self.now: float = 0.0
        self.history: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.history.append(seconds)
        self.now += seconds

    def time(self) -> float:
        return self.now


def _sample_frame(rows: int) -> pd.DataFrame:
    base = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base + timedelta(seconds=i) for i in range(rows)]
    values = [float(i) for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": [v + 0.5 for v in values],
            "low": [v - 0.5 for v in values],
            "close": [v + 0.25 for v in values],
            "volume": [100 + i for i in range(rows)],
        }
    )


def test_replay_scheduler_speed_adjustment() -> None:
    rows = 600
    frame = _sample_frame(rows)
    clock = VirtualClock()
    simulator = ReplaySimulator(
        frame,
        speed=12.0,
        sleep=clock.sleep,
        clock=clock.time,
    )
    observed: list[ReplayBar] = []
    last_timestamp: datetime | None = None
    expected_elapsed = 0.0
    current_speed = 12.0

    def _listener(bar: ReplayBar) -> None:
        nonlocal last_timestamp, expected_elapsed, current_speed
        if last_timestamp is not None:
            delta = (bar.timestamp - last_timestamp).total_seconds()
            expected_elapsed += delta / current_speed
        last_timestamp = bar.timestamp
        observed.append(bar)
        if bar.index == 199:
            current_speed = 24.0
            simulator.set_speed(current_speed)
        elif bar.index == 399:
            current_speed = 6.0
            simulator.set_speed(current_speed)

    simulator.add_listener(_listener)
    simulator.run()

    assert len(observed) == rows
    assert clock.now == pytest.approx(expected_elapsed, rel=1e-6)
    assert len(clock.history) >= rows - 1
