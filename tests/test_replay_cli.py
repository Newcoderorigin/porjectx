from __future__ import annotations

import json
import sys
import types
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "pandas" not in sys.modules:
    def _unsupported(*_: object, **__: object) -> None:
        raise RuntimeError("pandas stub used unexpectedly")

    pandas_stub = types.SimpleNamespace(
        Timestamp=object,
        Timedelta=object,
        isna=lambda value: False,
        to_datetime=lambda values, utc=True: values,
        DataFrame=object,
        read_csv=_unsupported,
        read_parquet=_unsupported,
    )
    sys.modules["pandas"] = pandas_stub  # type: ignore[assignment]

if "yaml" not in sys.modules:
    yaml_stub = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
    sys.modules["yaml"] = yaml_stub  # type: ignore[assignment]

from toptek.replay import sim
from toptek.replay.sim import ReplayBar


class DummySimulator:
    def __init__(self, bars: list[ReplayBar]) -> None:
        self._bars = bars
        self._bar_count = len(bars)
        self._listeners: list = []
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._running

    def add_listener(self, listener) -> None:
        self._listeners.append(listener)

    def start(self) -> None:
        def _worker() -> None:
            self._running = True
            for bar in self._bars:
                time.sleep(0.01)
                for listener in list(self._listeners):
                    listener(bar)
            self._running = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._running = False

    @property
    def bar_count(self) -> int:
        return self._bar_count


@pytest.fixture()
def dummy_replay(monkeypatch) -> DummySimulator:
    bars = [
        ReplayBar(index=0, timestamp=datetime(2024, 1, 1, 0, 0, 0), data={"open": 1.0}),
        ReplayBar(index=1, timestamp=datetime(2024, 1, 1, 0, 0, 1), data={"open": 2.0}),
    ]
    simulator = DummySimulator(bars)

    def _from_path(cls, path: str | Path, **_: object) -> DummySimulator:  # type: ignore[override]
        return simulator

    monkeypatch.setattr(sim.ReplaySimulator, "from_path", classmethod(_from_path))
    return simulator


def test_main_replays_and_prints_bars(dummy_replay: DummySimulator, capsys) -> None:
    assert dummy_replay.running is False
    exit_code = sim.main(["dummy-path.csv"])

    dummy_replay.stop()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]

    assert exit_code == 0
    assert len(lines) == dummy_replay.bar_count
    parsed = [json.loads(line) for line in lines]
    assert {item["open"] for item in parsed} == {1.0, 2.0}
