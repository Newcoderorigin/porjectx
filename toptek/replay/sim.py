"""Offline bar replay simulator with adjustable playback controls."""

from __future__ import annotations

import argparse
import json
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from toptek.core import utils

ClockFn = Callable[[], float]
SleepFn = Callable[[float], None]
Listener = Callable[["ReplayBar"], None]


@dataclass(frozen=True)
class ReplayBar:
    """Immutable container describing a replayed bar."""

    index: int
    timestamp: datetime
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable mapping containing the bar payload."""

        payload = dict(self.data)
        payload["timestamp"] = self.timestamp
        payload["index"] = self.index
        return payload


def detect_format(path: Path, preferred: str | None = None) -> str:
    """Return the dataset format inferred from *path* or *preferred*."""

    if preferred and preferred.lower() != "auto":
        normalised = preferred.lower()
        if normalised not in {"csv", "parquet"}:
            raise ValueError(f"Unsupported format override: {preferred}")
        return normalised
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return "csv"
    if suffix in {".pq", ".parquet"}:
        return "parquet"
    raise ValueError(
        "Unable to infer dataset format; pass --format csv|parquet explicitly"
    )


def _coerce_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (pd.Timedelta,)):
        return value.to_pytimedelta()
    if pd.isna(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        try:
            return value.item()
        except ValueError:
            pass
    return value


class ReplaySimulator:
    """Schedule and emit historical bars at configurable playback speeds."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        time_column: str = "timestamp",
        speed: float = 1.0,
        sleep: SleepFn | None = None,
        clock: ClockFn | None = None,
        max_step: float = 0.5,
    ) -> None:
        if data.empty:
            raise ValueError("ReplaySimulator requires at least one bar")
        if time_column not in data.columns:
            raise ValueError(f"Missing time column: {time_column}")
        timestamps = pd.to_datetime(data[time_column], utc=True)
        if timestamps.isnull().any():
            raise ValueError("Timestamp column contains null values")
        ordered = data.assign(**{time_column: timestamps}).sort_values(time_column)
        ordered = ordered.reset_index(drop=True)
        records: List[Dict[str, Any]] = []
        parsed_times: List[datetime] = []
        for _, row in ordered.iterrows():
            ts = row[time_column]
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            elif not isinstance(ts, datetime):
                raise TypeError("Timestamp column must be datetime-like")
            payload = {
                column: _coerce_value(value)
                for column, value in row.items()
                if column != time_column
            }
            payload[time_column] = ts
            records.append(payload)
            parsed_times.append(ts)
        self._records = records
        self._timestamps = parsed_times
        self._total = len(records)
        self._listeners: List[Listener] = []
        self._lock = threading.RLock()
        self._speed = float(speed)
        if self._speed <= 0:
            raise ValueError("Speed multiplier must be positive")
        self._sleep: SleepFn = sleep or time.sleep
        self._clock: ClockFn = clock or time.perf_counter
        self._max_step = max(0.01, float(max_step))
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._seek_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._next_index = 0
        self._logger = utils.build_logger("ReplaySimulator")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        fmt: str | None = None,
        time_column: str = "timestamp",
        speed: float = 1.0,
        limit: int | None = None,
    ) -> "ReplaySimulator":
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        dataset_format = detect_format(candidate, fmt)
        if dataset_format == "csv":
            frame = pd.read_csv(candidate)
        else:
            frame = pd.read_parquet(candidate)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            frame = frame.iloc[: int(limit)]
        if frame.empty:
            raise ValueError("Replay dataset contains no rows")
        return cls(frame, time_column=time_column, speed=speed)

    @property
    def total_bars(self) -> int:
        return self._total

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def running(self) -> bool:
        return self._running

    def add_listener(self, listener: Listener) -> None:
        """Register *listener* for bar callbacks."""

        with self._lock:
            self._listeners.append(listener)

    def clear_listeners(self) -> None:
        with self._lock:
            self._listeners.clear()

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("Speed multiplier must be positive")
        with self._lock:
            self._speed = float(speed)

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self._seek_event.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._running = False

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def seek(self, target: int | float | datetime | str) -> None:
        if self._total == 0:
            return
        index: int
        if isinstance(target, int):
            index = target
        elif isinstance(target, float):
            index = int(target * (self._total - 1))
        elif isinstance(target, datetime):
            index = bisect_left(self._timestamps, target)
        elif isinstance(target, str):
            parsed = pd.to_datetime(target, utc=True)
            if pd.isna(parsed):
                raise ValueError(f"Unable to parse seek timestamp: {target}")
            index = bisect_left(self._timestamps, parsed.to_pydatetime())
        else:
            raise TypeError("Seek target must be int, float, datetime, or str")
        index = max(0, min(index, self._total - 1))
        with self._lock:
            self._next_index = index
        self._seek_event.set()
        self._pause_event.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Replay already running")
        self._stop_event.clear()
        self._pause_event.set()
        with self._lock:
            if self._next_index >= self._total:
                self._next_index = 0
        self._seek_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def run(self) -> None:
        if self._running:
            raise RuntimeError("Replay already running")
        self._stop_event.clear()
        self._pause_event.set()
        with self._lock:
            if self._next_index >= self._total:
                self._next_index = 0
        self._seek_event.clear()
        self._run_loop()

    def _emit(self, index: int, payload: Dict[str, Any]) -> None:
        bar = ReplayBar(index=index, timestamp=self._timestamps[index], data=payload)
        listeners: List[Listener]
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(bar)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.exception("Replay listener failed: %s", exc)

    def _run_loop(self) -> None:
        self._running = True
        cursor = -1
        last_timestamp: datetime | None = None
        try:
            while not self._stop_event.is_set():
                if not self._pause_event.wait(timeout=0.1):
                    continue
                with self._lock:
                    target = self._next_index
                if target >= self._total:
                    break
                if target != cursor:
                    cursor = target
                    last_timestamp = None
                payload = self._records[cursor]
                current_timestamp = self._timestamps[cursor]
                if last_timestamp is not None:
                    delta = (current_timestamp - last_timestamp).total_seconds()
                    if delta > 0:
                        self._wait(delta)
                    if self._stop_event.is_set():
                        break
                    if self._seek_event.is_set():
                        self._seek_event.clear()
                        last_timestamp = None
                        continue
                self._emit(cursor, payload)
                last_timestamp = current_timestamp
                cursor += 1
                with self._lock:
                    self._next_index = cursor
                if cursor >= self._total:
                    break
        finally:
            self._running = False
            self._stop_event.set()
            self._pause_event.set()

    def _wait(self, bar_seconds: float) -> None:
        remaining = max(bar_seconds, 0.0)
        while remaining > 1e-6 and not self._stop_event.is_set():
            if not self._pause_event.wait(timeout=0.05):
                continue
            if self._seek_event.is_set():
                break
            with self._lock:
                speed = self._speed
            speed = max(speed, 1e-6)
            real_chunk = min(remaining / speed, self._max_step)
            if real_chunk <= 0:
                break
            start = self._clock()
            self._sleep(real_chunk)
            elapsed = max(self._clock() - start, real_chunk)
            remaining -= elapsed * speed
            if self._seek_event.is_set():
                break


def _queue_listener(queue: SimpleQueue[ReplayBar]) -> Listener:
    def _listener(bar: ReplayBar) -> None:
        queue.put(bar)

    return _listener


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play back historical bars")
    parser.add_argument("path", help="CSV or Parquet file containing bars")
    parser.add_argument("--format", choices=["auto", "csv", "parquet"], default="auto")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed multiplier"
    )
    parser.add_argument(
        "--time-column",
        default="timestamp",
        help="Name of the timestamp column (default: timestamp)",
    )
    parser.add_argument(
        "--seek",
        help="Seek to a zero-based index or ISO8601 timestamp before playback starts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of bars to stream",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        simulator = ReplaySimulator.from_path(
            args.path,
            fmt=args.format,
            time_column=args.time_column,
            speed=args.speed,
            limit=args.limit,
        )
    except Exception as exc:  # pragma: no cover - CLI entry point
        print(f"Error: {exc}")
        return 1
    queue: SimpleQueue[ReplayBar] = SimpleQueue()
    simulator.add_listener(_queue_listener(queue))
    if args.seek:
        try:
            if args.seek.isdigit():
                simulator.seek(int(args.seek))
            else:
                simulator.seek(args.seek)
        except Exception as exc:  # pragma: no cover - CLI guard
            print(f"Seek ignored: {exc}")
    simulator.start()
    try:
        while True:
            try:
                bar = queue.get(timeout=0.25)
            except Empty:
                if not simulator.running and queue.empty():
                    break
                continue
            print(json.dumps(bar.to_dict(), default=str))
    except KeyboardInterrupt:  # pragma: no cover - interactive guard
        simulator.stop()
    finally:
        simulator.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI bootstrap
    raise SystemExit(main())


__all__ = ["ReplayBar", "ReplaySimulator", "detect_format", "main"]
