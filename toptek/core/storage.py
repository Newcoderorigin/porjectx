"""Persistent user storage for GUI state and activity history.

This module provides a light-weight persistence layer for the GUI so that
session preferences, trained model summaries, and activity history survive
across program restarts. Data is written to JSON on disk and exposed via a
simple publish/subscribe API for real-time UI updates.

Example:
    >>> from pathlib import Path
    >>> from core.storage import UserStorage
    >>> store = UserStorage(Path("data/user/state.json"))
    >>> store.update_section("preferences", {"theme": "dark"})
    >>> store.append_history("login", "Signed in", {"username": "demo"})
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, MutableMapping


@dataclass
class UserEvent:
    """Represents a persisted user event."""

    timestamp: str
    event_type: str
    message: str
    payload: Dict[str, Any]


class UserStorage:
    """Manage durable user state and publish updates to listeners."""

    def __init__(self, state_path: Path, *, history_limit: int = 200) -> None:
        self.state_path = state_path
        self.history_limit = history_limit
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._state: Dict[str, Any] = {"sections": {}, "history": []}
        self._last_loaded = 0.0
        self._load()

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register *callback* to receive state snapshots when changes occur."""

        with self._lock:
            self._listeners.append(callback)
        callback(self.snapshot())

    def update_section(self, name: str, data: MutableMapping[str, Any]) -> None:
        """Persist a structured section under *name*."""

        with self._lock:
            self._state.setdefault("sections", {})[name] = dict(data)
            self._persist_locked()
            self._broadcast()

    def append_history(self, event_type: str, message: str, payload: Dict[str, Any] | None = None) -> None:
        """Append a history event with metadata."""

        event = UserEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            event_type=event_type,
            message=message,
            payload=payload or {},
        )
        with self._lock:
            history: List[Dict[str, Any]] = self._state.setdefault("history", [])
            history.append(event.__dict__)
            if len(history) > self.history_limit:
                del history[: len(history) - self.history_limit]
            self._persist_locked()
            self._broadcast()

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the storage state."""

        with self._lock:
            return json.loads(json.dumps(self._state))

    def get_section(self, name: str, default: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        """Retrieve a stored section."""

        with self._lock:
            sections = self._state.get("sections", {})
            if name not in sections:
                return default
            return json.loads(json.dumps(sections[name]))

    def reload_if_changed(self) -> bool:
        """Reload state from disk when the backing file changes."""

        try:
            mtime = self.state_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if mtime <= self._last_loaded:
            return False
        self._load()
        self._broadcast()
        return True

    # Internal helpers -------------------------------------------------

    def _broadcast(self) -> None:
        snapshot = self.snapshot()
        for listener in list(self._listeners):
            listener(snapshot)

    def _load(self) -> None:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                try:
                    self._state = json.load(handle)
                except json.JSONDecodeError:
                    self._state = {"sections": {}, "history": []}
            self._last_loaded = self.state_path.stat().st_mtime
        else:
            self._persist_locked()

    def _persist_locked(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2)
        self._last_loaded = self.state_path.stat().st_mtime


__all__ = ["UserStorage", "UserEvent"]
