"""Unit tests for the minimal LM Studio client."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from toptek import lmstudio


class DummyResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no-op
        return None


class DummyURLLib:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def urlopen(self, request: Any, timeout: int | None = None) -> DummyResponse:
        self.request = request
        self.timeout = timeout
        return DummyResponse(self._payload)


def test_chat_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "base_url": "http://example",
        "api_key": "token",
        "model": "stub-model",
    }
    payload = {"choices": [{"message": {"content": "ok"}}]}
    dummy = DummyURLLib(payload)
    monkeypatch.setattr(lmstudio.urllib.request, "urlopen", dummy.urlopen)
    monkeypatch.setattr(lmstudio.time, "perf_counter", lambda: 1.0)
    client = lmstudio.build_client(config)
    reply, latency = client.chat([{"role": "user", "content": "ping"}])
    assert reply == "ok"
    assert latency == pytest.approx(0.0)
    assert dummy.timeout == client.config.timeout_s
    assert dummy.request.get_header("Authorization") == "Bearer token"
