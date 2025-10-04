from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import pytest

from toptek.ai_server.config import AISettings
from toptek.ai_server.lmstudio import HTTPError, LMStudioClient


@dataclass
class _FakeResponse:
    status_code: int = 200
    json_payload: Dict[str, Any] | None = None
    lines: Iterable[str] | None = None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP status {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self.json_payload or {}

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in list(self.lines or []):
            await asyncio.sleep(0)
            yield line


class _FakeStreamResponse(_FakeResponse):
    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        return None


class _StubAsyncClient:
    def __init__(self) -> None:
        self._routes: Dict[tuple[str, str], Any] = {}
        self.stream_payloads: List[Dict[str, Any]] = []
        self.closed = False

    def add_get(self, path: str, response: Any) -> None:
        self._routes[("GET", path)] = response

    def add_stream(self, path: str, factory: Any) -> None:
        self._routes[("STREAM", path)] = factory

    async def get(self, path: str, *args: Any, **kwargs: Any) -> Any:
        key = ("GET", path)
        if key not in self._routes:
            raise AssertionError(f"Unexpected GET {path}")
        result = self._routes[key]
        if isinstance(result, Exception):
            raise result
        return result

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Dict[str, Any],
        timeout: Any,
    ) -> _FakeStreamResponse:
        key = ("STREAM", path)
        if key not in self._routes:
            raise AssertionError(f"Unexpected stream {method} {path}")
        factory = self._routes[key]
        if isinstance(factory, Exception):
            raise factory
        self.stream_payloads.append(json)
        response = factory()
        if not isinstance(response, _FakeStreamResponse):
            raise AssertionError("Stream factory must return _FakeStreamResponse")
        return response

    async def aclose(self) -> None:
        self.closed = True


def _settings() -> AISettings:
    return AISettings(
        base_url="http://localhost:1234/v1",
        port=1234,
        auto_start=False,
        poll_interval_seconds=0.1,
        poll_timeout_seconds=1.0,
        default_model="stable",
        default_role="system",
    )


def test_list_models_success() -> None:
    stub = _StubAsyncClient()
    stub.add_get(
        "/models",
        _FakeResponse(
            status_code=200,
            json_payload={
                "data": [
                    {
                        "id": "model-a",
                        "owned_by": "local",
                        "metadata": {"context_length": 8192, "display_name": "A"},
                        "capabilities": {"tool_calls": True},
                    },
                    {
                        "id": "model-b",
                        "metadata": {"context_window": 4096},
                        "performance": {"tokens_per_second": 40.5, "ttft": 120},
                    },
                ]
            },
        ),
    )

    async def _run() -> None:
        client = LMStudioClient(_settings(), client=stub)
        models = await client.list_models()

        assert [model.model_id for model in models] == ["model-a", "model-b"]
        assert models[0].supports_tools is True
        assert models[0].max_context == 8192
        assert models[1].tokens_per_second == pytest.approx(40.5)
        assert models[1].ttft == pytest.approx(120.0)

    asyncio.run(_run())


def test_list_models_http_error() -> None:
    stub = _StubAsyncClient()
    stub.add_get("/models", _FakeResponse(status_code=503))
    async def _run() -> None:
        client = LMStudioClient(_settings(), client=stub)
        with pytest.raises(HTTPError):
            await client.list_models()

    asyncio.run(_run())


def test_health_handles_timeout() -> None:
    stub = _StubAsyncClient()
    stub.add_get("/models", HTTPError("timeout"))
    async def _run() -> None:
        client = LMStudioClient(_settings(), client=stub)
        healthy = await client.health()
        assert healthy is False

    asyncio.run(_run())


def test_chat_stream_temperature_zero_is_deterministic() -> None:
    lines = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
        "",
        "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}",
    ]

    def factory() -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=lines)

    async def _run() -> None:
        stub = _StubAsyncClient()
        stub.add_stream("/chat/completions", factory)

        client = LMStudioClient(_settings(), client=stub)
        payload = {"model": "model-a", "temperature": 0.0, "messages": []}

        first_run = [chunk async for chunk in client.chat_stream(payload)]
        second_run = [chunk async for chunk in client.chat_stream(payload)]

        assert first_run == [lines[0], lines[2]]
        assert second_run == first_run
        assert all(item["temperature"] == 0.0 for item in stub.stream_payloads)

    asyncio.run(_run())

