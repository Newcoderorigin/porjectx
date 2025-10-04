from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import pytest

from toptek.lmstudio import HTTPError, LMStudioClient, Model


@dataclass
class _StubResponse:
    status: int = 200
    body: str | bytes = "{}"
    lines: Iterable[str] | None = None

    def read(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        return self.body.encode("utf-8")

    def iter_lines(self) -> Iterable[str]:
        for line in list(self.lines or []):
            yield line


class _StubTransport:
    def __init__(self) -> None:
        self.routes: Dict[Tuple[str, str], Any] = {}
        self.requests: List[Tuple[str, str, bytes | None, Dict[str, str] | None]] = []

    def add(self, method: str, url: str, response: Any) -> None:
        self.routes[(method, url)] = response

    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: Dict[str, str] | None = None,
        timeout: float = 0.0,
    ) -> _StubResponse:
        self.requests.append((method, url, data, headers))
        key = (method, url)
        if key not in self.routes:
            raise AssertionError(f"Unexpected request {method} {url}")
        response = self.routes[key]
        if isinstance(response, Exception):
            raise response
        return response


def _settings() -> Dict[str, Any]:
    return {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "timeout_s": 30,
    }


def test_list_models_success() -> None:
    transport = _StubTransport()
    url = "http://localhost:1234/v1/models"
    transport.add(
        "GET",
        url,
        _StubResponse(
            status=200,
            body=json.dumps(
                {
                    "data": [
                        {
                            "id": "model-a",
                            "owned_by": "local",
                            "metadata": {"context_length": 8192, "display_name": "Alpha"},
                        },
                        {
                            "id": "model-b",
                            "metadata": {"context_window": 4096},
                            "description": "Beta",
                        },
                    ]
                }
            ),
        ),
    )

    client = LMStudioClient(_settings(), transport=transport)
    models = client.list_models()

    assert isinstance(models[0], Model)
    assert [model.model_id for model in models] == ["model-a", "model-b"]
    assert models[0].max_context == 8192
    assert models[0].description == "Alpha"
    assert models[1].max_context == 4096


def test_list_models_http_error() -> None:
    transport = _StubTransport()
    transport.add("GET", "http://localhost:1234/v1/models", _StubResponse(status=503))

    client = LMStudioClient(_settings(), transport=transport)
    with pytest.raises(HTTPError):
        client.list_models()


def test_health_handles_failure() -> None:
    transport = _StubTransport()
    transport.add("GET", "http://localhost:1234/v1/models", HTTPError("timeout"))

    client = LMStudioClient(_settings(), transport=transport)
    assert client.health() is False


def test_chat_stream_temperature_zero_is_deterministic() -> None:
    lines = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
        "",
        "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}",
    ]
    transport = _StubTransport()
    transport.add(
        "POST",
        "http://localhost:1234/v1/chat/completions",
        _StubResponse(status=200, lines=lines),
    )

    client = LMStudioClient(_settings(), transport=transport)
    payload = {"model": "model-a", "temperature": 0.0, "messages": []}

    first_run = list(client.chat_stream(payload))
    second_run = list(client.chat_stream(payload))

    assert first_run == [lines[0], lines[2]]
    assert second_run == first_run

    request_bodies = [
        json.loads(body.decode("utf-8"))
        for _, _, body, _ in transport.requests
        if body is not None
    ]
    for body in request_bodies:
        assert body["temperature"] == 0.0

