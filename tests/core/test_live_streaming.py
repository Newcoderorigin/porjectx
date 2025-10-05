"""Deterministic tests for SignalR live streaming helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import types
from typing import Any, Callable, Dict, List, Tuple

if "httpx" not in sys.modules:  # pragma: no cover - import shim for optional dependency
    httpx_stub = types.ModuleType("httpx")

    class _StubResponse:
        def __init__(self, text: str = "") -> None:
            self.text = text

        def json(self) -> Dict[str, Any]:  # pragma: no cover - defensive only
            return {}

    class HTTPStatusError(Exception):
        def __init__(self, message: str = "", request: Any = None, response: Any = None) -> None:
            super().__init__(message)
            self.request = request
            self.response = response or _StubResponse()

    class Client:  # pragma: no cover - network disabled during tests
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def post(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("HTTP client stub does not support network calls")

        def close(self) -> None:  # noqa: D401 - interface compatibility
            """No-op close to satisfy gateway expectations."""

    httpx_stub.Client = Client
    httpx_stub.HTTPStatusError = HTTPStatusError
    httpx_stub.Response = _StubResponse
    sys.modules["httpx"] = httpx_stub

import pytest

from toptek.core import live


class DummySignalRConnection:
    """Test double that mimics the minimal SignalR hub API surface."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.sent: List[Tuple[str, List[Any]]] = []
        self._listeners: Dict[str, List[Tuple[str, Callable[[Any], None]]]] = defaultdict(list)
        self._open_callbacks: List[Callable[[], None]] = []
        self._close_callbacks: List[Callable[[Any], None]] = []
        self._counter = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def on(self, event: str, handler: Callable[[Any], None]) -> str:
        token = f"{event}-{self._counter}"
        self._counter += 1
        self._listeners[event].append((token, handler))
        return token

    def remove_listener(self, event: str, identifier: Any) -> None:
        listeners = self._listeners.get(event, [])
        if identifier is None:
            self._listeners[event] = []
            return

        remaining: List[Tuple[str, Callable[[Any], None]]] = []
        for token, handler in listeners:
            if identifier in (token, handler):
                continue
            remaining.append((token, handler))
        self._listeners[event] = remaining

    def off(self, event: str, identifier: Any | None = None) -> None:
        self.remove_listener(event, identifier)

    def send(self, method: str, args: List[Any]) -> None:
        self.sent.append((method, args))

    def on_open(self, callback: Callable[[], None]) -> None:
        self._open_callbacks.append(callback)

    def on_close(self, callback: Callable[[Any], None]) -> None:
        self._close_callbacks.append(callback)

    def trigger_open(self) -> None:
        for callback in list(self._open_callbacks):
            callback()

    def trigger_close(self, error: Any = None) -> None:
        for callback in list(self._close_callbacks):
            callback(error)

    def emit(self, event: str, payload: Any) -> None:
        for _, handler in list(self._listeners.get(event, [])):
            handler([payload])


class DummyHubConnectionBuilder:
    """Builder double compatible with :func:`connect_market_hub`."""

    instances: List["DummyHubConnectionBuilder"] = []

    def __init__(self) -> None:
        self.url: str | None = None
        self.options: Dict[str, Any] | None = None
        self.connection: DummySignalRConnection = DummySignalRConnection()
        DummyHubConnectionBuilder.instances.append(self)

    def with_url(self, url: str, options: Dict[str, Any] | None = None) -> "DummyHubConnectionBuilder":
        self.url = url
        self.options = options
        return self

    def build(self) -> DummySignalRConnection:
        return self.connection


@pytest.fixture(autouse=True)
def reset_builder_instances() -> None:
    DummyHubConnectionBuilder.instances.clear()


def test_connect_market_hub_merges_headers_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_require_signalr_builder", lambda: DummyHubConnectionBuilder)

    opened: List[bool] = []
    closed: List[Any] = []

    handle = live.connect_market_hub(
        "https://example.com/api",
        hub_path="stream",
        headers={"Authorization": "token"},
        options={"headers": {"User-Agent": "ProjectX"}},
        on_open=lambda: opened.append(True),
        on_close=lambda exc: closed.append(exc),
    )

    assert isinstance(handle, live.HubConnectionHandle)
    builder = DummyHubConnectionBuilder.instances[-1]
    assert builder.url == "https://example.com/api/stream"
    assert builder.options == {
        "headers": {"User-Agent": "ProjectX", "Authorization": "token"}
    }

    connection = handle.connection
    assert connection.started is True

    connection.trigger_open()
    assert opened == [True]

    connection.trigger_close(None)
    assert closed == [None]

    handle.close()
    assert connection.stopped is True


def test_subscribe_ticker_dispatch_and_unsubscribe() -> None:
    connection = DummySignalRConnection()
    events: List[Tuple[str, Any]] = []

    handle = live.subscribe_ticker(
        connection,
        "ES=F",
        lambda symbol, payload: events.append((symbol, payload)),
        event="ticker",
    )

    assert handle.connection is connection
    assert connection.sent == [("SubscribeTicker", ["ES=F"])]

    connection.emit("ticker", {"bid": 1})
    assert events == [("ES=F", {"bid": 1})]

    handle.unsubscribe()
    assert ("UnsubscribeTicker", ["ES=F"]) in connection.sent

    connection.emit("ticker", {"bid": 2})
    assert events == [("ES=F", {"bid": 1})]


def test_subscribe_bars_uses_handle_and_timeframe() -> None:
    handle = live.HubConnectionHandle(DummySignalRConnection())
    events: List[Tuple[str, str, Any]] = []

    subscription = live.subscribe_bars(
        handle,
        "NQ=F",
        "1m",
        lambda symbol, timeframe, payload: events.append((symbol, timeframe, payload)),
        event="bars",
    )

    connection = handle.connection
    assert connection.sent == [("SubscribeBars", ["NQ=F", "1m"])]

    connection.emit("bars", {"close": 4100})
    assert events == [("NQ=F", "1m", {"close": 4100})]

    subscription.unsubscribe()
    assert ("UnsubscribeBars", ["NQ=F", "1m"]) in connection.sent

    connection.emit("bars", {"close": 4200})
    assert events == [("NQ=F", "1m", {"close": 4100})]


def test_utils_module_behaviour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from toptek.core import utils

    logger = utils.build_logger("test-live-streaming", level="DEBUG")
    assert logger.name == "test-live-streaming"

    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text("foo: 1\n", encoding="utf-8")
    assert utils.load_yaml(yaml_path) == {"foo": 1}
    assert utils.load_yaml(tmp_path / "missing.yml") == {}

    paths = utils.build_paths(
        tmp_path,
        {"cache_directory": "cache_dir", "models_directory": "models_dir"},
    )
    utils.ensure_directories(paths)
    assert paths.cache.exists() and paths.models.exists()

    env_key = "TOPTEK_TEST_ENV"
    monkeypatch.delenv(env_key, raising=False)
    assert utils.env_or_default(env_key, "fallback") == "fallback"
    monkeypatch.setenv(env_key, "configured")
    assert utils.env_or_default(env_key, "fallback") == "configured"

    timestamp = utils.timestamp()
    assert timestamp.tzinfo == utils.DEFAULT_TIMEZONE
    json_blob = utils.json_dumps({"ts": timestamp})
    assert "ts" in json_blob

    assert utils._version_tuple("1.2.3-alpha") == (1, 2, 3)
    assert utils._version_tuple("1..2") == (1, 2)
    assert utils._version_tuple("1a.2") == (1, 2)
    assert utils._version_tuple("a1") == (0,)
    assert utils._compare_versions((1, 2), (1, 2, 1)) == -1
    assert utils._spec_matches("2.1.2", ">=2.1,<3") is True
    assert utils._spec_matches("1.0", ">=0.9,,<2") is True
    assert utils._spec_matches("1.0", ">= ") is True
    assert utils._spec_matches("1.0", "==1.1") is False

    resolved_versions = {"numpy": "2.1.2", "scipy": "1.14.1", "scikit-learn": "1.6.0"}

    monkeypatch.setattr(
        utils.metadata,
        "version",
        lambda package: resolved_versions[package],
    )
    utils.assert_numeric_stack(
        {
            "numpy": ">=2.1.2,<3",
            "scipy": ">=1.14.1,<2",
            "scikit-learn": "==1.6.0",
        }
    )

    def _version_with_error(package: str) -> str:
        if package == "numpy":
            raise utils.PackageNotFoundError
        if package == "scipy":
            return "1.14.0"
        return "1.6.0"

    monkeypatch.setattr(utils.metadata, "version", _version_with_error)
    with pytest.raises(RuntimeError) as exc:
        utils.assert_numeric_stack(
            {
                "numpy": ">=2.1.2,<3",
                "scipy": ">=1.14.1,<2",
                "scikit-learn": "==1.6.0",
            }
        )

    message = exc.value.args[0]
    assert "Missing packages" in message
    assert "Version mismatches" in message
