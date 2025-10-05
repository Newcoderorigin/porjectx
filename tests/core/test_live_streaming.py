"""Deterministic tests for the live streaming session abstraction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import types
from typing import Any, Callable, Dict, List, Tuple

if "httpx" not in sys.modules:  # pragma: no cover - optional dependency shim
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


class DummyGateway:
    """Gateway double that only tracks auth header refreshes."""

    def __init__(self) -> None:
        self._counter = 0
        self.base_url = "https://example.com/api"
        self.auth_requests: List[str] = []

    def auth_headers(self) -> Dict[str, str]:
        token = f"token-{self._counter}"
        self._counter += 1
        self.auth_requests.append(token)
        return {"Authorization": token}

    def search_open_orders(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"orders": payload}

    def search_positions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"positions": payload}


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
        self.sent.append((method, list(args)))

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
    """Builder double compatible with :class:`GatewayStreamingSession`."""

    instances: List["DummyHubConnectionBuilder"] = []

    def __init__(self) -> None:
        self.url: str | None = None
        self.options: Dict[str, Any] | None = None
        self.connection: DummySignalRConnection = DummySignalRConnection()
        DummyHubConnectionBuilder.instances.append(self)

    def with_url(self, url: str, options: Dict[str, Any] | None = None) -> "DummyHubConnectionBuilder":
        self.url = url
        self.options = options or {}
        return self

    def build(self) -> DummySignalRConnection:
        return self.connection


@pytest.fixture(autouse=True)
def reset_builder_instances() -> None:
    DummyHubConnectionBuilder.instances.clear()


def test_streaming_session_fanout_and_resubscribe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_require_signalr_builder", lambda: DummyHubConnectionBuilder)

    gateway = DummyGateway()
    session = live.GatewayStreamingSession(gateway)

    ticker_events: List[Tuple[str, Any]] = []
    bar_events: List[Tuple[str, str, Any]] = []
    depth_events: List[Tuple[str, Any]] = []
    order_events: List[Tuple[str, Any]] = []
    position_events: List[Tuple[str, Any]] = []
    trade_events: List[Tuple[str, Any]] = []
    account_events: List[Any] = []

    ticker_handle = session.market.subscribe_ticker(
        "ES=F", lambda symbol, payload: ticker_events.append((symbol, payload))
    )
    session.market.subscribe_bars(
        "NQ=F",
        "1m",
        lambda symbol, timeframe, payload: bar_events.append((symbol, timeframe, payload)),
    )
    session.market.subscribe_depth(
        "CL=F",
        lambda symbol, payload: depth_events.append((symbol, payload)),
    )

    session.user.subscribe_orders(
        "ACCT1", lambda account, payload: order_events.append((account, payload))
    )
    session.user.subscribe_positions(
        "ACCT1", lambda account, payload: position_events.append((account, payload))
    )
    session.user.subscribe_trades(
        "ACCT1", lambda account, payload: trade_events.append((account, payload))
    )
    session.user.subscribe_accounts(lambda payload: account_events.append(payload))

    assert len(DummyHubConnectionBuilder.instances) == 2

    market_builder, user_builder = DummyHubConnectionBuilder.instances
    assert market_builder.url == "https://example.com/api/marketHub"
    assert user_builder.url == "https://example.com/api/userHub"

    market_connection = market_builder.connection
    user_connection = user_builder.connection
    assert market_connection.started is True
    assert user_connection.started is True

    market_connection.trigger_open()
    user_connection.trigger_open()

    market_connection.emit("ticker_update", {"bid": 1})
    market_connection.emit("bar_update", {"close": 4100})
    market_connection.emit("depth_update", {"levels": []})
    user_connection.emit("order_update", {"id": 1})
    user_connection.emit("position_update", {"symbol": "ES=F"})
    user_connection.emit("trade_update", {"qty": 2})
    user_connection.emit("account_update", {"margin": 1000})

    assert ticker_events == [("ES=F", {"bid": 1})]
    assert bar_events == [("NQ=F", "1m", {"close": 4100})]
    assert depth_events == [("CL=F", {"levels": []})]
    assert order_events == [("ACCT1", {"id": 1})]
    assert position_events == [("ACCT1", {"symbol": "ES=F"})]
    assert trade_events == [("ACCT1", {"qty": 2})]
    assert account_events == [{"margin": 1000}]

    sent_methods = [method for method, _ in market_connection.sent]
    assert sent_methods.count("SubscribeTicker") == 1
    assert sent_methods.count("SubscribeBars") == 1
    assert sent_methods.count("SubscribeDepth") == 1

    market_connection.trigger_close(None)

    assert len(DummyHubConnectionBuilder.instances) == 3
    reconnect_builder = DummyHubConnectionBuilder.instances[-1]
    new_market_connection = reconnect_builder.connection
    assert reconnect_builder.options == {"headers": {"Authorization": "token-3"}}

    new_market_connection.trigger_open()
    resubscribe_calls = [method for method, _ in new_market_connection.sent]
    assert resubscribe_calls.count("SubscribeTicker") == 1
    assert resubscribe_calls.count("SubscribeBars") == 1
    assert resubscribe_calls.count("SubscribeDepth") == 1

    new_market_connection.emit("ticker_update", {"bid": 2})
    assert ticker_events[-1] == ("ES=F", {"bid": 2})

    ticker_handle.close()
    assert ("UnsubscribeTicker", ["ES=F"]) in new_market_connection.sent

    session.close()
    assert new_market_connection.stopped is True
    assert user_connection.stopped is True

    assert gateway.auth_requests[:2] == ["token-0", "token-1"]


def test_poll_helpers_use_gateway() -> None:
    gateway = DummyGateway()
    context = live.ExecutionContext(gateway=gateway, account_id="ACCT2")

    orders = live.poll_open_orders(context)
    positions = live.poll_positions(context)

    assert orders == {"orders": {"accountId": "ACCT2"}}
    assert positions == {"positions": {"accountId": "ACCT2"}}


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

