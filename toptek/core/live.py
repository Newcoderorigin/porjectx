"""Live trading utilities with optional SignalR streaming helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence

from .gateway import ProjectXGateway


_STREAMING_IMPORT_MESSAGE = (
    "SignalR streaming requires the 'signalrcore' package. "
    "Install it via the streaming extras profile or `pip install signalrcore`."
)


@dataclass
class ExecutionContext:
    """Represents the state required for placing orders."""

    gateway: ProjectXGateway
    account_id: str


@dataclass
class HubConnectionHandle:
    """Wrapper around a SignalR hub connection with a close helper."""

    connection: Any

    def close(self) -> None:
        """Stop the underlying connection if it exposes a stop method."""

        if hasattr(self.connection, "stop"):
            self.connection.stop()


@dataclass
class SubscriptionHandle:
    """Represents an active SignalR subscription that can be torn down."""

    connection: Any
    event: str
    handler: Callable[[Any], None]
    handler_token: Any
    unsubscribe_method: Optional[str]
    unsubscribe_payload: Sequence[Any]

    def unsubscribe(self) -> None:
        """Detach the handler and propagate an unsubscribe message."""

        _remove_listener(self.connection, self.event, self.handler, self.handler_token)
        if self.unsubscribe_method:
            self.connection.send(self.unsubscribe_method, list(self.unsubscribe_payload))


def poll_open_orders(context: ExecutionContext) -> Dict[str, object]:
    """Poll open orders using the REST API."""

    return context.gateway.search_open_orders({"accountId": context.account_id})


def poll_positions(context: ExecutionContext) -> Dict[str, object]:
    """Poll open positions."""

    return context.gateway.search_positions({"accountId": context.account_id})


def connect_market_hub(
    base_url: str,
    *,
    hub_path: str = "marketHub",
    headers: Optional[MutableMapping[str, str]] = None,
    options: Optional[MutableMapping[str, Any]] = None,
    auto_start: bool = True,
    on_open: Optional[Callable[[], None]] = None,
    on_close: Optional[Callable[[Optional[Exception]], None]] = None,
) -> HubConnectionHandle:
    """Create and optionally start a SignalR hub connection."""

    builder_cls = _require_signalr_builder()
    normalized_url = _join_url(base_url, hub_path)
    connection_options = _merge_options(options, headers)

    connection = builder_cls().with_url(normalized_url, options=connection_options).build()

    if on_open and hasattr(connection, "on_open"):
        connection.on_open(on_open)
    if on_close and hasattr(connection, "on_close"):
        connection.on_close(on_close)

    if auto_start and hasattr(connection, "start"):
        connection.start()

    return HubConnectionHandle(connection)


def subscribe_ticker(
    connection: HubConnectionHandle | Any,
    symbol: str,
    callback: Callable[[str, Any], None],
    *,
    event: str = "ticker_update",
    subscribe_method: Optional[str] = "SubscribeTicker",
    unsubscribe_method: Optional[str] = "UnsubscribeTicker",
) -> SubscriptionHandle:
    """Attach a ticker listener and optionally send subscribe/unsubscribe calls."""

    signalr_connection = _unwrap_connection(connection)
    handler, token = _register_handler(
        signalr_connection,
        event,
        _wrap_payload(callback, symbol),
    )

    if subscribe_method:
        signalr_connection.send(subscribe_method, [symbol])

    return SubscriptionHandle(
        signalr_connection,
        event,
        handler,
        token,
        unsubscribe_method,
        [symbol],
    )


def subscribe_bars(
    connection: HubConnectionHandle | Any,
    symbol: str,
    timeframe: str,
    callback: Callable[[str, str, Any], None],
    *,
    event: str = "bar_update",
    subscribe_method: Optional[str] = "SubscribeBars",
    unsubscribe_method: Optional[str] = "UnsubscribeBars",
) -> SubscriptionHandle:
    """Attach a bar listener for the provided symbol and timeframe."""

    signalr_connection = _unwrap_connection(connection)
    handler, token = _register_handler(
        signalr_connection,
        event,
        _wrap_payload(callback, symbol, timeframe),
    )

    if subscribe_method:
        signalr_connection.send(subscribe_method, [symbol, timeframe])

    return SubscriptionHandle(
        signalr_connection,
        event,
        handler,
        token,
        unsubscribe_method,
        [symbol, timeframe],
    )


def _require_signalr_builder():
    try:
        from signalrcore.hub_connection_builder import HubConnectionBuilder
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(_STREAMING_IMPORT_MESSAGE) from exc

    return HubConnectionBuilder


def _unwrap_connection(connection: HubConnectionHandle | Any) -> Any:
    return connection.connection if isinstance(connection, HubConnectionHandle) else connection


def _register_handler(connection: Any, event: str, handler: Callable[[Any], None]) -> tuple[Callable[[Any], None], Any]:
    token = connection.on(event, handler)
    return handler, token


def _wrap_payload(
    callback: Callable[..., None],
    *prefix_args: str,
) -> Callable[[Any], None]:
    def _inner(message: Any) -> None:
        payload = _extract_payload(message)
        callback(*prefix_args, payload)

    return _inner


def _extract_payload(message: Any) -> Any:
    if isinstance(message, (list, tuple)):
        return message[0] if message else None
    return message


def _remove_listener(connection: Any, event: str, handler: Callable[[Any], None], token: Any) -> None:
    if hasattr(connection, "remove_listener"):
        try:
            connection.remove_listener(event, token if token is not None else handler)
            return
        except TypeError:
            connection.remove_listener(event, handler)

    if hasattr(connection, "off"):
        try:
            if token is not None:
                connection.off(event, token)
                return
        except TypeError:
            pass
        try:
            connection.off(event, handler)
            return
        except TypeError:
            connection.off(event)


def _merge_options(
    options: Optional[MutableMapping[str, Any]],
    headers: Optional[MutableMapping[str, str]],
) -> MutableMapping[str, Any]:
    merged: Dict[str, Any] = dict(options or {})
    if headers:
        merged_headers = dict(merged.get("headers", {}))
        merged_headers.update(headers)
        merged["headers"] = merged_headers
    return merged


def _join_url(base_url: str, hub_path: str) -> str:
    if not hub_path:
        return base_url
    return f"{base_url.rstrip('/')}/{hub_path.lstrip('/')}"


__all__ = [
    "ExecutionContext",
    "HubConnectionHandle",
    "SubscriptionHandle",
    "poll_open_orders",
    "poll_positions",
    "connect_market_hub",
    "subscribe_ticker",
    "subscribe_bars",
]
