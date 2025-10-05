"""Live trading streaming abstractions for ProjectX hubs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

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
class HubSubscriptionHandle:
    """Represents an active hub subscription that can be torn down."""

    _subscription: "_Subscription | None"

    def close(self) -> None:
        """Close the underlying subscription."""

        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None

    unsubscribe = close


def poll_open_orders(context: ExecutionContext) -> Dict[str, object]:
    """Poll open orders using the REST API."""

    return context.gateway.search_open_orders({"accountId": context.account_id})


def poll_positions(context: ExecutionContext) -> Dict[str, object]:
    """Poll open positions."""

    return context.gateway.search_positions({"accountId": context.account_id})


class GatewayStreamingSession:
    """Manage live SignalR hubs bound to a :class:`ProjectXGateway`."""

    def __init__(
        self,
        gateway: ProjectXGateway,
        *,
        market_hub_path: str = "marketHub",
        user_hub_path: str = "userHub",
        reconnect_delay: float = 0.0,
        connection_builder: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._gateway = gateway
        self._reconnect_delay = reconnect_delay
        self._builder_factory = connection_builder or _require_signalr_builder
        self.market = MarketHub(self, market_hub_path)
        self.user = UserHub(self, user_hub_path)

    def close(self) -> None:
        """Close both hub connections."""

        self.market.close()
        self.user.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return self._gateway.auth_headers()

    def _build_url(self, hub_path: str) -> str:
        return _join_url(self._gateway.base_url, hub_path)

    def _connection_builder(self) -> Any:
        builder_cls = self._builder_factory()
        return builder_cls()

    def _refresh_token(self) -> None:
        self._gateway.auth_headers()


class MarketHub:
    """High level helpers for market hub subscriptions."""

    def __init__(self, session: GatewayStreamingSession, hub_path: str) -> None:
        self._session = session
        self._hub = _StreamingHub(session, hub_path)

    def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[str, Any], None],
        *,
        event: str = "ticker_update",
        subscribe_method: str = "SubscribeTicker",
        unsubscribe_method: str = "UnsubscribeTicker",
    ) -> HubSubscriptionHandle:
        return self._hub.subscribe(
            event=event,
            subscribe_method=subscribe_method,
            unsubscribe_method=unsubscribe_method,
            args=(symbol,),
            callback=callback,
            prefix_args=(symbol,),
        )

    def subscribe_bars(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[str, str, Any], None],
        *,
        event: str = "bar_update",
        subscribe_method: str = "SubscribeBars",
        unsubscribe_method: str = "UnsubscribeBars",
    ) -> HubSubscriptionHandle:
        return self._hub.subscribe(
            event=event,
            subscribe_method=subscribe_method,
            unsubscribe_method=unsubscribe_method,
            args=(symbol, timeframe),
            callback=callback,
            prefix_args=(symbol, timeframe),
        )

    def subscribe_depth(
        self,
        symbol: str,
        callback: Callable[[str, Any], None],
        *,
        event: str = "depth_update",
        subscribe_method: str = "SubscribeDepth",
        unsubscribe_method: str = "UnsubscribeDepth",
    ) -> HubSubscriptionHandle:
        return self._hub.subscribe(
            event=event,
            subscribe_method=subscribe_method,
            unsubscribe_method=unsubscribe_method,
            args=(symbol,),
            callback=callback,
            prefix_args=(symbol,),
        )

    def close(self) -> None:
        self._hub.close()


class UserHub:
    """Helpers for user/account hub subscriptions."""

    def __init__(self, session: GatewayStreamingSession, hub_path: str) -> None:
        self._session = session
        self._hub = _StreamingHub(session, hub_path)

    def subscribe_accounts(
        self,
        callback: Callable[[Any], None],
        account_id: Optional[str] = None,
        *,
        event: str = "account_update",
        subscribe_method: str = "SubscribeAccounts",
        unsubscribe_method: str = "UnsubscribeAccounts",
    ) -> HubSubscriptionHandle:
        args = tuple(filter(None, (account_id,)))
        prefix: Tuple[str, ...] = (account_id,) if account_id else tuple()
        return self._hub.subscribe(
            event=event,
            subscribe_method=subscribe_method,
            unsubscribe_method=unsubscribe_method,
            args=args,
            callback=callback,
            prefix_args=prefix,
        )

    def subscribe_orders(
        self,
        account_id: str,
        callback: Callable[[str, Any], None],
        *,
        event: str = "order_update",
        subscribe_method: str = "SubscribeOrders",
        unsubscribe_method: str = "UnsubscribeOrders",
    ) -> HubSubscriptionHandle:
        return self._account_subscription(
            event,
            subscribe_method,
            unsubscribe_method,
            account_id,
            callback,
        )

    def subscribe_positions(
        self,
        account_id: str,
        callback: Callable[[str, Any], None],
        *,
        event: str = "position_update",
        subscribe_method: str = "SubscribePositions",
        unsubscribe_method: str = "UnsubscribePositions",
    ) -> HubSubscriptionHandle:
        return self._account_subscription(
            event,
            subscribe_method,
            unsubscribe_method,
            account_id,
            callback,
        )

    def subscribe_trades(
        self,
        account_id: str,
        callback: Callable[[str, Any], None],
        *,
        event: str = "trade_update",
        subscribe_method: str = "SubscribeTrades",
        unsubscribe_method: str = "UnsubscribeTrades",
    ) -> HubSubscriptionHandle:
        return self._account_subscription(
            event,
            subscribe_method,
            unsubscribe_method,
            account_id,
            callback,
        )

    def close(self) -> None:
        self._hub.close()

    def _account_subscription(
        self,
        event: str,
        subscribe_method: str,
        unsubscribe_method: str,
        account_id: str,
        callback: Callable[[str, Any], None],
    ) -> HubSubscriptionHandle:
        return self._hub.subscribe(
            event=event,
            subscribe_method=subscribe_method,
            unsubscribe_method=unsubscribe_method,
            args=(account_id,),
            callback=callback,
            prefix_args=(account_id,),
        )


class _StreamingHub:
    """Manage a single SignalR hub connection and its subscriptions."""

    def __init__(self, session: GatewayStreamingSession, hub_path: str) -> None:
        self._session = session
        self._hub_path = hub_path
        self._lock = threading.RLock()
        self._connection: Any | None = None
        self._dispatchers: Dict[str, Callable[[Any], None]] = {}
        self._tokens: Dict[str, Any] = {}
        self._subscriptions: Dict[str, list[_Subscription]] = defaultdict(list)
        self._stopping = False
        self._reconnect_delay = session._reconnect_delay
        self._pending_timer: threading.Timer | None = None

    def subscribe(
        self,
        *,
        event: str,
        subscribe_method: Optional[str],
        unsubscribe_method: Optional[str],
        args: Sequence[Any],
        callback: Callable[..., None],
        prefix_args: Sequence[Any] = (),
        adapter: Optional[Callable[[Any], Any]] = None,
    ) -> HubSubscriptionHandle:
        subscription = _Subscription(
            hub=self,
            event=event,
            subscribe_method=subscribe_method,
            unsubscribe_method=unsubscribe_method,
            args=tuple(args),
            callback=callback,
            prefix_args=tuple(prefix_args),
            adapter=adapter,
        )

        connection: Any
        should_start = False
        with self._lock:
            connection, should_start = self._ensure_connection_locked()
            self._register_dispatcher_locked(event, connection)
            self._subscriptions[event].append(subscription)

        if should_start:
            _start_connection(connection)

        if subscribe_method:
            connection.send(subscribe_method, list(args))
            subscription.synced = True

        return HubSubscriptionHandle(subscription)

    def close(self) -> None:
        with self._lock:
            self._stopping = True
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            connection = self._connection
            self._connection = None
            self._dispatchers.clear()
            self._tokens.clear()
            self._subscriptions.clear()
        if connection is not None and hasattr(connection, "stop"):
            connection.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_connection_locked(self) -> tuple[Any, bool]:
        if self._connection is not None:
            return self._connection, False

        builder = self._session._connection_builder()
        headers = self._session._headers()
        options: MutableMapping[str, Any] = _merge_options({}, headers)
        connection = builder.with_url(
            self._session._build_url(self._hub_path), options=options
        ).build()

        if hasattr(connection, "on_open"):
            connection.on_open(self._handle_open)
        if hasattr(connection, "on_close"):
            connection.on_close(self._handle_close)

        for event, dispatcher in self._dispatchers.items():
            _, token = _register_handler(connection, event, dispatcher)
            self._tokens[event] = token

        self._connection = connection
        return connection, True

    def _register_dispatcher_locked(self, event: str, connection: Any) -> None:
        if event in self._dispatchers:
            if event not in self._tokens:
                handler = self._dispatchers[event]
                _, token = _register_handler(connection, event, handler)
                self._tokens[event] = token
            return

        def _dispatcher(message: Any) -> None:
            self._fan_out(event, message)

        self._dispatchers[event] = _dispatcher
        _, token = _register_handler(connection, event, _dispatcher)
        self._tokens[event] = token

    def _fan_out(self, event: str, message: Any) -> None:
        for subscription in list(self._subscriptions.get(event, [])):
            subscription.deliver(message)

    def _handle_open(self) -> None:
        subscriptions: Iterable[_Subscription]
        with self._lock:
            subscriptions = [
                subscription
                for subs in self._subscriptions.values()
                for subscription in subs
                if not subscription.synced and subscription.subscribe_method
            ]
        if not subscriptions:
            return
        connection = self._connection
        if connection is None:
            return
        for subscription in subscriptions:
            connection.send(subscription.subscribe_method, list(subscription.args))
        with self._lock:
            for subscription in subscriptions:
                subscription.synced = True

    def _handle_close(self, error: Any = None) -> None:
        with self._lock:
            if self._stopping:
                return
            for subscriptions in self._subscriptions.values():
                for subscription in subscriptions:
                    subscription.synced = False
            self._connection = None
            self._tokens.clear()
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
        self._session._refresh_token()
        if self._reconnect_delay <= 0:
            self._ensure_connection()
        else:
            timer = threading.Timer(self._reconnect_delay, self._ensure_connection)
            timer.daemon = True
            with self._lock:
                if self._stopping:
                    return
                self._pending_timer = timer
            timer.start()

    def _ensure_connection(self) -> None:
        connection: Any
        should_start = False
        with self._lock:
            if self._stopping:
                return
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            connection, should_start = self._ensure_connection_locked()
        if should_start:
            _start_connection(connection)

    def _remove_subscription(self, subscription: "_Subscription") -> None:
        connection: Any | None = None
        send_unsubscribe = False
        unsubscribe_method: Optional[str] = None
        args: Tuple[Any, ...] = ()

        with self._lock:
            subscriptions = self._subscriptions.get(subscription.event)
            if not subscriptions or subscription not in subscriptions:
                return
            subscriptions.remove(subscription)
            if not subscriptions:
                self._subscriptions.pop(subscription.event, None)
                handler = self._dispatchers.pop(subscription.event, None)
                token = self._tokens.pop(subscription.event, None)
                connection = self._connection
                if connection is not None and handler is not None:
                    _remove_listener(connection, subscription.event, handler, token)
            if subscription.unsubscribe_method and subscription.synced:
                connection = connection or self._connection
                send_unsubscribe = True
                unsubscribe_method = subscription.unsubscribe_method
                args = subscription.args

        if send_unsubscribe and connection is not None and unsubscribe_method:
            connection.send(unsubscribe_method, list(args))


class _Subscription:
    """Track individual subscriber state."""

    def __init__(
        self,
        *,
        hub: _StreamingHub,
        event: str,
        subscribe_method: Optional[str],
        unsubscribe_method: Optional[str],
        args: Tuple[Any, ...],
        callback: Callable[..., None],
        prefix_args: Tuple[Any, ...],
        adapter: Optional[Callable[[Any], Any]],
    ) -> None:
        self._hub = hub
        self.event = event
        self.subscribe_method = subscribe_method
        self.unsubscribe_method = unsubscribe_method
        self.args = tuple(arg for arg in args if arg is not None)
        self.callback = callback
        self.prefix_args = prefix_args
        self.adapter = adapter
        self.synced = False
        self._closed = False

    def deliver(self, message: Any) -> None:
        if self._closed:
            return
        payload = _extract_payload(message)
        if self.adapter is not None:
            payload = self.adapter(payload)
        self.callback(*self.prefix_args, payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub._remove_subscription(self)


def _require_signalr_builder():
    try:
        from signalrcore.hub_connection_builder import HubConnectionBuilder
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(_STREAMING_IMPORT_MESSAGE) from exc

    return HubConnectionBuilder


def _start_connection(connection: Any) -> None:
    if hasattr(connection, "start"):
        connection.start()


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


def _register_handler(connection: Any, event: str, handler: Callable[[Any], None]) -> tuple[Callable[[Any], None], Any]:
    token = connection.on(event, handler)
    return handler, token


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
    "GatewayStreamingSession",
    "HubSubscriptionHandle",
    "MarketHub",
    "UserHub",
    "poll_open_orders",
    "poll_positions",
]

