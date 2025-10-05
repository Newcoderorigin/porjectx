"""FastAPI route registration for ProjectX gateway helpers."""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional

try:  # pragma: no cover - prefer real FastAPI when available
    from fastapi import HTTPException, Request, WebSocket
except ModuleNotFoundError:  # pragma: no cover
    from toptek.ai_server._fastapi_stub import HTTPException  # type: ignore

    Request = Any  # type: ignore
    WebSocket = Any  # type: ignore

from toptek.core import live as live_module

from .models import GatewaySettings, load_gateway_settings

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from toptek.core.gateway import ProjectXGateway

__all__ = ["RateLimiter", "register_gateway_routes"]


@dataclass
class RateLimiter:
    """Async rate limiter enforcing a minimum interval between gateway calls."""

    min_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._last_call = 0.0

    async def __aenter__(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        await self._lock.acquire()
        now = time.monotonic()
        delay = self.min_interval_seconds - (now - self._last_call)
        if delay > 0:
            await asyncio.sleep(delay)
        self._last_call = time.monotonic()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._lock is None:
            raise RuntimeError("RateLimiter lock missing during release")
        self._lock.release()
        return False


def _require_api_key(headers: Mapping[str, str], expected: str) -> None:
    provided = headers.get("x-api-key") or headers.get("X-API-Key")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _normalize_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    raise HTTPException(status_code=400, detail="Payload must be an object")


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _numeric_field(raw: Mapping[str, Any], *candidates: str) -> float:
    for key in candidates:
        if key in raw and raw[key] is not None:
            return _coerce_float(raw[key])
    return 0.0


def _summarize_accounts(response: Mapping[str, Any]) -> Dict[str, Any]:
    accounts = response.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        raise HTTPException(status_code=502, detail="Gateway returned no accounts")

    summaries = []
    for raw in accounts:
        if not isinstance(raw, Mapping):
            continue
        summary = {
            "id": raw.get("accountId") or raw.get("id"),
            "name": raw.get("accountName") or raw.get("name"),
            "available": _numeric_field(raw, "available", "cashAvailable"),
            "allocated": _numeric_field(raw, "allocated", "marginUsed"),
            "profit": _numeric_field(raw, "profit", "profitLoss"),
            "equity": _numeric_field(raw, "equity", "netLiquidation"),
        }
        summaries.append(summary)

    if not summaries:
        raise HTTPException(
            status_code=502, detail="Gateway returned invalid account payload"
        )

    return {"accounts": summaries}


def _headers_from_request(request: Request | None) -> Mapping[str, str]:
    if request is None:
        return {}
    headers = getattr(request, "headers", None)
    if headers is None:
        return {}
    if isinstance(headers, Mapping):
        return headers
    return {key: value for key, value in dict(headers).items()}


async def _call_gateway(
    func: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
    limiter: RateLimiter | None,
) -> Dict[str, Any]:
    if limiter is None:
        return await asyncio.to_thread(func, payload)
    async with limiter:
        return await asyncio.to_thread(func, payload)


def register_gateway_routes(
    app: Any,
    *,
    gateway_settings: GatewaySettings | None = None,
    gateway: "ProjectXGateway" | None = None,
    rate_limiter: RateLimiter | None = None,
    live=live_module,
) -> None:
    """Register ProjectX gateway routes on ``app``."""

    settings = gateway_settings or load_gateway_settings()
    limiter = rate_limiter or RateLimiter()
    if gateway is None:
        from toptek.core.gateway import ProjectXGateway as _ProjectXGateway

        client = _ProjectXGateway(
            settings.base_url, settings.username, settings.api_key
        )
    else:
        client = gateway

    def _register_post(path: str, method_name: str) -> None:
        method = getattr(client, method_name)

        @app.post(f"/gateway{path}")
        async def _endpoint(
            payload: Optional[Dict[str, Any]] = None,
            request: Request | None = None,
        ) -> Dict[str, Any]:
            headers = _headers_from_request(request)
            _require_api_key(headers, settings.api_key)
            normalized = _normalize_payload(payload)
            return await _call_gateway(method, normalized, limiter)

    route_map = {
        "/accounts/search": "search_accounts",
        "/contracts/search": "search_contracts",
        "/contracts/by-id": "contract_by_id",
        "/contracts/available": "contract_available",
        "/history/bars": "retrieve_bars",
        "/orders/place": "place_order",
        "/orders/modify": "modify_order",
        "/orders/cancel": "cancel_order",
        "/orders/search": "search_orders",
        "/orders/search-open": "search_open_orders",
        "/positions/search": "search_positions",
        "/positions/close": "close_position",
        "/positions/partial-close": "partial_close_position",
        "/trades/search": "search_trades",
    }

    for path, method_name in route_map.items():
        _register_post(path, method_name)

    @app.get("/gateway/healthz")
    async def gateway_health(request: Request | None = None) -> Dict[str, Any]:
        headers = _headers_from_request(request)
        _require_api_key(headers, settings.api_key)
        report: Dict[str, Any] = {
            "rest": False,
            "market_hub": False,
            "user_hub": False,
            "details": {},
        }

        try:
            await _call_gateway(lambda _: client.login(), {}, limiter)
            report["rest"] = True
        except Exception as exc:  # pragma: no cover - network errors
            report["details"]["rest"] = str(exc)

        async def _probe_hub(base: str, path: str, label: str) -> None:
            try:
                handle = await asyncio.to_thread(
                    live.connect_market_hub,
                    base,
                    hub_path=path,
                    auto_start=False,
                )
                await asyncio.to_thread(handle.close)
            except Exception as exc:  # pragma: no cover - optional dependency/network
                report["details"][label] = str(exc)
            else:
                report[label] = True

        await _probe_hub(settings.market_hub_base, settings.market_hub_path, "market_hub")
        await _probe_hub(settings.user_hub_base, settings.user_hub_path, "user_hub")
        return report

    @app.get("/gateway/account")
    async def gateway_account(request: Request | None = None) -> Dict[str, Any]:
        headers = _headers_from_request(request)
        _require_api_key(headers, settings.api_key)
        raw = await _call_gateway(client.search_accounts, {}, limiter)
        if not isinstance(raw, Mapping):
            raise HTTPException(
                status_code=502, detail="Gateway returned unexpected account payload"
            )
        return _summarize_accounts(raw)

    async def _reject_websocket(websocket: WebSocket, *, reason: str) -> None:
        await websocket.close(code=1008, reason=reason)

    async def _authorize_ws(websocket: WebSocket) -> bool:
        headers = getattr(websocket, "headers", {})
        value = headers.get("x-api-key") or headers.get("X-API-Key")
        if value and secrets.compare_digest(value, settings.api_key):
            return True
        token = getattr(websocket, "query_params", {}).get("api_key")  # type: ignore[attr-defined]
        if token and secrets.compare_digest(token, settings.api_key):
            return True
        await _reject_websocket(websocket, reason="Invalid API key")
        return False

    if hasattr(app, "websocket"):
        @app.websocket("/gateway/ws/market")
        async def market_ws(websocket: WebSocket) -> None:  # pragma: no cover - integration only
            if not await _authorize_ws(websocket):
                return
            await websocket.accept()
            await websocket.send_json({"detail": "Market hub relay not yet implemented"})
            await websocket.close()

        @app.websocket("/gateway/ws/user")
        async def user_ws(websocket: WebSocket) -> None:  # pragma: no cover - integration only
            if not await _authorize_ws(websocket):
                return
            await websocket.accept()
            await websocket.send_json({"detail": "User hub relay not yet implemented"})
            await websocket.close()

    if not hasattr(app.state, "gateway_client"):
        app.state.gateway_client = client  # type: ignore[attr-defined]
        app.state.gateway_settings = settings  # type: ignore[attr-defined]
        app.state.gateway_rate_limiter = limiter  # type: ignore[attr-defined]
