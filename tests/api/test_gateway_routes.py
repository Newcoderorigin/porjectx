from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from toptek.ai_server._fastapi_stub import FastAPI, HTTPException
from toptek.api.models import GatewaySettings, RequiredGatewayEnv, load_gateway_settings
from toptek.api.routes_gateway import RateLimiter, register_gateway_routes


class DummyGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.account_response: dict[str, object] = {
            "accounts": [
                {
                    "accountId": "U12345",
                    "accountName": "Main",
                    "available": 125000.0,
                    "allocated": 50000.0,
                    "profit": 2500.5,
                    "equity": 175000.5,
                }
            ]
        }

    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("place_order", dict(payload)))
        return {"status": "ok"}

    def login(self) -> None:
        self.calls.append(("login", {}))

    def search_accounts(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("search_accounts", dict(payload)))
        return dict(self.account_response)

    def __getattr__(self, item: str):
        def _call(payload: dict[str, object]) -> dict[str, object]:
            self.calls.append((item, dict(payload)))
            return {"status": "ok"}

        return _call


class DummyLimiter:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> None:
        self.entered += 1

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.exited += 1
        return False


class LiveStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.closed: list[tuple[str, str]] = []

    def connect_market_hub(self, base_url: str, *, hub_path: str, auto_start: bool = True):
        self.calls.append((base_url, hub_path))

        stub = self

        class _Handle:
            def close(self_inner) -> None:
                stub.closed.append((base_url, hub_path))

        return _Handle()


def test_rate_limiter_lazy_lock_instantiation() -> None:
    limiter = RateLimiter()

    async def _use_limiter() -> None:
        async with limiter:
            pass

    asyncio.run(_use_limiter())
    assert limiter._lock is not None


def _settings() -> GatewaySettings:
    return GatewaySettings(
        base_url="https://gateway-api.example.com/api",
        username="bot",
        api_key="top-secret",
        market_hub_base="https://gateway-rtc.example.com",
        market_hub_path="hubs/market",
        user_hub_base="https://gateway-rtc.example.com",
        user_hub_path="hubs/user",
    )


def test_load_gateway_settings_requires_all_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in RequiredGatewayEnv:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError):
        load_gateway_settings()

    monkeypatch.setenv("PX_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("PX_MARKET_HUB", "https://rtc.example.com/hubs/market")
    monkeypatch.setenv("PX_USER_HUB", "https://rtc.example.com/hubs/user")
    monkeypatch.setenv("PX_USERNAME", "bot")
    monkeypatch.setenv("PX_API_KEY", "secret")

    settings = load_gateway_settings()
    assert settings.market_hub_base == "https://rtc.example.com"
    assert settings.market_hub_path == "hubs/market"
    assert settings.user_hub_base == "https://rtc.example.com"
    assert settings.user_hub_path == "hubs/user"


def test_gateway_routes_enforce_api_key_and_rate_limit() -> None:
    app = FastAPI(title="test")
    gateway = DummyGateway()
    limiter = DummyLimiter()
    settings = _settings()

    register_gateway_routes(
        app,
        gateway_settings=settings,
        gateway=gateway,
        rate_limiter=limiter,
    )

    handler = app.get_route("POST", "/gateway/orders/place")
    with pytest.raises(HTTPException):
        asyncio.run(handler({"symbol": "ES"}))

    request = SimpleNamespace(headers={"X-API-Key": settings.api_key})
    result = asyncio.run(handler({"symbol": "ES"}, request))

    assert result == {"status": "ok"}
    assert gateway.calls == [("place_order", {"symbol": "ES"})]
    assert limiter.entered == 1
    assert limiter.exited == 1


def test_gateway_health_reports_components(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI(title="test")
    gateway = DummyGateway()
    limiter = DummyLimiter()
    live = LiveStub()
    settings = _settings()

    register_gateway_routes(
        app,
        gateway_settings=settings,
        gateway=gateway,
        rate_limiter=limiter,
        live=live,
    )

    handler = app.get_route("GET", "/gateway/healthz")
    request = SimpleNamespace(headers={"X-API-Key": settings.api_key})
    report = asyncio.run(handler(request))

    assert report["rest"] is True
    assert report["market_hub"] is True
    assert report["user_hub"] is True
    assert report["details"] == {}
    assert ("https://gateway-rtc.example.com", "hubs/market") in live.calls
    assert ("https://gateway-rtc.example.com", "hubs/user") in live.calls
    assert limiter.entered >= 1


def test_gateway_health_captures_failures() -> None:
    app = FastAPI(title="test")
    gateway = DummyGateway()
    settings = _settings()

    class FailingGateway(DummyGateway):
        def login(self) -> None:
            raise RuntimeError("login failed")

    class FailingLive(LiveStub):
        def connect_market_hub(self, *args, **kwargs):
            raise RuntimeError("hub down")

    failing_gateway = FailingGateway()
    failing_live = FailingLive()

    register_gateway_routes(
        app,
        gateway_settings=settings,
        gateway=failing_gateway,
        live=failing_live,
    )

    handler = app.get_route("GET", "/gateway/healthz")
    request = SimpleNamespace(headers={"X-API-Key": settings.api_key})
    report = asyncio.run(handler(request))

    assert report["rest"] is False
    assert report["market_hub"] is False
    assert report["user_hub"] is False
    assert "login failed" in report["details"]["rest"]
    assert "hub down" in report["details"]["market_hub"]
    assert "hub down" in report["details"]["user_hub"]


def test_gateway_account_snapshot() -> None:
    app = FastAPI(title="test")
    gateway = DummyGateway()
    limiter = DummyLimiter()
    settings = _settings()

    register_gateway_routes(
        app,
        gateway_settings=settings,
        gateway=gateway,
        rate_limiter=limiter,
    )

    handler = app.get_route("GET", "/gateway/account")
    request = SimpleNamespace(headers={"X-API-Key": settings.api_key})
    snapshot = asyncio.run(handler(request))

    assert snapshot == {
        "accounts": [
            {
                "id": "U12345",
                "name": "Main",
                "available": 125000.0,
                "allocated": 50000.0,
                "profit": 2500.5,
                "equity": 175000.5,
            }
        ]
    }
    assert ("search_accounts", {}) in gateway.calls
    assert limiter.entered >= 1


def test_gateway_account_requires_api_key() -> None:
    app = FastAPI(title="test")
    gateway = DummyGateway()
    settings = _settings()

    register_gateway_routes(
        app,
        gateway_settings=settings,
        gateway=gateway,
    )

    handler = app.get_route("GET", "/gateway/account")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(handler(SimpleNamespace(headers={})))

    assert excinfo.value.status_code == 401
    assert ("search_accounts", {}) not in gateway.calls
