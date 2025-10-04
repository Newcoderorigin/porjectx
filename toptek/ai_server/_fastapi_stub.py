"""Fallback FastAPI-compatible stubs for offline test execution."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    Optional,
    Tuple,
    cast,
)

RouteKey = Tuple[str, str]


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str | None = None) -> None:
        super().__init__(detail or "")
        self.status_code = status_code
        self.detail = detail or ""


@dataclass
class JSONResponse:
    content: Any
    status_code: int = 200

    def json(self) -> Any:
        return self.content


@dataclass
class HTMLResponse:
    content: str
    status_code: int = 200

    def text(self) -> str:
        return self.content


class StreamingResponse:
    def __init__(
        self, iterator: AsyncGenerator[bytes, None], media_type: str = "text/plain"
    ) -> None:
        self.iterator = iterator
        self.media_type = media_type
        self.status_code = 200
        self._cached: Optional[list[bytes]] = None

    def _ensure_cached(self) -> list[bytes]:
        if self._cached is None:

            async def _collect() -> list[bytes]:
                items: list[bytes] = []
                async for chunk in self.iterator:
                    if isinstance(chunk, str):
                        items.append(chunk.encode("utf-8"))
                    else:
                        items.append(chunk)
                return items

            self._cached = asyncio.run(_collect())
        return self._cached

    def iter_lines(self) -> Iterable[str]:
        for chunk in self._ensure_cached():
            text = chunk.decode("utf-8")
            for line in text.splitlines():
                yield line


class FastAPI:
    def __init__(
        self,
        *,
        title: str,
        lifespan: Optional[
            Callable[["FastAPI"], AbstractAsyncContextManager[None]]
        ] = None,
    ) -> None:
        self.title = title
        self._routes: Dict[RouteKey, Callable[..., Awaitable[Any]]] = {}
        self._lifespan_factory = lifespan
        self._lifespan_cm: Optional[AbstractAsyncContextManager[None]] = None
        self.state = SimpleNamespace()

    def _register(
        self, method: str, path: str, func: Callable[..., Awaitable[Any]]
    ) -> None:
        self._routes[(method.upper(), path)] = func

    def get(self, path: str, response_class: Optional[type] = None) -> Callable:
        def decorator(
            func: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            self._register("GET", path, func)
            return func

        return decorator

    def post(self, path: str, response_class: Optional[type] = None) -> Callable:
        def decorator(
            func: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            self._register("POST", path, func)
            return func

        return decorator

    async def _enter_lifespan(self) -> None:
        if self._lifespan_factory is not None and self._lifespan_cm is None:
            self._lifespan_cm = self._lifespan_factory(self)
            await self._lifespan_cm.__aenter__()

    async def _exit_lifespan(self) -> None:
        if self._lifespan_cm is not None:
            await self._lifespan_cm.__aexit__(None, None, None)
            self._lifespan_cm = None

    def get_route(self, method: str, path: str) -> Callable[..., Awaitable[Any]]:
        return self._routes[(method.upper(), path)]


class _ClientResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        if isinstance(payload, (JSONResponse, StreamingResponse, HTMLResponse)):
            self.status_code = payload.status_code
        else:
            self.status_code = 200

    def json(self) -> Any:
        if isinstance(self._payload, JSONResponse):
            return self._payload.json()
        if isinstance(self._payload, dict):
            return self._payload
        raise TypeError("Response is not JSON compatible")

    def text(self) -> str:
        if isinstance(self._payload, HTMLResponse):
            return self._payload.text()
        raise TypeError("Response is not text compatible")

    def iter_lines(self) -> Iterable[str]:
        if isinstance(self._payload, StreamingResponse):
            return self._payload.iter_lines()
        raise TypeError("Response is not streaming")


class TestClient:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def __enter__(self) -> "TestClient":
        asyncio.run(self._app._enter_lifespan())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        asyncio.run(self._app._exit_lifespan())

    def get(self, path: str) -> _ClientResponse:
        handler = self._app.get_route("GET", path)
        result = asyncio.run(cast(Coroutine[Any, Any, Any], handler()))
        return _ClientResponse(result)

    def post(
        self, path: str, json: Optional[Dict[str, Any]] = None, stream: bool = False
    ) -> _ClientResponse:
        handler = self._app.get_route("POST", path)
        if json is None:
            payload = asyncio.run(cast(Coroutine[Any, Any, Any], handler()))
        else:
            payload = asyncio.run(cast(Coroutine[Any, Any, Any], handler(json)))
        return _ClientResponse(payload)


__all__ = [
    "FastAPI",
    "HTTPException",
    "HTMLResponse",
    "JSONResponse",
    "StreamingResponse",
    "TestClient",
]
