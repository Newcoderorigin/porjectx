"""Minimal httpx stub for offline tests."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict


class HTTPError(Exception):
    pass


class Response:
    def __init__(self, status_code: int = 200, json_data: Dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP status {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json


class AsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - placeholder
        pass

    async def get(self, *args: Any, **kwargs: Any) -> Response:
        raise HTTPError("httpx not available in stub mode")

    async def aclose(self) -> None:  # pragma: no cover - placeholder
        return None

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Response]:  # pragma: no cover - placeholder
        raise HTTPError("Streaming unsupported in stub mode")


class Timeout:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - placeholder
        pass


__all__ = ["AsyncClient", "HTTPError", "Response", "Timeout"]
