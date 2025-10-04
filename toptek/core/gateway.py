"""ProjectX Gateway client for authenticated HTTP requests.

The client wraps POST endpoints exposed by ProjectX, handling JWT-based
authentication and providing typed helper methods for common operations such as
searching accounts, retrieving market data, and managing orders.

Example:
    >>> client = ProjectXGateway(base_url, username, api_key)
    >>> client.login()
    >>> accounts = client.search_accounts({"status": "Open"})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


AUTH_LOGIN = "/api/Auth/loginKey"
AUTH_VALIDATE = "/api/Auth/validate"


@dataclass
class GatewayConfig:
    """Configuration for connecting to ProjectX."""

    base_url: str
    username: str
    api_key: str


class GatewayError(Exception):
    """Base exception for gateway-related errors."""


class AuthenticationError(GatewayError):
    """Raised when authentication fails."""


class ProjectXGateway:
    """HTTP client for ProjectX Gateway endpoints."""

    def __init__(self, base_url: str, username: str, api_key: str) -> None:
        self._config = GatewayConfig(
            base_url=base_url.rstrip("/"), username=username, api_key=api_key
        )
        self._client = httpx.Client(base_url=self._config.base_url, timeout=20.0)
        self._token: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def login(self) -> None:
        """Authenticate and store the JWT token."""

        payload = {"userName": self._config.username, "apiKey": self._config.api_key}
        response = self._client.post(AUTH_LOGIN, json=payload)
        response.raise_for_status()
        data = response.json()
        token = data.get("token")
        if not token:
            raise AuthenticationError("ProjectX login did not return a token")
        self._token = token

    def _validate(self) -> None:
        if not self._token:
            self.login()
            return
        response = self._client.post(AUTH_VALIDATE, headers=self._headers)
        if response.status_code == 401:
            self.login()
        else:
            response.raise_for_status()

    @property
    def _headers(self) -> Dict[str, str]:
        if not self._token:
            raise AuthenticationError("ProjectX gateway requires login before use")
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request with automatic token validation."""

        if not self._token:
            self.login()
        response = self._client.post(endpoint, json=payload, headers=self._headers)
        if response.status_code == 401:
            self._validate()
            response = self._client.post(endpoint, json=payload, headers=self._headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - simple translation
            raise GatewayError(f"Gateway request failed: {exc.response.text}") from exc
        return response.json()

    # ------------------------------------------------------------------
    # Public API wrappers
    # ------------------------------------------------------------------
    def search_accounts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Account/search", payload)

    def search_contracts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Contract/search", payload)

    def contract_by_id(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Contract/searchById", payload)

    def contract_available(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Contract/available", payload)

    def retrieve_bars(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/History/retrieveBars", payload)

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Order/place", payload)

    def modify_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Order/modify", payload)

    def cancel_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Order/cancel", payload)

    def search_orders(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Order/search", payload)

    def search_open_orders(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Order/searchOpen", payload)

    def search_positions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Position/searchOpen", payload)

    def close_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Position/closeContract", payload)

    def partial_close_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Position/partialCloseContract", payload)

    def search_trades(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/Trade/search", payload)

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()


__all__ = ["ProjectXGateway", "GatewayError", "AuthenticationError", "GatewayConfig"]
