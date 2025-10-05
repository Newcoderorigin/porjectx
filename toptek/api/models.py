"""Gateway settings and payload models for the ProjectX API surface."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple
from urllib.parse import urlsplit

__all__ = ["GatewaySettings", "load_gateway_settings", "RequiredGatewayEnv"]

RequiredGatewayEnv = (
    "PX_BASE_URL",
    "PX_MARKET_HUB",
    "PX_USER_HUB",
    "PX_USERNAME",
    "PX_API_KEY",
)


@dataclass(frozen=True)
class GatewaySettings:
    """Concrete configuration required to talk to the ProjectX gateway."""

    base_url: str
    username: str
    api_key: str
    market_hub_base: str
    market_hub_path: str
    user_hub_base: str
    user_hub_path: str

    def as_dict(self) -> dict[str, str]:
        """Return a serialisable snapshot of the gateway configuration."""

        return {
            "base_url": self.base_url,
            "username": self.username,
            "market_hub_base": self.market_hub_base,
            "market_hub_path": self.market_hub_path,
            "user_hub_base": self.user_hub_base,
            "user_hub_path": self.user_hub_path,
        }


def _split_hub_url(raw_url: str, *, field: str) -> Tuple[str, str]:
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"{field} must include protocol and host")
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or ""
    return base.rstrip("/"), path.lstrip("/")


def _require_env(env: Mapping[str, str], key: str) -> str:
    raw = env.get(key, "").strip()
    if not raw:
        raise RuntimeError(
            "Missing LIVE trading environment variable: {0}. "
            "Populate it in your .env or shell to enable gateway access.".format(key)
        )
    return raw


def load_gateway_settings(env: Mapping[str, str] | None = None) -> GatewaySettings:
    """Validate the LIVE gateway environment variables and return settings."""

    mapping = env or os.environ
    missing: Sequence[str] = [key for key in RequiredGatewayEnv if not mapping.get(key)]
    if missing:
        raise RuntimeError(
            "Missing LIVE trading environment variables: {0}".format(
                ", ".join(sorted(missing))
            )
        )

    base_url = _require_env(mapping, "PX_BASE_URL").rstrip("/")
    username = _require_env(mapping, "PX_USERNAME")
    api_key = _require_env(mapping, "PX_API_KEY")
    market_base, market_path = _split_hub_url(
        _require_env(mapping, "PX_MARKET_HUB"), field="PX_MARKET_HUB"
    )
    user_base, user_path = _split_hub_url(
        _require_env(mapping, "PX_USER_HUB"), field="PX_USER_HUB"
    )

    return GatewaySettings(
        base_url=base_url,
        username=username,
        api_key=api_key,
        market_hub_base=market_base,
        market_hub_path=market_path,
        user_hub_base=user_base,
        user_hub_path=user_path,
    )
