"""Configuration loader for the Auto AI Server."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore[import]


@dataclass(frozen=True)
class AISettings:
    """Runtime settings for the AI server and LM Studio integration."""

    base_url: str
    port: int
    auto_start: bool
    poll_interval_seconds: float
    poll_timeout_seconds: float
    default_model: Optional[str]
    default_role: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "port": self.port,
            "auto_start": self.auto_start,
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "default_model": self.default_model,
            "default_role": self.default_role,
        }


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as boolean")


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(config_path: Path | str = Path("configs/ai.yml")) -> AISettings:
    """Construct :class:`AISettings` from YAML + environment overrides."""

    if sys.version_info >= (3, 12):
        raise RuntimeError(
            "Python 3.12+ is not supported by the pinned scientific stack; "
            "use Python 3.10 or 3.11."
        )

    path = Path(config_path)
    config = _load_config_file(path)

    defaults = {
        "base_url": "http://localhost:1234/v1",
        "port": 1234,
        "auto_start": True,
        "poll_interval_seconds": 0.5,
        "poll_timeout_seconds": 10.0,
        "default_model": None,
        "default_role": "You are the Quant Co-Pilot. Provide actionable, risk-aware answers.",
    }
    defaults.update(config)

    env = os.environ

    base_url = env.get("LMSTUDIO_BASE_URL", defaults["base_url"])

    try:
        port = int(env.get("LMSTUDIO_PORT", defaults["port"]))
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError("LMSTUDIO_PORT must be an integer") from exc

    default_model = env.get("LMSTUDIO_MODEL", defaults["default_model"] or None)

    default_role = env.get("AI_DEFAULT_ROLE", defaults["default_role"])

    auto_start = _coerce_bool(env.get("LMSTUDIO_AUTO_START"), defaults["auto_start"])

    try:
        poll_interval = float(defaults["poll_interval_seconds"])
        poll_timeout = float(defaults["poll_timeout_seconds"])
    except (TypeError, ValueError) as exc:  # pragma: no cover - config validation
        raise ValueError("Poll interval/timeout must be numeric") from exc

    return AISettings(
        base_url=base_url.rstrip("/"),
        port=port,
        auto_start=auto_start,
        poll_interval_seconds=poll_interval,
        poll_timeout_seconds=poll_timeout,
        default_model=default_model,
        default_role=default_role,
    )


__all__ = ["AISettings", "load_settings"]
