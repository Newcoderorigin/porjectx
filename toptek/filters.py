"""Utility filters for scrubbing sensitive text before transmission."""

from __future__ import annotations

import re
from typing import Any, Dict

_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_TICKER_PATTERN = re.compile(r"\b(?=.*[=0-9])[A-Z0-9=]{1,10}\b")


def redact_text(value: str) -> str:
    """Redact ticker symbols and IPv4 addresses from *value*.

    The implementation intentionally keeps the transformation simple so that
    redaction is deterministic and easy to reason about for tests.
    """

    if not value:
        return value
    redacted = _IP_PATTERN.sub("[REDACTED_IP]", value)
    redacted = _TICKER_PATTERN.sub(_mask_ticker, redacted)
    return redacted


def _mask_ticker(match: re.Match[str]) -> str:
    token = match.group(0)
    if token.startswith("[REDACTED_"):
        return token
    return "[REDACTED_TICKER]"


def redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a recursively redacted copy of *payload*."""

    def _scrub(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {
                child_key: _scrub(child_val, key=child_key)
                for child_key, child_val in value.items()
            }
        if isinstance(value, list):
            return [_scrub(item, key=key) for item in value]
        if isinstance(value, str):
            if key in {"symbol", "account_id", "route"}:
                return "[REDACTED_TICKER]"
            return redact_text(value)
        return value

    return _scrub(dict(payload))


__all__ = ["redact_payload", "redact_text"]
