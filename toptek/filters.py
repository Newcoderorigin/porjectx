"""Utility filters for UI log redaction."""

from __future__ import annotations

import re
from typing import Iterable


_ALLOWLIST = {
    "CPU",
    "GPU",
    "HTTP",
    "HTTPS",
    "API",
    "TK",
}

_SECRET_PATTERNS: Iterable[re.Pattern[str]] = (
    re.compile(r"(?i)(api|access|secret|token|key)[=:\s]+[A-Za-z0-9\-_]{8,}"),
    re.compile(r"(?i)(bearer\s+[A-Za-z0-9\-_]{8,})"),
)

_TICKER_PATTERN = re.compile(r"\b([A-Z]{2,6}=F)\b")


def redact(value: str) -> str:
    """Redact potential secrets and futures tickers from *value*."""

    text = value
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("{REDACTED}", text)

    def _replace_ticker(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in _ALLOWLIST:
            return token
        return "{" + token + "}"

    text = _TICKER_PATTERN.sub(_replace_ticker, text)
    return text
