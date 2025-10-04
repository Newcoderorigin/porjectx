"""Minimal LM Studio chat client using stdlib only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Mapping, MutableSequence, Sequence, Tuple


Message = Mapping[str, str]


@dataclass
class LMStudioConfig:
    """Configuration required to talk to an LM Studio compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: int = 30


class LMStudioClient:
    """Small helper that posts chat completions to LM Studio."""

    def __init__(self, config: LMStudioConfig) -> None:
        self._config = config

    @property
    def config(self) -> LMStudioConfig:
        return self._config

    def chat(self, messages: Sequence[Message]) -> Tuple[str, float]:
        """Send *messages* to LM Studio and return (reply, latency_ms)."""

        payload = {
            "model": self._config.model,
            "messages": _normalise_messages(messages),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "top_p": self._config.top_p,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self._config.base_url.rstrip('/')}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
                raw = response.read()
        except urllib.error.URLError as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"Failed to reach LM Studio: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("LM Studio returned invalid JSON") from exc
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError("LM Studio returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LM Studio response missing content")
        return content, latency_ms


def _normalise_messages(messages: Sequence[Message]) -> List[Mapping[str, str]]:
    normalised: MutableSequence[Mapping[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip() or "user"
        content = str(message.get("content", ""))
        normalised.append({"role": role, "content": content})
    return list(normalised)


def build_client(config: Mapping[str, object]) -> LMStudioClient:
    """Build a client from a dictionary-like config section."""

    return LMStudioClient(
        LMStudioConfig(
            base_url=str(config.get("base_url", "")),
            api_key=str(config.get("api_key", "")),
            model=str(config.get("model", "")),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            max_tokens=int(config.get("max_tokens", 512)),
            timeout_s=int(config.get("timeout_s", 30)),
        )
    )
