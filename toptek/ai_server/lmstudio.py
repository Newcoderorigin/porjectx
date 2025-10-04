"""Async client for interacting with the LM Studio OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

try:  # pragma: no cover - prefer real httpx
    import httpx

    AsyncClient = httpx.AsyncClient
    HTTPError = httpx.HTTPError
    Timeout = httpx.Timeout
except ModuleNotFoundError:  # pragma: no cover - offline fallback
    from ._httpx_stub import AsyncClient, HTTPError, Timeout

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from httpx import AsyncClient as AsyncClientType
else:  # pragma: no cover - fallback type alias
    AsyncClientType = Any

from .config import AISettings


@dataclass
class ModelStats:
    """Rich metadata about an LM Studio model."""

    model_id: str
    owned_by: Optional[str]
    max_context: Optional[int]
    supports_tools: bool
    tokens_per_second: Optional[float]
    ttft: Optional[float]
    description: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_reasoning_model(self) -> bool:
        modality = self.raw.get("modalities") or []
        caps = self.raw.get("capabilities") or {}
        return "reasoning" in modality or bool(caps.get("reasoning"))

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ModelStats":
        metadata = payload.get("metadata") or {}
        capabilities = payload.get("capabilities") or {}
        performance = payload.get("performance") or {}

        def _extract_float(*keys: str) -> Optional[float]:
            for key in keys:
                value = performance.get(key, metadata.get(key, capabilities.get(key)))
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return None

        context = metadata.get("context_length") or metadata.get("context_window")
        try:
            context_window = int(context) if context is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            context_window = None

        supports_tools = bool(
            capabilities.get("tool_calls")
            or capabilities.get("functions")
            or metadata.get("tool_calls")
        )

        return cls(
            model_id=str(payload.get("id")),
            owned_by=payload.get("owned_by"),
            max_context=context_window,
            supports_tools=supports_tools,
            tokens_per_second=_extract_float("tokens_per_second", "throughput"),
            ttft=_extract_float("ttft", "time_to_first_token_ms"),
            description=payload.get("description") or metadata.get("display_name"),
            raw=payload,
        )


class LMStudioClient:
    """Thin async wrapper around LM Studio's OpenAI-compatible REST API."""

    def __init__(
        self,
        settings: AISettings,
        *,
        client: Optional[AsyncClientType] = None,
        request_timeout: float = 30.0,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or AsyncClient(
            base_url=settings.base_url,
            timeout=Timeout(request_timeout, connect=5.0),
        )
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return True
        except HTTPError:
            return False

    async def list_models(self) -> List[ModelStats]:
        response = await self._client.get("/models")
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):  # pragma: no cover - remote contract guard
            return []
        return [ModelStats.from_payload(item) for item in data]

    async def chat_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream chat completions back from LM Studio."""

        async with self._lock:  # serialise streaming calls for predictable ordering
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                timeout=None,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    yield line

    async def model_usage_snapshot(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest usage metrics exposed by LM Studio, if available."""

        response = await self._client.get(f"/models/{model_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError:  # pragma: no cover - contract guard
            return None


__all__ = ["LMStudioClient", "ModelStats"]
