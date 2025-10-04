"""Minimal synchronous client for LM Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional
import urllib.error
import urllib.request


class HTTPError(RuntimeError):
    """Error raised when an HTTP request fails."""


@dataclass
class Model:
    model_id: str
    owned_by: Optional[str] = None
    max_context: Optional[int] = None
    description: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "Model":
        context = payload.get("metadata", {}).get("context_length")
        if context is None:
            context = payload.get("metadata", {}).get("context_window")
        try:
            context_window = int(context) if context is not None else None
        except (TypeError, ValueError):
            context_window = None
        return cls(
            model_id=str(payload.get("id")),
            owned_by=payload.get("owned_by"),
            max_context=context_window,
            description=payload.get("description")
            or payload.get("metadata", {}).get("display_name"),
        )


class _URLLibResponse:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.status = getattr(response, "status", getattr(response, "code", None))

    def read(self) -> bytes:
        data = self._response.read()
        self._response.close()
        return data

    def iter_lines(self) -> Iterator[str]:
        try:
            iterator: Iterable[bytes] = self._response
        except TypeError as exc:  # pragma: no cover - defensive
            raise HTTPError("Response is not iterable") from exc
        try:
            for chunk in iterator:
                yield chunk.decode("utf-8").rstrip("\n")
        finally:
            self._response.close()


class _URLLibTransport:
    def __init__(self, opener: Optional[urllib.request.OpenerDirector] = None) -> None:
        self._opener = opener or urllib.request.build_opener()

    def request(
        self,
        method: str,
        url: str,
        *,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> _URLLibResponse:
        request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            response = self._opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network failure
            raise HTTPError(f"HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            raise HTTPError(str(exc.reason)) from exc
        return _URLLibResponse(response)


class LMStudioClient:
    """Blocking LM Studio client using stdlib ``urllib`` for portability."""

    def __init__(
        self,
        settings: Dict[str, Any],
        *,
        transport: Optional[_URLLibTransport] = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = settings.get("base_url", "http://localhost:1234/v1").rstrip("/")
        self._api_key = settings.get("api_key", "")
        self._model = settings.get("model")
        self._timeout = timeout or float(settings.get("timeout_s", 30))
        self._transport = transport or _URLLibTransport()

    def list_models(self) -> list[Model]:
        response = self._request("GET", "/models")
        data = self._decode_json(response)
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        return [Model.from_payload(item) for item in models if isinstance(item, dict)]

    def health(self) -> bool:
        try:
            self._request("GET", "/models")
        except HTTPError:
            return False
        return True

    def chat_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        request_payload = dict(payload)
        if self._model and "model" not in request_payload:
            request_payload["model"] = self._model
        response = self._request("POST", "/chat/completions", payload=request_payload, stream=True)
        for line in response.iter_lines():
            if not line:
                continue
            yield line

    # ------------------------------------------------------------------ internals
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> _URLLibResponse:
        url = f"{self._base_url}{path}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        response = self._transport.request(
            method,
            url,
            data=data,
            headers=self._headers(),
            timeout=self._timeout,
        )
        status = response.status or 0
        if status >= 400:
            raise HTTPError(f"HTTP {status}")
        if stream:
            return response
        # For non-streaming calls we still return a response wrapper to allow JSON decode.
        return response

    def _decode_json(self, response: _URLLibResponse) -> Dict[str, Any]:
        body = response.read()
        if not body:
            return {}
        if isinstance(body, str):
            text = body
        else:
            text = body.decode("utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise HTTPError("Invalid JSON response") from exc


__all__ = ["HTTPError", "LMStudioClient", "Model"]
