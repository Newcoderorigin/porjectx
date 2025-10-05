"""FastAPI application exposing the Auto AI Server."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

try:  # pragma: no cover - prefer the real FastAPI implementation when available
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - offline fallback
    from ._fastapi_stub import (
        FastAPI,
        HTTPException,
        HTMLResponse,
        JSONResponse,
        StreamingResponse,
    )

    FASTAPI_AVAILABLE = False

try:  # pragma: no cover - prefer real pydantic when available
    from pydantic import BaseModel, Field, root_validator
except ModuleNotFoundError:  # pragma: no cover
    from ._pydantic_stub import BaseModel, Field, root_validator

from toptek.api import load_gateway_settings, register_gateway_routes

from .config import AISettings, load_settings
from .lmstudio import LMStudioClient
from .process import LMStudioNotInstalledError, LMStudioProcessManager
from .router import ModelRouter, infer_task
from .tools import (
    BacktestRequest,
    metrics_report,
    run_backtest_tool,
    walk_forward_report,
)

LOGGER = logging.getLogger(__name__)
UI_DIR = Path(__file__).parent / "ui"
UI_INDEX = UI_DIR / "index.html"


class ChatMessage(BaseModel):
    role: str
    content: str

    @root_validator(pre=True)
    def strip_content(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        content = values.get("content")
        if isinstance(content, str):
            values["content"] = content.strip()
        return values


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: float = 0.7
    tools: Optional[List[Dict[str, Any]]] = None
    system_role: Optional[str] = None

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        messages = data.get("messages", [])
        converted: List[ChatMessage] = []
        for entry in messages:
            if isinstance(entry, ChatMessage):
                converted.append(entry)
            elif isinstance(entry, dict):
                converted.append(ChatMessage(**entry))
            else:
                raise TypeError("Invalid message payload")
        data["messages"] = converted
        super().__init__(**data)

    @root_validator
    def ensure_messages(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        messages = values.get("messages")
        if not messages:
            raise ValueError("messages cannot be empty")
        return values

    def build_payload(self, *, model_id: str, default_role: str) -> Dict[str, Any]:
        payload_messages: List[Dict[str, str]] = []
        has_system = any(msg.role == "system" for msg in self.messages)
        if not has_system:
            role_text = self.system_role or default_role
            payload_messages.append({"role": "system", "content": role_text})
        payload_messages.extend(
            {"role": msg.role, "content": msg.content} for msg in self.messages
        )
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": payload_messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.tools:
            payload["tools"] = self.tools
        return payload


class BacktestPayload(BaseModel):
    symbol: str
    start: str
    end: str
    costs: float = Field(ge=0.0)
    slippage: float = Field(ge=0.0)
    vol_target: float = Field(gt=0.0)


class WalkForwardPayload(BaseModel):
    config_path: str


class MetricsPayload(BaseModel):
    pnl: List[float]


if TYPE_CHECKING:  # pragma: no cover - imports for typing only
    from toptek.core.gateway import ProjectXGateway


class AppState:
    def __init__(
        self,
        settings: AISettings,
        client: LMStudioClient,
        process: LMStudioProcessManager,
        router: ModelRouter,
        gateway: "ProjectXGateway",
    ) -> None:
        self.settings = settings
        self.client = client
        self.process = process
        self.router = router
        self.startup_error: Optional[str] = None
        self.gateway = gateway


@lru_cache(maxsize=1)
def _load_ui() -> str:
    if not UI_INDEX.exists():
        raise FileNotFoundError(f"Missing UI entrypoint at {UI_INDEX}")
    return UI_INDEX.read_text(encoding="utf-8")


async def _stream_chat(
    *,
    client: LMStudioClient,
    router: ModelRouter,
    settings: AISettings,
    request: ChatRequest,
) -> AsyncGenerator[bytes, None]:
    models = await client.list_models()
    router.register_models(models)
    task = infer_task(
        tools=request.tools,
        max_tokens=request.max_tokens,
        system_role=request.system_role,
        manual_model=request.model,
    )
    decision = router.select(task)
    payload = request.build_payload(
        model_id=decision.model_id, default_role=settings.default_role
    )

    try:
        async for chunk in client.chat_stream(payload):
            text = chunk if chunk.startswith("data:") else f"data: {chunk}"
            yield (text + "\n\n").encode("utf-8")
            try:
                data = text.split("data:", 1)[1].strip()
                if data and data != "[DONE]":
                    obj = json.loads(data)
                    usage = obj.get("usage")
                    if usage:
                        router.record_usage(decision.model_id, usage)
            except (IndexError, json.JSONDecodeError):
                continue
    except Exception as exc:  # pragma: no cover - network/runtime failures
        LOGGER.error("Chat stream failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    snapshot = await client.model_usage_snapshot(decision.model_id)
    if snapshot and "usage" in snapshot:
        router.record_usage(decision.model_id, snapshot["usage"])


def create_app(
    *,
    settings: Optional[AISettings] = None,
    client: Optional[LMStudioClient] = None,
    process: Optional[LMStudioProcessManager] = None,
    router: Optional[ModelRouter] = None,
) -> FastAPI:
    if settings is None:
        settings = load_settings()
    if client is None:
        client = LMStudioClient(settings)
    if router is None:
        router = ModelRouter(settings)
    if process is None:
        process = LMStudioProcessManager(settings, client)

    gateway_settings = load_gateway_settings()
    from toptek.core.gateway import ProjectXGateway

    gateway = ProjectXGateway(
        gateway_settings.base_url,
        gateway_settings.username,
        gateway_settings.api_key,
    )

    state = AppState(settings, client, process, router, gateway)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await process.ensure_running()
        except LMStudioNotInstalledError as exc:
            LOGGER.warning("LM Studio auto-start failed: %s", exc)
            state.startup_error = str(exc)
        except TimeoutError as exc:
            LOGGER.error("LM Studio health check timed out: %s", exc)
            state.startup_error = str(exc)
        else:
            models = await client.list_models()
            router.register_models(models)
        yield
        process.terminate()
        await client.close()
        state.gateway.close()

    app = FastAPI(title="Toptek Auto AI Server", lifespan=lifespan)

    register_gateway_routes(
        app,
        gateway_settings=gateway_settings,
        gateway=gateway,
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_load_ui())

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        healthy = await client.health()
        quant_ok = True
        status = {
            "lmstudio": healthy,
            "quant": quant_ok,
            "startup_error": state.startup_error,
        }
        http_status = 200 if healthy and quant_ok else 503
        return JSONResponse(status, status_code=http_status)

    @app.get("/models")
    async def models_endpoint() -> Dict[str, Any]:
        models = await client.list_models()
        router.register_models(models)
        payload = []
        for model in models:
            telemetry = router.describe(model.model_id)
            payload.append(
                {
                    "id": model.model_id,
                    "owned_by": model.owned_by,
                    "supports_tools": model.supports_tools,
                    "max_context": model.max_context,
                    "tokens_per_second": telemetry["tokens_per_second"],
                    "ttft_ms": telemetry["ttft_ms"],
                }
            )
        return {
            "models": payload,
            "selected": router.current_selection(),
        }

    @app.post("/models/select")
    async def select_model(body: Dict[str, Any]) -> Dict[str, Any]:
        model_id = body.get("model_id")
        try:
            router.manual_override(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"selected": model_id}

    @app.post("/chat")
    async def chat_endpoint(request_body: Dict[str, Any]) -> StreamingResponse:
        chat = ChatRequest(**request_body)
        stream = _stream_chat(
            client=client,
            router=router,
            settings=settings,
            request=chat,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post("/tools/backtest")
    async def tool_backtest(payload: Dict[str, Any]) -> Dict[str, Any]:
        request_model = BacktestPayload(**payload)
        request = BacktestRequest(**request_model.dict())
        return run_backtest_tool(request)

    @app.post("/tools/walkforward")
    async def tool_walk_forward(payload: Dict[str, Any]) -> Dict[str, Any]:
        request = WalkForwardPayload(**payload)
        return walk_forward_report(request.config_path)

    @app.post("/tools/metrics")
    async def tool_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
        request = MetricsPayload(**payload)
        return metrics_report(request.pnl)

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
