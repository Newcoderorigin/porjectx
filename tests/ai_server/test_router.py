from toptek.ai_server.config import AISettings
from toptek.ai_server.lmstudio import ModelStats
from toptek.ai_server.router import ChatTask, ModelRouter


def _settings(default_model: str | None = None) -> AISettings:
    return AISettings(
        base_url="http://localhost:1234/v1",
        port=1234,
        auto_start=False,
        poll_interval_seconds=0.1,
        poll_timeout_seconds=1.0,
        default_model=default_model,
        default_role="system",
    )


def _model(model_id: str, *, tools: bool, context: int, tps: float = 0.0) -> ModelStats:
    return ModelStats(
        model_id=model_id,
        owned_by="local",
        max_context=context,
        supports_tools=tools,
        tokens_per_second=tps,
        ttft=None,
        description=None,
        raw={"modalities": ["text"]},
    )


def test_router_prefers_tool_capable_model():
    router = ModelRouter(_settings())
    router.register_models(
        [
            _model("fast", tools=False, context=4096, tps=120.0),
            _model("tool", tools=True, context=4096, tps=90.0),
        ]
    )

    task = ChatTask(requires_tools=True, expected_tokens=None, reasoning_required=False)
    decision = router.select(task)
    assert decision.model_id == "tool"


def test_router_respects_manual_override():
    router = ModelRouter(_settings(default_model="fast"))
    router.register_models(
        [
            _model("fast", tools=False, context=4096, tps=120.0),
            _model("deep", tools=True, context=8192, tps=80.0),
        ]
    )
    router.manual_override("deep")
    decision = router.select(
        ChatTask(requires_tools=False, expected_tokens=512, reasoning_required=False)
    )
    assert decision.model_id == "deep"
    assert router.current_selection() == "deep"


def test_router_updates_telemetry():
    router = ModelRouter(_settings())
    router.register_models([_model("alpha", tools=False, context=4096, tps=50.0)])
    router.record_usage("alpha", {"tokens_per_second": 42.0, "ttft": 120})
    description = router.describe("alpha")
    assert description["tokens_per_second"] == 42.0
    assert description["ttft_ms"] == 120.0
