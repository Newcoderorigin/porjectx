import pytest

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - fallback for offline stubs
    from toptek.ai_server._fastapi_stub import TestClient

from toptek.ai_server.app import create_app
from toptek.ai_server.config import AISettings
from toptek.ai_server.lmstudio import ModelStats
from toptek.ai_server.router import ModelRouter


class FakeLMStudioClient:
    def __init__(self, models):
        self._models = models
        self.chat_payloads = []

    async def health(self):
        return True

    async def list_models(self):
        return self._models

    async def chat_stream(self, payload):
        self.chat_payloads.append(payload)
        yield 'data: {"choices": [{"delta": {"content": "hello"}}]}'
        yield "data: [DONE]"

    async def model_usage_snapshot(self, model_id):
        return {"usage": {"tokens_per_second": 100.0, "ttft": 80.0}}


class FakeProcess:
    def __init__(self):
        self.started = False

    async def ensure_running(self):
        self.started = True

    def terminate(self):  # pragma: no cover - not triggered in tests
        self.started = False


@pytest.fixture
def app_fixture():
    settings = AISettings(
        base_url="http://localhost:1234/v1",
        port=1234,
        auto_start=False,
        poll_interval_seconds=0.1,
        poll_timeout_seconds=1.0,
        default_model=None,
        default_role="system",
    )
    models = [
        ModelStats(
            model_id="demo",
            owned_by="local",
            max_context=4096,
            supports_tools=True,
            tokens_per_second=50.0,
            ttft=120.0,
            description="demo model",
            raw={"modalities": ["text"]},
        )
    ]
    client = FakeLMStudioClient(models)
    router = ModelRouter(settings)
    process = FakeProcess()
    app = create_app(settings=settings, client=client, router=router, process=process)
    return app, client


def test_health_and_models_endpoints(app_fixture):
    app, _ = app_fixture
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        payload = client.get("/models").json()
        assert payload["models"][0]["id"] == "demo"


def test_chat_stream_and_router_selection(app_fixture):
    app, fake_client = app_fixture
    with TestClient(app) as client:
        response = client.post(
            "/chat", json={"messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
        chunks = list(response.iter_lines())
        assert any("hello" in chunk for chunk in chunks)
        assert fake_client.chat_payloads[0]["model"] == "demo"


def test_tool_endpoints(app_fixture):
    app, _ = app_fixture
    with TestClient(app) as client:
        backtest = client.post(
            "/tools/backtest",
            json={
                "symbol": "ES",
                "start": "2020-01-01",
                "end": "2020-12-31",
                "costs": 0.1,
                "slippage": 0.05,
                "vol_target": 0.15,
            },
        )
        assert backtest.status_code == 200
        walk = client.post(
            "/tools/walkforward", json={"config_path": "configs/config.yml"}
        )
        assert walk.status_code == 200
        metrics = client.post("/tools/metrics", json={"pnl": [0.01, -0.02, 0.01]})
        assert metrics.status_code == 200
