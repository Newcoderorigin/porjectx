import pytest

import asyncio

from toptek.ai_server.config import AISettings
from toptek.ai_server.process import (
    LMStudioNotInstalledError,
    LMStudioProcessManager,
)


class StubClient:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    async def health(self):
        result = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return result

    async def list_models(self):  # pragma: no cover - compatibility stub
        return []


class StubProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):  # pragma: no cover - simple stub
        return None

    def terminate(self):
        self.terminated = True


def test_ensure_running_starts_once(monkeypatch):
    settings = AISettings(
        base_url="http://localhost:1234/v1",
        port=1234,
        auto_start=True,
        poll_interval_seconds=0.01,
        poll_timeout_seconds=0.05,
        default_model=None,
        default_role="system",
    )

    client = StubClient([False, True])
    process = StubProcess()

    monkeypatch.setattr("toptek.ai_server.process.shutil.which", lambda _: "lms")
    monkeypatch.setattr(
        "toptek.ai_server.process.subprocess.Popen", lambda *args, **kwargs: process
    )

    manager = LMStudioProcessManager(settings, client)
    asyncio.run(manager.ensure_running())
    assert client.calls >= 2
    assert manager._state.started is True  # pylint: disable=protected-access


def test_ensure_running_raises_when_missing_cli(monkeypatch):
    settings = AISettings(
        base_url="http://localhost:1234/v1",
        port=1234,
        auto_start=True,
        poll_interval_seconds=0.01,
        poll_timeout_seconds=0.05,
        default_model=None,
        default_role="system",
    )
    client = StubClient([False])
    monkeypatch.setattr("toptek.ai_server.process.shutil.which", lambda _: None)

    manager = LMStudioProcessManager(settings, client)
    with pytest.raises(LMStudioNotInstalledError):
        asyncio.run(manager.ensure_running())
