"""Golden snapshot tests for the AI server tool endpoints."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import random

pytest_plugins = ["tests.ai_server.test_app"]

try:  # pragma: no cover - prefer FastAPI's TestClient when available
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - fallback to local stub
    from toptek.ai_server._fastapi_stub import TestClient

SNAPSHOT_DIR = Path(__file__).with_name("snapshots")
_ORIGINAL_RANDOM = random.Random


def _load_snapshot(name: str) -> dict:
    path = SNAPSHOT_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def _stable_random(seed: int = 1234):
    def _factory(_seed: int) -> random.Random:
        return _ORIGINAL_RANDOM(seed)

    with patch("toptek.ai_server.tools.random.Random", _factory):
        yield


def test_backtest_endpoint_snapshot(app_fixture) -> None:
    app, _ = app_fixture
    with _stable_random(), TestClient(app) as client:
        response = client.post(
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
        assert response.status_code == 200
        assert response.json() == _load_snapshot("backtest.json")


def test_walkforward_endpoint_snapshot(app_fixture) -> None:
    app, _ = app_fixture
    with _stable_random(seed=4321), TestClient(app) as client:
        response = client.post(
            "/tools/walkforward", json={"config_path": "configs/config.yml"}
        )
        assert response.status_code == 200
        assert response.json() == _load_snapshot("walkforward.json")


def test_metrics_endpoint_snapshot(app_fixture) -> None:
    app, _ = app_fixture
    with _stable_random(seed=1337), TestClient(app) as client:
        response = client.post("/tools/metrics", json={"pnl": [0.01, -0.02, 0.01]})
        assert response.status_code == 200
        assert response.json() == _load_snapshot("metrics.json")
