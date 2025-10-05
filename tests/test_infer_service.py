from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sqlite3
import pickle

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from toptek.data import BarRecord, SQLiteBarFeed, run_migrations
from toptek.loops.infer import InferConfig, PredictionService
from toptek.model.metrics import MetricsAPI


class _StubCalibrator:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.tanh(X[:, 0] / 500.0)
        prob_up = 0.5 + 0.5 * logits
        prob_up = np.clip(prob_up, 0.01, 0.99)
        return np.column_stack([1.0 - prob_up, prob_up])


def _generate_bars(symbol: str, count: int) -> list[BarRecord]:
    start = datetime(2024, 1, 1, 9, 30)
    records: list[BarRecord] = []
    price = 4800.0
    for idx in range(count):
        ts = start + timedelta(minutes=5 * idx)
        drift = 0.5 + 0.05 * np.sin(idx / 3)
        close = price + drift
        records.append(
            BarRecord(
                symbol=symbol,
                ts=ts,
                open=price,
                high=price + 1.5,
                low=price - 1.5,
                close=close,
                volume=1200 + idx,
            )
        )
        price = close
    return records


def test_prediction_service_updates_metrics(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    calibrator = _StubCalibrator()
    with (models_dir / "calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle)
    card = {
        "feature_names": [
            "sma_10",
            "sma_20",
            "ema_12",
            "ema_26",
            "rsi_14",
        ],
        "features_hash": "integration",
        "versions": {"model": "stub-v1"},
    }
    (models_dir / "model_card.json").write_text(json.dumps(card), encoding="utf-8")

    conn = sqlite3.connect(tmp_path / "predictions.db")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)

    feed = SQLiteBarFeed(conn)
    symbol = "ESZ25"
    all_records = _generate_bars(symbol, 48)
    feed.insert(all_records[:32])

    config = InferConfig(models_dir=models_dir, symbols=(symbol,), decision_threshold=0.55, max_history=256)
    service = PredictionService(config, conn=conn)

    first_pass = service.run_once()
    assert symbol in first_pass and first_pass[symbol]

    predictions = pd.read_sql_query(
        "SELECT * FROM model_predictions WHERE symbol = ? ORDER BY ts", conn, params=(symbol,)
    )
    assert not predictions.empty
    unresolved_before = predictions["realized_ts"].isna().sum()
    assert unresolved_before >= 1

    metrics_api = MetricsAPI(conn, window=64)
    snapshot_before = metrics_api.payload([symbol])[symbol]
    assert snapshot_before["observations"] >= 1
    assert snapshot_before["hit_rate"] >= 0.0

    feed.insert(all_records[32:])
    second_pass = service.run_once()
    assert symbol in second_pass and second_pass[symbol]

    predictions_after = pd.read_sql_query(
        "SELECT * FROM model_predictions WHERE symbol = ? ORDER BY ts", conn, params=(symbol,)
    )
    assert len(predictions_after) > len(predictions)
    resolved = predictions_after[predictions_after["realized_ts"].notna()]
    assert len(resolved) >= len(predictions_after) - 1

    snapshot_after = metrics_api.payload([symbol])[symbol]
    assert snapshot_after["observations"] >= snapshot_before["observations"]
    assert snapshot_after["confidence"] == pytest.approx(
        float(predictions_after["prob_up"].iloc[-1])
    )
    assert snapshot_after["hit_rate"] >= 0.0

    service.close()
    conn.close()
