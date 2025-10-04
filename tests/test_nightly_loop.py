from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from toptek.features import FeatureBundle  # noqa: E402


def test_save_report_creates_timestamped_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from toptek.loops import learn

    class _Frozen(datetime):
        @classmethod
        def utcnow(cls) -> "_Frozen":
            return cls(2024, 4, 15, 3, 21, 0)

    monkeypatch.setattr(learn, "datetime", _Frozen)
    payload = {"status": "ok", "metrics": {"hit_rate": 0.62}}
    path = learn._save_report(payload, tmp_path)

    assert path.name == "learning_run_20240415.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload
    assert path.parent == tmp_path / "reports"


def test_gather_user_data_merges_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    from toptek.loops import learn

    trades = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2024-01-01T10:00:00Z"]),
            "symbol": ["ES"],
            "qty": [1],
        }
    )
    predictions = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01T10:00:00Z"]),
            "prob_up": [0.7],
            "prob_down": [0.3],
        }
    )

    def _fake_read_sql(query: str, _conn: object) -> pd.DataFrame:
        if "trades" in query:
            return trades.copy()
        if "model_predictions" in query:
            return predictions.copy()
        raise AssertionError(query)

    monkeypatch.setattr(learn.pd, "read_sql_query", _fake_read_sql)

    result = learn._gather_user_data(object())
    assert "prob_up" in result.columns
    assert result.loc[0, "prob_up"] == pytest.approx(0.7)


@dataclass
class _StubCalibrator:
    predict_proba: Any


def test_main_writes_nightly_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from toptek.loops import learn
    from toptek.model.train import TrainConfig

    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    models_dir.mkdir()
    cache_dir.mkdir()

    config = TrainConfig(
        seed=7,
        data_path=tmp_path / "bars.parquet",
        models_dir=models_dir,
        cache_dir=cache_dir,
        method="isotonic",
        min_coverage=0.25,
        min_expected_value=0.0,
        avg_win=120.0,
        avg_loss=80.0,
        fees=2.5,
    )

    args_namespace = type("Args", (), {"config": str(tmp_path / "loop.yml")})()
    monkeypatch.setattr(learn, "_parse_args", lambda: args_namespace)
    monkeypatch.setattr(learn, "_load_loop_config", lambda _path: config)

    class _Conn:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = _Conn()
    monkeypatch.setattr(learn.io, "connect", lambda: connection)
    monkeypatch.setattr(learn.io, "run_migrations", lambda _conn: None)
    monkeypatch.setattr(learn, "_gather_user_data", lambda _conn: pd.DataFrame())

    feature_bundle = FeatureBundle(
        X=np.ones((8, 2)),
        y=np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
        meta={
            "feature_names": ["f1", "f2"],
            "valid_index": ["2024-01-01T00:00:00"],
            "mask": [1] * 8,
            "cache_key": "deadbeef",
            "dropped_rows": 0,
        },
    )
    monkeypatch.setattr(
        learn, "build_features", lambda _bars, cache_dir=None: feature_bundle
    )
    monkeypatch.setattr(
        learn.pd, "read_parquet", lambda _path: pd.DataFrame({"close": np.arange(8)})
    )

    calibrator = _StubCalibrator(predict_proba=lambda _: np.linspace(0.1, 0.9, 8))

    def _fake_train_bundle(bundle, cfg):
        return calibrator, {"hit_rate": 0.6}

    monkeypatch.setattr(learn, "train_bundle", _fake_train_bundle)

    def _fake_save_artifacts(_cal, metrics, _meta, _cfg):
        model_path = models_dir / "model.pkl"
        model_path.write_bytes(b"model")
        card_path = models_dir / "model_card.json"
        card_path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
        return {"model": model_path, "calibrator": model_path, "card": card_path}

    monkeypatch.setattr(learn, "save_artifacts", _fake_save_artifacts)

    class _Frozen(datetime):
        @classmethod
        def utcnow(cls) -> "_Frozen":
            return cls(2024, 4, 16, 6, 0, 0)

    monkeypatch.setattr(learn, "datetime", _Frozen)

    learn.main()
    captured = json.loads(capsys.readouterr().out)

    assert captured["status"] == "completed"
    report_path = Path(captured["report"])
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["metrics"]["hit_rate"] == pytest.approx(0.6)
    assert connection.closed is True
