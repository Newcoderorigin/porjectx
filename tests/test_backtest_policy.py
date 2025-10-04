"""Backtest policy tests ensuring thresholded decisions improve hit rate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from toptek.backtest import run as backtest_run
from toptek.data import io
from toptek.features import build_features
from toptek.model import train as train_module


def _write_config(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": 42,
        "data": {"bars_parquet": "data/demo_bars.parquet"},
        "model": {"type": "logistic", "calibration": "isotonic"},
        "threshold": {"min_coverage": 0.30, "min_expected_value": -0.05},
        "economics": {"avg_win": 100.0, "avg_loss": 70.0},
        "fees": {"per_trade": 1.0, "slippage": 0.5},
        "output": {"models_dir": "models", "cache_dir": "cache"},
    }
    path = config_dir / "config.yml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def test_threshold_backtest_improves_metrics(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    var_dir = tmp_path / "var"
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    for directory in (data_dir, var_dir, models_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    original_paths_cls = io.IOPaths

    def _mock_paths() -> io.IOPaths:
        return original_paths_cls(root=tmp_path, var=var_dir, data=data_dir, db=var_dir / "toptek.db")

    monkeypatch.setattr(io, "IOPaths", _mock_paths)
    monkeypatch.setattr(io, "DATA_DIR", data_dir)
    monkeypatch.setattr(io, "VAR_DIR", var_dir)

    conn = io.connect(var_dir / "toptek.db")
    io.run_migrations(conn)
    io.load_demo_data(conn, rows=200)
    conn.close()

    config_path = _write_config(tmp_path)
    train_config = train_module.load_config(config_path)
    train_config = train_module.TrainConfig(
        seed=train_config.seed,
        data_path=data_dir / "demo_bars.parquet",
        models_dir=models_dir,
        cache_dir=cache_dir,
        method=train_config.method,
        min_coverage=train_config.min_coverage,
        min_expected_value=train_config.min_expected_value,
        avg_win=train_config.avg_win,
        avg_loss=train_config.avg_loss,
        fees=train_config.fees,
    )

    df_bars = pd.read_parquet(data_dir / "demo_bars.parquet")
    if "ts" in df_bars.columns:
        df_bars = df_bars.set_index("ts")
    df_bars.index = pd.to_datetime(df_bars.index)
    bundle = build_features(df_bars, cache_dir=cache_dir)
    calibrator, metrics = train_module.train_bundle(bundle, train_config)
    train_module.save_artifacts(calibrator, metrics, bundle.meta, train_config)

    report = backtest_run.run(config_path, use_calibrated=True, optimize_threshold=True)

    assert report["coverage_after"] >= 0.30
    assert report["hit_rate_after"] >= report["hit_rate_before"] + 0.05
    assert report["expectancy_after"] >= 0.95 * report["expectancy_before"]
    assert report["brier_after"] <= report["brier_before"] - 0.02
