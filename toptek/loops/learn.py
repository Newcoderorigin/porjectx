"""Nightly learning loop that incorporates user data and retrains models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from toptek.data import io
from toptek.features import build_features
from toptek.model.train import TrainConfig, load_config as load_train_config, save_artifacts, train_bundle


def _load_loop_config(path: Path) -> TrainConfig:
    return load_train_config(path)


def _gather_user_data(conn) -> pd.DataFrame:
    trades = pd.read_sql_query("SELECT * FROM trades", conn)
    predictions = pd.read_sql_query("SELECT * FROM model_predictions", conn)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])
    predictions["ts"] = pd.to_datetime(predictions["ts"])
    merged = trades.merge(predictions, left_on="entry_ts", right_on="ts", how="left", suffixes=("_trade", "_pred"))
    return merged


def _save_report(report: dict, root: Path) -> Path:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"learning_run_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightly learning loop")
    parser.add_argument("--config", required=True, help="Training config path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = _load_loop_config(config_path)
    root = config_path.parent.parent

    conn = io.connect()
    io.run_migrations(conn)
    user_data = _gather_user_data(conn)
    conn.close()

    bars = pd.read_parquet(config.data_path)
    bundle = build_features(bars, cache_dir=config.cache_dir)
    calibrator, metrics = train_bundle(bundle, config)
    artifacts = save_artifacts(calibrator, metrics, bundle.meta, config)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "user_rows": int(len(user_data)),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    report_path = _save_report(report, root)
    print(json.dumps({"status": "completed", "report": str(report_path), "metrics": metrics}))


if __name__ == "__main__":
    main()
