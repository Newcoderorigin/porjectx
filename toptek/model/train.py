"""Training entry point with calibration and reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from toptek.features import FeatureBundle, build_features


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    data_path: Path
    models_dir: Path
    cache_dir: Path
    method: str
    min_coverage: float
    min_expected_value: float
    avg_win: float
    avg_loss: float
    fees: float


def load_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    project_root = path.parent.parent
    data_path = project_root / raw["data"]["bars_parquet"]
    models_dir = project_root / raw["output"]["models_dir"]
    cache_dir = project_root / raw["output"].get("cache_dir", ".cache")
    method = raw["model"].get("calibration", "isotonic")
    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        data_path=data_path,
        models_dir=models_dir,
        cache_dir=cache_dir,
        method=method,
        min_coverage=float(raw["threshold"].get("min_coverage", 0.30)),
        min_expected_value=float(raw["threshold"].get("min_expected_value", 0.0)),
        avg_win=float(raw["economics"].get("avg_win", 1.0)),
        avg_loss=float(raw["economics"].get("avg_loss", 1.0)),
        fees=float(raw["fees"].get("per_trade", 0.0) + raw["fees"].get("slippage", 0.0)),
    )


def _load_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bars parquet not found: {path}")
    return pd.read_parquet(path)


def _expected_value(hit_rate: float, config: TrainConfig) -> float:
    gross = hit_rate * config.avg_win - (1 - hit_rate) * config.avg_loss
    return gross - config.fees


def train_bundle(bundle: FeatureBundle, config: TrainConfig) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    X = bundle.X
    y = bundle.y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=True,
        stratify=y,
        random_state=config.seed,
    )

    base_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.seed)
    base_model.fit(X_train, y_train)

    try:
        calibrator = CalibratedClassifierCV(base_estimator=base_model, method=config.method, cv="prefit")
        calibrator.fit(X_test, y_test)
    except ValueError:
        fallback = "sigmoid" if config.method != "sigmoid" else "isotonic"
        calibrator = CalibratedClassifierCV(base_estimator=base_model, method=fallback, cv="prefit")
        calibrator.fit(X_test, y_test)

    probs = calibrator.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_test, probs)),
        "auroc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "hit_rate": float((preds == y_test).mean()),
    }
    metrics["expected_value"] = _expected_value(metrics["precision"], config)

    return calibrator, metrics


def save_artifacts(calibrator: CalibratedClassifierCV, metrics: Dict[str, Any], meta: Dict[str, Any], config: TrainConfig) -> Dict[str, Path]:
    config.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.models_dir / "model.pkl"
    calibrator_path = config.models_dir / "calibrator.pkl"
    joblib.dump(calibrator.base_estimator, model_path)
    joblib.dump(calibrator, calibrator_path)

    card = {
        "versions": {
            "model": "1.0.0",
            "data_cache": meta.get("cache_key"),
        },
        "metrics": metrics,
        "features_hash": meta.get("cache_key"),
        "feature_names": meta.get("feature_names"),
    }
    card_path = config.models_dir / "model_card.json"
    with card_path.open("w", encoding="utf-8") as handle:
        json.dump(card, handle, indent=2)

    return {"model": model_path, "calibrator": calibrator_path, "card": card_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibrated Toptek models")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(Path(args.config).resolve())
    np.random.seed(config.seed)

    bars = _load_bars(config.data_path)
    bundle = build_features(bars, cache_dir=config.cache_dir)
    calibrator, metrics = train_bundle(bundle, config)
    paths = save_artifacts(calibrator, metrics, bundle.meta, config)

    report = {
        "models": {k: str(v) for k, v in paths.items()},
        "metrics": metrics,
        "dropped_rows": bundle.meta.get("dropped_rows"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
