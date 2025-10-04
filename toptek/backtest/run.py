"""Backtest runner using calibrated models and threshold optimisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml

from toptek.features import build_features
from toptek.model.threshold import opt_threshold


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_model(models_dir: Path) -> joblib:
    calibrator_path = models_dir / "calibrator.pkl"
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator missing at {calibrator_path}")
    return joblib.load(calibrator_path)


def _performance_metrics(returns: np.ndarray) -> Dict[str, float]:
    if returns.size == 0:
        return {"hit_rate": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "expectancy": 0.0}
    hit_rate = float((returns > 0).mean())
    expectancy = float(np.mean(returns))
    std = np.std(returns)
    sharpe = float(expectancy / std * np.sqrt(252)) if std > 1e-9 else 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(np.max(drawdown)) if drawdown.size else 0.0
    return {
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "expectancy": expectancy,
    }


def _apply_policy(probs: np.ndarray, returns: np.ndarray, tau: float) -> Dict[str, float]:
    mask = probs >= tau
    selected_returns = returns[mask]
    coverage = float(mask.mean())
    metrics = _performance_metrics(selected_returns)
    metrics["coverage"] = coverage
    metrics["threshold"] = tau
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated backtests")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--use-calibrated", action="store_true", help="Use calibrated probabilities")
    parser.add_argument("--optimize-threshold", action="store_true", help="Optimise tau for hit-rate")
    return parser.parse_args()


def run(config_path: Path, *, use_calibrated: bool, optimize_threshold: bool) -> Dict[str, float]:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    project_root = config_path.parent.parent
    bars_path = project_root / config["data"]["bars_parquet"]
    models_dir = project_root / config["output"]["models_dir"]

    bars = pd.read_parquet(bars_path)
    bundle = build_features(bars, cache_dir=project_root / config["output"].get("cache_dir", ".cache"))

    calibrator = _load_model(models_dir)
    raw_probs = calibrator.base_estimator.predict_proba(bundle.X)[:, 1]
    calibrated_probs = calibrator.predict_proba(bundle.X)[:, 1]
    probs = calibrated_probs if use_calibrated else raw_probs

    forward_returns = bars["close"].pct_change().shift(-1)
    forward_returns = forward_returns.loc[bundle.meta["valid_index"]].to_numpy(dtype=float)
    labels = (forward_returns > 0).astype(int)

    baseline = _apply_policy(probs, forward_returns, tau=0.5)

    if optimize_threshold:
        tau, curve = opt_threshold(
            probs,
            labels,
            min_coverage=config["threshold"].get("min_coverage", 0.30),
            min_expected_value=config["threshold"].get("min_expected_value", 0.0),
        )
        optimised = _apply_policy(probs, forward_returns, tau)
    else:
        tau = 0.5
        curve = []
        optimised = baseline

    brier_before = float(np.mean((raw_probs - labels) ** 2))
    brier_after = float(np.mean((calibrated_probs - labels) ** 2))

    report = {
        "hit_rate_before": baseline["hit_rate"],
        "hit_rate_after": optimised["hit_rate"],
        "coverage_before": baseline["coverage"],
        "coverage_after": optimised["coverage"],
        "expectancy_before": baseline["expectancy"],
        "expectancy_after": optimised["expectancy"],
        "sharpe": optimised["sharpe"],
        "max_drawdown": optimised["max_drawdown"],
        "threshold": tau,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "curve": curve,
    }
    return report


def main() -> None:
    args = _parse_args()
    report = run(Path(args.config), use_calibrated=args.use_calibrated, optimize_threshold=args.optimize_threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
