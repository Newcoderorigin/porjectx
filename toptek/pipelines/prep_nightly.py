"""Nightly preparation pipeline orchestrating ingest, training, and reports."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from toptek.databank import Bank, SyntheticBars
from toptek.features import build_features
from toptek.monitor import compute_drift_report


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    coverage: float
    expected_value: float


@dataclass(frozen=True)
class PipelineResult:
    daily_brief: Path
    threshold_curve: Path
    model_dir: Path
    metrics: dict[str, object]


def _optimise_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    min_coverage: float,
    min_ev: float,
    grid: Sequence[float] | None = None,
) -> ThresholdPoint:
    if probabilities.size != labels.size:
        raise ValueError("probabilities and labels must align")
    if probabilities.size == 0:
        raise ValueError("empty probabilities")
    if grid is None:
        grid = np.linspace(0.4, 0.9, 26)
    best: ThresholdPoint | None = None
    for tau in grid:
        mask = probabilities >= tau
        coverage = float(mask.mean())
        if coverage < min_coverage:
            continue
        if not np.any(mask):
            continue
        subset = probabilities[mask]
        ev = float(np.mean(2 * subset - 1))
        if ev < min_ev:
            continue
        candidate = ThresholdPoint(tau, coverage, ev)
        if best is None or candidate.expected_value > best.expected_value:
            best = candidate
    if best is None:
        idx = int(np.argmax(probabilities))
        tau = float(probabilities[idx])
        best = ThresholdPoint(tau, float(1 / probabilities.size), float(2 * tau - 1))
    return best


def _prepare_training_split(
    bundle,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = bundle.X
    y = bundle.y
    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    n = len(X)
    if n < 10:
        raise ValueError("Not enough samples to train")
    split = int(math.floor(n * (1 - validation_fraction)))
    if split <= 1 or split >= n:
        raise ValueError("Invalid split size")
    return X[:split], X[split:], y[:split], y[split:]


def _save_threshold_curve(path: Path, points: Sequence[ThresholdPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [p.threshold for p in points]
    coverage = [p.coverage for p in points]
    evs = [p.expected_value for p in points]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, coverage, label="Coverage", color="tab:blue")
    ax1.set_ylabel("Coverage", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(xs, evs, label="EV", color="tab:orange")
    ax2.set_ylabel("Expected Value", color="tab:orange")
    ax1.set_xlabel("Threshold")
    ax1.set_title("Threshold Optimisation Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=144)
    plt.close(fig)


def _build_model_card(
    *,
    version: str,
    training_rows: int,
    validation_rows: int,
    threshold: ThresholdPoint,
    brier: float,
    drift_overall: str,
) -> dict[str, object]:
    return {
        "version": version,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "threshold": threshold.threshold,
        "coverage": threshold.coverage,
        "expected_value": threshold.expected_value,
        "brier_score": brier,
        "drift_status": drift_overall,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(
    *,
    target_date: date,
    symbol: str = "ES",
    timeframe: str = "5m",
    days: int = 365,
    min_coverage: float = 0.1,
    min_ev: float = 0.0,
    bank_root: Path | str = Path("data/bank"),
    reports_root: Path | str = Path("reports"),
    models_root: Path | str = Path("models"),
) -> PipelineResult:
    bank = Bank(Path(bank_root), provider=SyntheticBars())
    bank.ingest(
        symbol,
        timeframe,
        days=days,
        end=datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc),
    )
    data = bank.read(symbol, timeframe)
    bundle = build_features(data)
    X_train, X_val, y_train, y_val = _prepare_training_split(bundle)
    base = LogisticRegression(max_iter=500, solver="lbfgs")
    base.fit(X_train, y_train)
    calibrator = CalibratedClassifierCV(
        base_estimator=base, method="sigmoid", cv="prefit"
    )
    calibrator.fit(X_val, y_val)
    proba_val = calibrator.predict_proba(X_val)[:, 1]
    threshold = _optimise_threshold(
        proba_val,
        y_val.astype(float),
        min_coverage=min_coverage,
        min_ev=min_ev,
    )
    grid = [
        ThresholdPoint(
            float(t),
            float((proba_val >= t).mean()),
            (
                float(np.mean(2 * proba_val[proba_val >= t] - 1))
                if np.any(proba_val >= t)
                else float("nan")
            ),
        )
        for t in np.linspace(0.3, 0.9, 25)
    ]
    curve_path = Path(reports_root) / "threshold_curve.png"
    valid_points = [p for p in grid if not math.isnan(p.expected_value)]
    if valid_points:
        _save_threshold_curve(curve_path, valid_points)
    brier = float(brier_score_loss(y_val, proba_val))
    feature_names = bundle.meta["feature_names"]
    reference = pd.DataFrame(X_train, columns=feature_names)
    current = pd.DataFrame(X_val, columns=feature_names)
    drift_report = compute_drift_report(reference, current)
    daily_brief = {
        "date": target_date.isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "tau": threshold.threshold,
        "coverage": threshold.coverage,
        "expected_value": threshold.expected_value,
        "brier_score": brier,
        "drift": drift_report.summary,
    }
    reports_dir = Path(reports_root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    brief_path = reports_dir / f"daily_brief_{target_date.strftime('%Y%m%d')}.json"
    brief_path.write_text(json.dumps(daily_brief, indent=2), encoding="utf-8")
    version = target_date.strftime("%Y%m%d")
    model_dir = Path(models_root) / version
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, model_dir / "model.pkl")
    joblib.dump(calibrator, model_dir / "calibrator.pkl")
    model_card = _build_model_card(
        version=version,
        training_rows=len(X_train),
        validation_rows=len(X_val),
        threshold=threshold,
        brier=brier,
        drift_overall=drift_report.overall.name,
    )
    (model_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )
    return PipelineResult(
        daily_brief=brief_path,
        threshold_curve=curve_path,
        model_dir=model_dir,
        metrics=daily_brief,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toptek nightly preparation pipeline")
    parser.add_argument(
        "--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date()
    )
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--min-coverage", type=float, default=0.1)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--bank-root", default="data/bank")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--models-root", default="models")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_pipeline(
        target_date=args.date,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        min_coverage=args.min_coverage,
        min_ev=args.min_ev,
        bank_root=Path(args.bank_root),
        reports_root=Path(args.reports_root),
        models_root=Path(args.models_root),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
