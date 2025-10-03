"""Simple machine-learning helpers for classification models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    """Container for training outcomes."""

    model_path: Path
    metrics: Dict[str, float]
    threshold: float


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "logistic",
    models_dir: Path,
    threshold: float = 0.65,
) -> TrainResult:
    """Train a basic classifier and persist it to ``models_dir``."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "gbm":
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_type}_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    return TrainResult(model_path=model_path, metrics=metrics, threshold=threshold)


def load_model(model_path: Path):
    """Load a persisted model from disk."""

    with model_path.open("rb") as handle:
        return pickle.load(handle)


def calibrate_classifier(
    model_path: Path,
    calibration_data: Tuple[np.ndarray, np.ndarray],
    *,
    method: str = "sigmoid",
    output_path: Path | None = None,
) -> Path:
    """Calibrate a pre-trained classifier using hold-out data.

    Parameters
    ----------
    model_path:
        Location of the previously fitted classifier pipeline.
    calibration_data:
        Tuple containing the feature matrix and labels reserved for calibration.
    method:
        Calibration method supported by :class:`CalibratedClassifierCV` (``sigmoid`` or ``isotonic``).
    output_path:
        Optional override for where the calibrated model should be persisted.

    Returns
    -------
    Path
        Filesystem path to the calibrated model artefact.
    """

    X_cal, y_cal = calibration_data
    pipeline = load_model(model_path)
    calibrator = CalibratedClassifierCV(base_estimator=pipeline, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    target_path = output_path or model_path.with_name(f"{model_path.stem}_calibrated.pkl")
    with target_path.open("wb") as handle:
        pickle.dump(calibrator, handle)
    return target_path


__all__ = ["train_classifier", "load_model", "TrainResult", "calibrate_classifier"]
