"""Simple machine-learning helpers for classification models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class TrainResult:
    """Container for training outcomes."""

    model_path: Path
    metrics: Dict[str, float]
    threshold: float
    preprocessing: Dict[str, int] = field(default_factory=dict)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "logistic",
    models_dir: Path,
    threshold: float = 0.65,
) -> TrainResult:
    """Train a basic classifier and persist it to ``models_dir``.

    The training routine converts non-finite feature values to ``NaN``, drops rows and
    columns that are entirely missing, and applies a median :class:`SimpleImputer`
    before fitting the estimator. Summary statistics about these preprocessing steps
    are returned in the :class:`TrainResult`.

    Raises
    ------
    ValueError
        If the feature matrix is not two-dimensional, lacks usable rows or columns
        after cleaning, contains invalid target values, or the target labels collapse
        into a single class.
    """

    if X.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if not np.isfinite(y).all():
        raise ValueError("Target labels contain NaN or inf values; clean the labels before training")

    non_finite_mask = ~np.isfinite(X)
    imputed_cells = int(non_finite_mask.sum())
    if imputed_cells:
        X = X.copy()
        X[non_finite_mask] = np.nan

    row_all_nan = np.isnan(X).all(axis=1)
    dropped_rows = int(row_all_nan.sum())
    if dropped_rows:
        X = X[~row_all_nan]
        y = y[~row_all_nan]

    if X.size == 0:
        raise ValueError("No valid feature rows remain after removing all-NaN rows")

    col_all_nan = np.isnan(X).all(axis=0)
    dropped_columns = int(col_all_nan.sum())
    if dropped_columns:
        X = X[:, ~col_all_nan]
        if X.shape[1] == 0:
            raise ValueError("All feature columns were empty after cleaning; cannot train")

    if np.unique(y).size < 2:
        raise ValueError("Training requires at least two target classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    if model_type == "logistic":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "gbm":
        classifier = GradientBoostingClassifier()
    else:
        raise ValueError("Unknown model type")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_type}_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)
    return TrainResult(
        model_path=model_path,
        metrics=metrics,
        threshold=threshold,
        preprocessing={
            "imputed_cells": imputed_cells,
            "dropped_rows": dropped_rows,
            "dropped_columns": dropped_columns,
        },
    )


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
    calibrator = CalibratedClassifierCV(estimator=pipeline, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    target_path = output_path or model_path.with_name(f"{model_path.stem}_calibrated.pkl")
    with target_path.open("wb") as handle:
        pickle.dump(calibrator, handle)
    return target_path


__all__ = ["train_classifier", "load_model", "TrainResult", "calibrate_classifier"]
