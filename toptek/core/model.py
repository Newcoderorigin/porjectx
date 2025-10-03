"""Simple machine-learning helpers for classification models.

This module provides utilities to train and calibrate lightweight
classifiers used throughout the Toptek application. Models are persisted
to disk so they can be reused by the GUI, CLI, and backtesting modules.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    """Container for training outcomes."""

    model_path: Path
    metrics: Dict[str, float]
    threshold: float
    model_type: str
    calibration_curve: List[Tuple[float, float]]


@dataclass
class CalibrationResult:
    """Represents calibration artefacts for a trained classifier."""

    model_path: Path
    metrics: Dict[str, float]
    calibration_curve: List[Tuple[float, float]]


def _clean_xy(
    X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Align feature matrix and labels while dropping degenerate columns."""

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        columns = [f"feature_{idx}" for idx in range(X_arr.shape[1])]
        X_df = pd.DataFrame(X_arr, columns=columns)

    y_s = pd.Series(y).copy()

    mask_valid_y = y_s.notna().values
    X_df = X_df.loc[mask_valid_y]
    y_s = y_s.loc[mask_valid_y]

    X_df = X_df.dropna(axis=1, how="all")
    nunique = X_df.nunique(dropna=True)
    X_df = X_df.loc[:, nunique > 1]
    if X_df.empty:
        raise ValueError("No usable features remain after cleaning")
    return X_df.reset_index(drop=True), y_s.reset_index(drop=True)


def _build_preprocessor(columns: Sequence[str]) -> ColumnTransformer:
    """Construct the preprocessing pipeline used by all classifiers."""

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer([("num", num_pipe, list(columns))], remainder="drop")


def _select_estimator(model_type: str) -> object:
    """Return the estimator associated with ``model_type``."""

    model_type = model_type.lower()
    if model_type == "logistic":
        return LogisticRegression(max_iter=200, solver="lbfgs")
    if model_type in {"gbm", "gradient_boosting"}:
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_classifier(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    models_dir: Path,
    threshold: float = 0.65,
    model_type: str = "logistic",
) -> TrainResult:
    """Train a classifier specified by ``model_type`` and persist it.

    Args:
        X: Feature matrix.
        y: Binary labels aligned with ``X``.
        models_dir: Directory to store trained artefacts.
        threshold: Classification threshold used for metrics.
        model_type: Either ``"logistic"`` or ``"gbm"``.

    Returns:
        :class:`TrainResult` describing where the model was stored and
        summary metrics from a hold-out split.
    """

    X_df, y_s = _clean_xy(X, y)
    pre = _build_preprocessor(X_df.columns)
    estimator = _select_estimator(model_type)
    pipeline = Pipeline(steps=[("pre", pre), ("clf", estimator)])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, shuffle=True, random_state=42
    )

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    frac_of_pos, mean_pred_value = calibration_curve(
        y_test, proba, n_bins=10, strategy="quantile"
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "n_samples": int(len(y_s)),
        "n_features": int(X_df.shape[1]),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    suffix = "gbm" if model_type.lower() in {"gbm", "gradient_boosting"} else "logistic"
    model_path = models_dir / f"{suffix}_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)

    calibration_points = list(zip(mean_pred_value.tolist(), frac_of_pos.tolist()))
    return TrainResult(
        model_path=model_path,
        metrics=metrics,
        threshold=threshold,
        model_type=suffix,
        calibration_curve=calibration_points,
    )


def calibrate_classifier(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    base_model_path: Path,
    models_dir: Path,
    method: str = "isotonic",
) -> CalibrationResult:
    """Calibrate the probabilities of an existing classifier.

    Args:
        X: Feature matrix used for calibration.
        y: Binary labels aligned with ``X``.
        base_model_path: Path of the previously trained model.
        models_dir: Directory to store the calibrated estimator.
        method: Calibration approach (``"isotonic"`` or ``"sigmoid"``).

    Returns:
        :class:`CalibrationResult` describing the calibrated artefact.
    """

    X_df, y_s = _clean_xy(X, y)
    base_model = load_model(base_model_path)
    calibrator = CalibratedClassifierCV(base_estimator=base_model, method=method, cv=3)
    calibrator.fit(X_df, y_s)
    proba = calibrator.predict_proba(X_df)[:, 1]
    frac_of_pos, mean_pred_value = calibration_curve(
        y_s, proba, n_bins=10, strategy="quantile"
    )
    metrics = {
        "brier_score": float(brier_score_loss(y_s, proba)),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    calibrated_path = models_dir / f"{base_model_path.stem}_calibrated.pkl"
    with calibrated_path.open("wb") as handle:
        pickle.dump(calibrator, handle)

    calibration_points = list(zip(mean_pred_value.tolist(), frac_of_pos.tolist()))
    return CalibrationResult(
        model_path=calibrated_path,
        metrics=metrics,
        calibration_curve=calibration_points,
    )


def load_model(model_path: Path):
    """Load a persisted model from disk."""

    with model_path.open("rb") as handle:
        return pickle.load(handle)


__all__ = [
    "train_classifier",
    "calibrate_classifier",
    "load_model",
    "TrainResult",
    "CalibrationResult",
]
