"""Simple machine-learning helpers for classification models.

This module offers light-weight wrappers around scikit-learn estimators to
standardise the training and calibration workflows used by the Toptek GUI and
CLI tools.

Example
-------
>>> import numpy as np
>>> from pathlib import Path
>>> X = np.random.randn(200, 6)
>>> y = (X[:, 0] > 0).astype(int)
>>> result = train_classifier(X, y, models_dir=Path("models"))
>>> _ = calibrate_classifier(result.model_path, (X[:40], y[:40]))
"""

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
    """Container for training outcomes.

    Attributes
    ----------
    model_path:
        Filesystem location of the persisted estimator pipeline.
    metrics:
        Dictionary containing evaluation metrics computed on the validation
        split.
    threshold:
        Decision threshold used when deriving discrete class predictions from
        probabilities.
    preprocessing:
        Summary statistics describing how the feature matrix was sanitised.
    retained_columns:
        Tuple of retained column indices relative to the original feature
        matrix. ``None`` when no trimming occurred.
    original_feature_count:
        Column count observed before preprocessing.
    """

    model_path: Path
    metrics: Dict[str, float]
    threshold: float
    preprocessing: Dict[str, int] = field(default_factory=dict)
    retained_columns: tuple[int, ...] | None = None
    original_feature_count: int | None = None


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "logistic",
    models_dir: Path,
    threshold: float = 0.65,
) -> TrainResult:
    """Train a basic classifier and persist it to ``models_dir``.

    Parameters
    ----------
    X:
        Feature matrix to train on. The routine casts the payload to ``float`` and
        sanitises non-finite entries prior to fitting.
    y:
        Target labels associated with ``X``. The labels must contain at least two
        distinct classes.
    model_type:
        Name of the estimator to fit (``"logistic"`` or ``"gbm"``).
    models_dir:
        Directory where the fitted pipeline should be persisted.
    threshold:
        Probability threshold for translating predictions into class labels when
        deriving simple metrics.

    Returns
    -------
    TrainResult
        Metadata about the persisted model, including preprocessing telemetry.

    Raises
    ------
    ValueError
        If the feature matrix is not two-dimensional, lacks usable rows or columns
        after cleaning, contains invalid target values, or the target labels collapse
        into a single class.

    Example
    -------
    >>> import numpy as np
    >>> from pathlib import Path
    >>> X = np.random.rand(120, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> train_classifier(X, y, models_dir=Path("models"))
    TrainResult(...)
    """

    if X.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    original_feature_count = X.shape[1]

    if not np.isfinite(y).all():
        raise ValueError(
            "Target labels contain NaN or inf values; clean the labels before training"
        )

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
    retained_columns: tuple[int, ...] | None = None
    if dropped_columns:
        valid_column_mask = ~col_all_nan
        X = X[:, valid_column_mask]
        if X.shape[1] == 0:
            raise ValueError("All feature columns were empty after cleaning; cannot train")
        retained_columns = tuple(int(idx) for idx in np.flatnonzero(valid_column_mask))
    else:
        retained_columns = None

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
        retained_columns=retained_columns,
        original_feature_count=original_feature_count,
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
    feature_mask: tuple[int, ...] | np.ndarray | None = None,
    original_feature_count: int | None = None,
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

    feature_mask:
        Optional tuple of retained column indices from the original feature matrix.
        When provided, the calibration features will be subset or reordered to
        match the training-time dimensionality.
    original_feature_count:
        Number of columns in the original training feature matrix. Used to
        determine whether the calibration payload still contains the untouched
        feature space or has already been trimmed.

    Returns
    -------
    Path
        Filesystem path to the calibrated model artefact.

    Raises
    ------
    ValueError
        If the calibration payload is malformed, references invalid feature
        indices, or lacks class diversity.

    Example
    -------
    >>> from pathlib import Path
    >>> import numpy as np
    >>> model_path = Path("models/logistic_model.pkl")
    >>> X_cal = np.random.rand(50, 4)
    >>> y_cal = (X_cal[:, 0] > 0.5).astype(int)
    >>> calibrate_classifier(model_path, (X_cal, y_cal))
    PosixPath('models/logistic_model_calibrated.pkl')
    """

    X_cal, y_cal = calibration_data
    X_cal = np.asarray(X_cal, dtype=float)
    y_cal = np.asarray(y_cal).ravel()

    if X_cal.ndim != 2:
        raise ValueError("Calibration feature matrix must be 2-dimensional")

    if not np.isfinite(y_cal).all():
        raise ValueError(
            "Calibration labels contain NaN or inf values; clean the labels before calibrating"
        )

    if feature_mask is not None:
        indices = np.asarray(feature_mask, dtype=int)
        if indices.ndim != 1:
            raise ValueError("Feature mask must be a 1-D sequence of column indices")
        if indices.size == 0:
            raise ValueError("Feature mask is empty; cannot realign calibration features")
        if (indices < 0).any():
            raise ValueError("Feature mask cannot include negative column indices")
        if not np.all(np.diff(indices) >= 0):
            raise ValueError("Feature mask must be sorted in ascending order")
        if np.unique(indices).size != indices.size:
            raise ValueError("Feature mask contains duplicate column indices")

        max_index = int(indices.max())
        if original_feature_count is None or original_feature_count == X_cal.shape[1]:
            if max_index >= X_cal.shape[1]:
                raise ValueError(
                    "Feature mask references columns beyond the calibration matrix bounds"
                )
            X_cal = X_cal[:, indices]
        elif X_cal.shape[1] == indices.size:
            # Calibration payload already trimmed to the retained columns. We assume the
            # supplied feature order already matches the mask order since we no longer
            # have the dropped columns to cross-check against.
            pass
        else:
            raise ValueError(
                "Calibration payload has unexpected dimensionality relative to the training mask"
            )

    non_finite = ~np.isfinite(X_cal)
    if non_finite.any():
        X_cal = X_cal.copy()
        X_cal[non_finite] = np.nan

    row_all_nan = np.isnan(X_cal).all(axis=1)
    if row_all_nan.any():
        X_cal = X_cal[~row_all_nan]
        y_cal = y_cal[~row_all_nan]
    if X_cal.size == 0:
        raise ValueError("No valid calibration rows remain after cleaning")

    if np.unique(y_cal).size < 2:
        raise ValueError("Calibration requires at least two target classes")

    pipeline = load_model(model_path)
    expected_features = getattr(pipeline, "n_features_in_", None)
    if expected_features is not None and X_cal.shape[1] != expected_features:
        raise ValueError(
            "Calibration feature matrix shape does not match the fitted model"
        )
    calibrator = CalibratedClassifierCV(estimator=pipeline, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    target_path = output_path or model_path.with_name(f"{model_path.stem}_calibrated.pkl")
    with target_path.open("wb") as handle:
        pickle.dump(calibrator, handle)
    return target_path


__all__ = ["train_classifier", "load_model", "TrainResult", "calibrate_classifier"]
