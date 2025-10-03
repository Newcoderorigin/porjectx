"""Simple machine-learning helpers for classification models."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    """Container for training outcomes."""

    model_path: Path
    metrics: Dict[str, float]
    threshold: float


def _clean_xy(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align feature matrix and labels while dropping degenerate columns."""

    X_df = pd.DataFrame(X).copy()
    y_s = pd.Series(y).copy()

    mask_valid_y = y_s.notna().values
    X_df = X_df.loc[mask_valid_y]
    y_s = y_s.loc[mask_valid_y]

    X_df = X_df.dropna(axis=1, how="all")
    nunique = X_df.nunique(dropna=True)
    X_df = X_df.loc[:, nunique > 1]
    return X_df, y_s


def train_classifier(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    models_dir: Path,
    threshold: float = 0.65,
) -> TrainResult:
    """Train a basic logistic classifier and persist it to ``models_dir``."""

    X_df, y_s = _clean_xy(X, y)

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    pre = ColumnTransformer([("num", num_pipe, list(range(X_df.shape[1])))], remainder="drop")

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, shuffle=True, random_state=42
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_samples": int(len(y_s)),
        "n_features": int(X_df.shape[1]),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logistic_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    return TrainResult(model_path=model_path, metrics=metrics, threshold=threshold)


def load_model(model_path: Path):
    """Load a persisted model from disk."""

    with model_path.open("rb") as handle:
        return pickle.load(handle)


__all__ = ["train_classifier", "load_model", "TrainResult"]
