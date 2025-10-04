"""Calibration and threshold optimisation tests."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from toptek.model.threshold import opt_threshold


def test_calibration_improves_brier_and_threshold_hits() -> None:
    X, y = make_classification(
        n_samples=600,
        n_features=5,
        n_informative=4,
        class_sep=1.5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=24, stratify=y)

    base = LogisticRegression(max_iter=1000)
    base.fit(X_train, y_train)
    raw_probs = base.predict_proba(X_test)[:, 1]

    calibrator = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv="prefit")
    calibrator.fit(X_test, y_test)
    calibrated_probs = calibrator.predict_proba(X_test)[:, 1]

    brier_raw = brier_score_loss(y_test, raw_probs)
    brier_cal = brier_score_loss(y_test, calibrated_probs)
    assert brier_cal <= brier_raw - 0.02

    tau, curve = opt_threshold(calibrated_probs, y_test, min_coverage=0.3, min_expected_value=-0.1)
    assert tau >= 0.5
    baseline_mask = calibrated_probs >= 0.5
    baseline_hit_rate = float(y_test[baseline_mask].mean()) if baseline_mask.any() else 0.0
    selected = calibrated_probs >= tau
    hit_rate = float(y_test[selected].mean())
    assert hit_rate >= baseline_hit_rate + 0.05
    assert curve
