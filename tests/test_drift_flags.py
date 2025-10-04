"""Deterministic checks for the drift severity report."""

from __future__ import annotations

import pytest

from toptek.monitor import Severity, compute_drift_report


def _build_column(zero_count: int, one_count: int) -> dict[str, list[int]]:
    return {"feature": [0] * zero_count + [1] * one_count}


def test_compute_drift_report_stable_flag():
    reference = _build_column(50, 50)
    current = _build_column(50, 50)

    report = compute_drift_report(reference, current, bins=2)

    assert report.overall == Severity.STABLE
    feature_report = report.features["feature"]
    assert feature_report.metric.psi == pytest.approx(0.0)
    assert feature_report.metric.ks == pytest.approx(0.0)
    assert feature_report.severity == Severity.STABLE
    assert feature_report.message == "No material drift detected."


def test_compute_drift_report_watch_flag():
    reference = _build_column(50, 50)
    current = _build_column(68, 32)

    report = compute_drift_report(reference, current, bins=2)

    feature_report = report.features["feature"]
    assert feature_report.psi_severity == Severity.WATCH
    assert feature_report.ks_severity == Severity.WATCH
    assert feature_report.severity == Severity.WATCH
    assert report.overall == Severity.WATCH


def test_compute_drift_report_alert_flag():
    reference = _build_column(50, 50)
    current = _build_column(80, 20)

    report = compute_drift_report(reference, current, bins=2)

    feature_report = report.features["feature"]
    assert feature_report.severity == Severity.ALERT
    assert feature_report.metric.psi > 0.25
    assert feature_report.metric.ks > 0.2
    assert "alert" in report.summary


def test_compute_drift_report_unknown_when_empty():
    reference = _build_column(50, 50)
    current = {"feature": []}

    report = compute_drift_report(reference, current)

    feature_report = report.features["feature"]
    assert feature_report.severity == Severity.UNKNOWN
    assert report.overall == Severity.UNKNOWN
    assert report.summary == "One or more features lacked data for drift assessment."
