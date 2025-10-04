from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("joblib")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from toptek.pipelines.prep_nightly import PipelineResult, run_pipeline


def test_run_pipeline(tmp_path: Path, monkeypatch) -> None:
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    bank_root = tmp_path / "bank"
    result: PipelineResult = run_pipeline(
        target_date=date(2024, 2, 1),
        bank_root=bank_root,
        reports_root=reports,
        models_root=models,
        days=30,
    )
    brief = result.daily_brief
    assert brief.exists()
    data = brief.read_text(encoding="utf-8")
    assert "tau" in data and "drift" in data
    assert result.threshold_curve.exists()
    version_dir = result.model_dir
    assert (version_dir / "model.pkl").exists()
    assert (version_dir / "calibrator.pkl").exists()
    assert (version_dir / "model_card.json").exists()
