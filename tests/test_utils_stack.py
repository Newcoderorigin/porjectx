"""Tests for numeric stack validation and logging utilities."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pytest

from toptek.core import utils


def test_assert_numeric_stack_writes_report(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    versions = utils.assert_numeric_stack(reports_dir=reports_dir)

    report_path = reports_dir / "run_stack.json"
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["required"]["numpy"] == versions["numpy"]
    assert payload["expected"]["scipy"] == utils.STACK_REQUIREMENTS["scipy"]


def test_assert_numeric_stack_raises_on_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(utils.STACK_REQUIREMENTS, "numpy", "0.0.0")

    with pytest.raises(RuntimeError) as excinfo:
        utils.assert_numeric_stack(reports_dir=tmp_path)

    assert "scripts/setup_env.ps1" in str(excinfo.value)

    report = json.loads((tmp_path / "run_stack.json").read_text(encoding="utf-8"))
    assert report["status"] == "error"


def test_set_seeds_reproducible() -> None:
    utils.set_seeds(123)
    first = np.random.random(3)
    utils.set_seeds(123)
    second = np.random.random(3)

    assert np.allclose(first, second)


def test_configure_logging_installs_rotating_handler(tmp_path: Path) -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    try:
        log_path = utils.configure_logging(tmp_path, level="INFO")
        assert log_path.exists()
        assert any(
            isinstance(handler, RotatingFileHandler)
            for handler in logging.getLogger().handlers
        )
    finally:
        for handler in logging.getLogger().handlers:
            handler.close()
        logging.getLogger().handlers.clear()
        for handler in original_handlers:
            logging.getLogger().addHandler(handler)
