"""Regression tests for the numeric stack guard in ``toptek.main``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:  # pragma: no cover - prefer site-packages import
    from toptek.core import utils
except ModuleNotFoundError:  # pragma: no cover - fallback when running from repo root
    sys.path.insert(0, str(PROJECT_ROOT))
    from toptek.core import utils


class _DummyUI:
    """Lightweight stand-in for :class:`UIConfig` used in tests."""

    def __init__(self) -> None:
        self.shell = type(
            "Shell",
            (),
            {
                "symbol": "ES",
                "interval": "5m",
                "lookback_bars": 100,
                "model": "logistic",
            },
        )()
        self.chart = type("Chart", (), {"fps": 12})()

    def as_dict(self) -> Dict[str, object]:
        return {}

    def with_updates(self, **_updates):  # pragma: no cover - not exercised
        return self


@pytest.fixture(autouse=True)
def _restore_argv(monkeypatch):
    """Ensure tests run with a clean ``sys.argv``."""

    original = list(sys.argv)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    yield
    monkeypatch.setattr(sys, "argv", original)


@pytest.fixture
def patched_main(monkeypatch):
    """Keep ``main`` deterministic when the guard allows execution."""

    import toptek.main as main_module

    def _fake_load_configs():
        return {"app": {}, "risk": {}, "features": {}, "ui": {}}, _DummyUI()

    monkeypatch.setattr(main_module, "load_configs", _fake_load_configs)
    return main_module


def test_main_surfaces_numeric_stack_error(monkeypatch, patched_main):
    """Stack mismatch should raise a helpful RuntimeError before SciPy loads."""

    monkeypatch.setitem(utils.STACK_REQUIREMENTS, "numpy", ">=999.0")
    with pytest.raises(RuntimeError) as excinfo:
        patched_main.main()
    message = str(excinfo.value)
    assert "Incompatible numeric stack" in message
    assert "numpy" in message
    assert "pip install -r toptek/requirements-lite.txt" in message
