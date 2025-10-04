"""Regression tests for the Trade tab widget."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")

tk = pytest.importorskip("tkinter")
from tkinter import ttk

import toptek.core as core_package

sys.modules.setdefault("core", core_package)

from toptek.core import utils
from toptek.gui.widgets import TradeTab


def _paths(root: Path) -> utils.AppPaths:
    return utils.AppPaths(root=root, cache=root / "cache", models=root / "models")


def test_trade_tab_guard_initialises_with_configured_text(tmp_path: Path) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    guard_text = "Topstep Guard: ready"
    configs: dict[str, dict[str, object]] = {
        "ui": {
            "status": {
                "guard": {
                    "pending": guard_text,
                    "intro": "Guard intro text",
                }
            }
        },
        "risk": {},
    }

    tab = TradeTab(notebook, configs, _paths(tmp_path))

    assert tab.guard_status.get() == guard_text
    assert tab.guard_label is not None
    assert tab.guard_label.cget("textvariable") == str(tab.guard_status)

    root.destroy()
