"""Regression tests for the Trade tab widget."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

import toptek.core as core_package  # noqa: E402

sys.modules.setdefault("core", core_package)

from toptek.core import utils  # noqa: E402
from toptek.gui import DARK_PALETTE  # noqa: E402
from toptek.gui.widgets import TradeTab  # noqa: E402


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

    assert tab.guard_status.get() == f"{guard_text} · Mode PAPER"
    assert tab.guard_label is not None
    assert tab.guard_label.cget("textvariable") == str(tab.guard_status)

    root.destroy()


def test_trade_tab_guard_refresh_updates_badge_and_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {
        "ui": {"status": {"guard": {"pending": "Topstep Guard: pending"}}},
        "risk": {},
    }
    tab = TradeTab(notebook, configs, _paths(tmp_path))
    monkeypatch.setattr(
        "toptek.gui.widgets.messagebox.showinfo", lambda *args, **kwargs: None
    )

    report = tab._refresh_guard(show_modal=False)

    assert report.status == "OK"
    assert tab.guard_status.get() == "Topstep Guard: OK · Mode PAPER"
    assert tab.guard_label is not None
    assert tab.guard_label.cget("foreground") == DARK_PALETTE["success"]

    payload = tab.output.get("1.0", "end-1c")
    assert '"status": "OK"' in payload
    assert configs["trade"]["report"]["status"] == "OK"

    root.destroy()


def test_trade_tab_guard_refresh_trips_defensive_when_policy_breached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {
        "ui": {"status": {"guard": {"pending": "Guard pending"}}},
        "risk": {"max_position_size": 0, "max_daily_loss": 10000},
    }
    tab = TradeTab(notebook, configs, _paths(tmp_path))
    monkeypatch.setattr(
        "toptek.gui.widgets.messagebox.showwarning", lambda *args, **kwargs: None
    )

    report = tab._refresh_guard(show_modal=False)

    assert report.status == "DEFENSIVE_MODE"
    assert tab.guard_status.get() == "Topstep Guard: DEFENSIVE_MODE · Mode PAPER"
    assert tab.guard_label is not None
    assert tab.guard_label.cget("foreground") == DARK_PALETTE["danger"]

    payload = tab.output.get("1.0", "end-1c")
    assert '"DEFENSIVE_MODE"' in payload
    assert configs["trade"]["report"]["status"] == "DEFENSIVE_MODE"

    root.destroy()


def test_trade_tab_live_mode_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {
        "ui": {"status": {"guard": {"pending": "Guard pending"}}},
        "risk": {},
    }
    tab = TradeTab(notebook, configs, _paths(tmp_path))

    monkeypatch.setattr(
        "toptek.gui.widgets.messagebox.askyesno", lambda *args, **kwargs: False
    )
    tab._set_mode("LIVE")
    assert tab.trading_mode.get() == "PAPER"
    assert tab.guard_status.get().endswith("Mode PAPER")

    monkeypatch.setattr(
        "toptek.gui.widgets.messagebox.askyesno", lambda *args, **kwargs: True
    )
    tab._set_mode("LIVE")
    assert tab.trading_mode.get() == "LIVE"
    assert tab.guard_status.get().endswith("Mode LIVE")
    assert configs["trade"]["mode"] == "LIVE"

    root.destroy()


def test_trade_tab_panic_hotkey_forces_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {
        "ui": {"status": {"guard": {"pending": "Guard pending"}}},
        "risk": {},
    }
    tab = TradeTab(notebook, configs, _paths(tmp_path))
    monkeypatch.setattr(
        "toptek.gui.widgets.messagebox.askyesno", lambda *args, **kwargs: True
    )
    tab._set_mode("LIVE")
    assert tab.trading_mode.get() == "LIVE"

    tab._handle_panic(None)

    assert tab.trading_mode.get() == "PAPER"
    assert tab.guard_status.get().endswith("Mode PAPER")
    assert configs["trade"]["mode"] == "PAPER"
    assert "panic_at" in configs["trade"]

    root.destroy()
