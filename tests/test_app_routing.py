from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from toptek.core import utils


def test_toptek_app_routes_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    tk = pytest.importorskip("tkinter")
    from tkinter import ttk

    import toptek.gui.app as app_module

    original_widgets = sys.modules.get("toptek.gui.widgets")
    stub_module = types.ModuleType("toptek.gui.widgets")

    class _StubTab(ttk.Frame):
        def __init__(self, master, configs, paths):
            super().__init__(master)
            self.configs = configs
            self.paths = paths
            self.activated = False

        def on_activated(self) -> None:
            self.activated = True

    for name in [
        "DashboardTab",
        "LoginTab",
        "ResearchTab",
        "TrainTab",
        "BacktestTab",
        "ReplayTab",
        "TradeTab",
    ]:
        setattr(stub_module, name, type(name, (_StubTab,), {}))

    monkeypatch.setitem(sys.modules, "toptek.gui.widgets", stub_module)
    app_module = importlib.reload(app_module)

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()

    paths = utils.AppPaths(
        root=tmp_path, cache=tmp_path / "cache", models=tmp_path / "models"
    )
    callback_events: list[tuple[int, str, str]] = []

    app = app_module.ToptekApp(
        root,
        configs={"ui": {}, "risk": {}},
        paths=paths,
        on_tab_change=lambda idx, name, guidance: callback_events.append(
            (idx, name, guidance)
        ),
    )

    app.initialise_guidance()
    assert callback_events
    first_idx, first_name, first_guidance = callback_events[0]
    assert first_idx == 0
    assert first_name == "Dashboard"
    assert "Overview" in first_guidance

    dashboard_tab = app._tab_widgets[first_name]
    assert getattr(dashboard_tab, "activated") is True

    app._dispatch_tab_change(3)
    second_idx, second_name, second_guidance = callback_events[-1]
    assert second_idx == 3
    assert second_name == "Train"
    assert "Step 3" in second_guidance

    root.destroy()

    if original_widgets is not None:
        monkeypatch.setitem(sys.modules, "toptek.gui.widgets", original_widgets)
    importlib.reload(app_module)
