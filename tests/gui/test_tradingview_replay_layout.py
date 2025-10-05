"""Regression tests covering TradingView and Replay tab layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

from toptek.core import utils  # noqa: E402
from toptek.gui.tradingview import TradingViewRouter  # noqa: E402
from toptek.gui.widgets import ReplayTab, TradingViewTab  # noqa: E402


def _paths(root: Path) -> utils.AppPaths:
    return utils.AppPaths(root=root, cache=root / "cache", models=root / "models")


def test_tradingview_tab_sections_and_weights(tmp_path: Path) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI display availability
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {"ui": {}, "tradingview": {}}
    router = TradingViewRouter(
        app_config={
            "tv": {
                "enabled": True,
                "favorites": [
                    {"symbol": "ES=F", "interval": "5", "label": "ES session"}
                ],
            }
        },
        ui_config={},
    )

    tab = TradingViewTab(notebook, configs, _paths(tmp_path), tv_router=router)

    try:
        assert "symbol_section" in tab.children
        assert "launch_section" in tab.children

        symbol_section = tab.children["symbol_section"]
        launch_section = tab.children["launch_section"]

        assert int(symbol_section.grid_columnconfigure(1)["weight"]) == 1
        assert int(launch_section.grid_columnconfigure(1)["weight"]) == 1

        launch_children = [child for child in launch_section.winfo_children() if isinstance(child, ttk.Button)]
        assert any(
            child.cget("text") == "Launch TradingView (Ctrl+Shift+T)"
            for child in launch_children
        )
    finally:
        root.destroy()


def test_replay_tab_grouped_labelframes(tmp_path: Path) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI display availability
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {"ui": {}}

    tab = ReplayTab(notebook, configs, _paths(tmp_path))

    try:
        controls = tab.children.get("replay_controls")
        assert controls is not None
        expected_groups = {"dataset_group", "playback_group", "status_group"}
        assert expected_groups.issubset(controls.children.keys())

        dataset_group = controls.children["dataset_group"]
        playback_group = controls.children["playback_group"]
        status_group = controls.children["status_group"]

        assert int(dataset_group.grid_columnconfigure(1)["weight"]) == 1
        assert int(playback_group.grid_columnconfigure(1)["weight"]) == 1
        assert int(playback_group.grid_columnconfigure(3)["weight"]) == 1

        status_labels = [
            child
            for child in status_group.winfo_children()
            if isinstance(child, ttk.Label)
            and child.cget("style") == "StatusInfo.TLabel"
        ]
        assert status_labels and str(status_labels[0].cget("textvariable")) == str(tab.status_var)
    finally:
        root.destroy()
