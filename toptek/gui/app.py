"""Tkinter application bootstrap for Toptek."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict

from core import utils


class ToptekApp(ttk.Notebook):
    """Main application notebook containing all tabs."""

    def __init__(self, master: tk.Tk, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master)
        self.configs = configs
        self.paths = paths
        self._build_tabs()

    def _build_tabs(self) -> None:
        from . import widgets

        tabs = {
            "Login": widgets.LoginTab,
            "Research": widgets.ResearchTab,
            "Train": widgets.TrainTab,
            "Backtest": widgets.BacktestTab,
            "Trade": widgets.TradeTab,
        }
        for name, cls in tabs.items():
            frame = cls(self, self.configs, self.paths)
            self.add(frame, text=name)


def launch_app(*, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
    """Initialise and start the Tkinter main loop."""

    root = tk.Tk()
    root.title("Toptek Starter")
    root.geometry("900x600")
    notebook = ToptekApp(root, configs, paths)
    notebook.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


__all__ = ["launch_app", "ToptekApp"]
