"""Tkinter application bootstrap for Toptek."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List

from core import utils


class ToptekApp(ttk.Notebook):
    """Main application notebook containing all tabs."""

    def __init__(
        self,
        master: tk.Misc,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        on_tab_change: callable | None = None,
    ) -> None:
        super().__init__(master)
        self.configs = configs
        self.paths = paths
        self._on_tab_change = on_tab_change
        self._tab_names: List[str] = []
        self._tab_guidance: Dict[str, str] = {}
        self._build_tabs()
        self.bind("<<NotebookTabChanged>>", self._handle_tab_change)

    def _build_tabs(self) -> None:
        from . import widgets

        tabs = {
            "Login": (
                widgets.LoginTab,
                "Step 1 · Secure your API keys and verify environment access.",
            ),
            "Research": (
                widgets.ResearchTab,
                "Step 2 · Explore market structure and feature snapshots.",
            ),
            "Train": (
                widgets.TrainTab,
                "Step 3 · Fit and calibrate models before risking capital.",
            ),
            "Backtest": (
                widgets.BacktestTab,
                "Step 4 · Validate expectancy and drawdown resilience.",
            ),
            "Trade": (
                widgets.TradeTab,
                "Step 5 · Check Topstep guardrails and plan manual execution.",
            ),
        }
        for name, (cls, guidance) in tabs.items():
            frame = cls(self, self.configs, self.paths)
            self.add(frame, text=name)
            self._tab_names.append(name)
            self._tab_guidance[name] = guidance

    def initialise_guidance(self) -> None:
        """Invoke the guidance callback for the default tab."""

        if not self._tab_names:
            return
        self._dispatch_tab_change(0)

    def _handle_tab_change(self, _: tk.Event) -> None:
        index = self.index("current")
        self._dispatch_tab_change(index)

    def _dispatch_tab_change(self, index: int) -> None:
        if self._on_tab_change is None:
            return
        name = self._tab_names[index]
        guidance = self._tab_guidance.get(name, "")
        self._on_tab_change(index, name, guidance)


def launch_app(*, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
    """Initialise and start the Tkinter main loop."""

    root = tk.Tk()
    root.title("Toptek Mission Control")
    root.geometry("1024x680")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        # ``clam`` is widely available, but gracefully fallback if missing.
        pass
    style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"))
    style.configure("SubHeader.TLabel", font=("Segoe UI", 12))
    style.configure("Step.TLabel", font=("Segoe UI", 11))
    style.configure("Guidance.TLabelframe", padding=(12, 10))
    style.configure("Guidance.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
    style.configure("TNotebook.Tab", padding=(14, 8))

    container = ttk.Frame(root, padding=16)
    container.pack(fill=tk.BOTH, expand=True)

    header = ttk.Frame(container)
    header.pack(fill=tk.X, pady=(0, 12))
    ttk.Label(header, text="Project X · Manual Trading Copilot", style="Header.TLabel").pack(anchor=tk.W)
    ttk.Label(
        header,
        text="Follow the guided workflow from credentials to Topstep-compliant trade plans.",
        style="SubHeader.TLabel",
    ).pack(anchor=tk.W)

    guidance_card = ttk.Labelframe(container, text="Mission Checklist", style="Guidance.TLabelframe")
    guidance_card.pack(fill=tk.X, pady=(0, 12))

    step_label = ttk.Label(guidance_card, style="Step.TLabel")
    step_label.pack(anchor=tk.W)
    progress = ttk.Progressbar(guidance_card, maximum=4, mode="determinate", length=220)
    progress.pack(anchor=tk.W, pady=(8, 0))

    def handle_tab_change(index: int, name: str, guidance: str) -> None:
        step_label.config(text=f"{guidance}\n→ Current focus: {name} tab")
        progress["value"] = index

    notebook = ToptekApp(container, configs, paths, on_tab_change=handle_tab_change)
    notebook.pack(fill=tk.BOTH, expand=True)
    notebook.initialise_guidance()

    root.mainloop()


__all__ = ["launch_app", "ToptekApp"]
