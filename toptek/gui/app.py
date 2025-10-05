"""Tkinter application bootstrap for Toptek."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List

from core import utils

from . import DARK_PALETTE
from .webshell import WebFrontendHandle, launch_web_frontend


class ToptekApp(ttk.Notebook):
    """Main application notebook containing all tabs."""

    def __init__(
        self,
        master: tk.Misc,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        on_tab_change: Callable[[int, str, str], None] | None = None,
    ) -> None:
        super().__init__(master, style="Dashboard.TNotebook")
        self.configs = configs
        self.paths = paths
        self._on_tab_change = on_tab_change
        self._tab_names: List[str] = []
        self._tab_guidance: Dict[str, str] = {}
        self._tab_widgets: Dict[str, ttk.Frame] = {}
        self._build_tabs()
        self.bind("<<NotebookTabChanged>>", self._handle_tab_change)

    def _build_tabs(self) -> None:
        from . import widgets

        tabs = [
            (
                "Dashboard",
                lambda parent: widgets.DashboardTab(parent, self.configs, self.paths),
                "Overview · Monitor readiness metrics before taking action.",
            ),
            (
                "Login",
                lambda parent: widgets.LoginTab(parent, self.configs, self.paths),
                "Step 1 · Secure your API keys and verify environment access.",
            ),
            (
                "Research",
                lambda parent: widgets.ResearchTab(parent, self.configs, self.paths),
                "Step 2 · Explore market structure and feature snapshots.",
            ),
            (
                "Train",
                lambda parent: widgets.TrainTab(parent, self.configs, self.paths),
                "Step 3 · Fit and calibrate models before risking capital.",
            ),
            (
                "Backtest",
                lambda parent: widgets.BacktestTab(parent, self.configs, self.paths),
                "Step 4 · Validate expectancy and drawdown resilience.",
            ),
            (
                "Replay",
                lambda parent: widgets.ReplayTab(parent, self.configs, self.paths),
                "Step 5 · Rehearse the playbook against recorded sessions before trading live.",
            ),
            (
                "Trade",
                lambda parent: widgets.TradeTab(parent, self.configs, self.paths),
                "Step 6 · Check Topstep guardrails and plan manual execution.",
            ),
        ]

        lm_config = self.configs.get("lmstudio", {})
        if lm_config.get("enabled"):
            try:
                from toptek.lmstudio import build_client
                from toptek.ui import LiveTab as LMStudioLiveTab

                lm_client = build_client(lm_config)
            except Exception:  # pragma: no cover - defensive
                lm_client = None
                LMStudioLiveTab = None  # type: ignore[assignment]
            if LMStudioLiveTab is not None:
                tabs.append(
                    (
                        "Live",
                        lambda parent: LMStudioLiveTab(parent, lm_config, client=lm_client),
                        "Live · Chat with LM Studio and monitor response latency.",
                    )
                )

        research_config = self.configs.get("futures_research", {})
        if research_config.get("enabled"):
            try:
                from toptek.ui import FuturesResearchTab
            except Exception:  # pragma: no cover - defensive
                FuturesResearchTab = None  # type: ignore[assignment]
            if FuturesResearchTab is not None:
                tabs.append(
                    (
                        "Futures",
                        lambda parent: FuturesResearchTab(parent, research_config),
                        "Research · Inspect futures history via Yahoo Finance.",
                    )
                )

        for name, factory, guidance in tabs:
            frame = factory(self)
            self.add(frame, text=name)
            self._tab_names.append(name)
            self._tab_guidance[name] = guidance
            self._tab_widgets[name] = frame

    def initialise_guidance(self) -> None:
        """Invoke the guidance callback for the default tab."""

        if not self._tab_names:
            return
        self._dispatch_tab_change(0)

    def _handle_tab_change(self, _: tk.Event) -> None:
        index = self.index("current")
        self._dispatch_tab_change(index)

    def _dispatch_tab_change(self, index: int) -> None:
        name = self._tab_names[index]
        widget = self._tab_widgets.get(name)
        if widget is not None:
            on_activated = getattr(widget, "on_activated", None)
            if callable(on_activated):
                on_activated()
        if self._on_tab_change is None:
            return
        guidance = self._tab_guidance.get(name, "")
        self._on_tab_change(index, name, guidance)


def launch_app(*, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
    """Initialise and start the Tkinter main loop."""

    root = tk.Tk()
    root.title("Toptek Mission Control")
    root.geometry("1024x680")

    web_handle: WebFrontendHandle | None = None
    web_message: str | None = None
    web_config = configs.get("web_frontend", {})
    if isinstance(web_config, dict) and web_config.get("enabled"):
        web_logger = utils.build_logger("WebFrontend")
        port_value = web_config.get("port")
        try:
            port = int(port_value) if port_value is not None else None
        except (TypeError, ValueError):
            port = None
        auto_value = web_config.get("auto_open")
        if isinstance(auto_value, str):
            auto_open = auto_value.lower() in {"1", "true", "yes", "on"}
        else:
            auto_open = bool(auto_value)
        web_handle = launch_web_frontend(paths, port=port, auto_open=auto_open, logger=web_logger)
        if web_handle is None:
            web_message = "Web console assets missing. Run npm run build in toptek/ui/web."

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        # ``clam`` is widely available, but gracefully fallback if missing.
        pass

    root.configure(background=DARK_PALETTE["canvas"])
    style.configure(
        ".",
        background=DARK_PALETTE["canvas"],
        foreground=DARK_PALETTE["text"],
    )
    style.configure("TFrame", background=DARK_PALETTE["canvas"])
    style.configure(
        "TLabel", background=DARK_PALETTE["canvas"], foreground=DARK_PALETTE["text"]
    )
    style.configure("TNotebook", background=DARK_PALETTE["canvas"], borderwidth=0)
    style.configure(
        "Dashboard.TNotebook",
        background=DARK_PALETTE["canvas"],
        borderwidth=0,
        tabmargins=(0, 12, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text_muted"],
        padding=(16, 10),
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", DARK_PALETTE["surface_alt"]),
            ("active", DARK_PALETTE["surface_muted"]),
        ],
        foreground=[
            ("selected", DARK_PALETTE["text"]),
            ("active", DARK_PALETTE["text"]),
        ],
    )
    style.configure(
        "DashboardBackground.TFrame",
        background=DARK_PALETTE["canvas"],
    )
    style.configure(
        "AppContainer.TFrame",
        background=DARK_PALETTE["canvas"],
    )
    style.configure(
        "Header.TLabel",
        background=DARK_PALETTE["canvas"],
        foreground=DARK_PALETTE["text"],
        font=("Segoe UI", 20, "bold"),
    )
    style.configure(
        "SubHeader.TLabel",
        background=DARK_PALETTE["canvas"],
        foreground=DARK_PALETTE["text_muted"],
        font=("Segoe UI", 12),
    )
    style.configure(
        "Step.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["accent_alt"],
        font=("Segoe UI", 11),
    )
    style.configure(
        "Body.TLabel",
        background=DARK_PALETTE["canvas"],
        foreground=DARK_PALETTE["text"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "Surface.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "Muted.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text_muted"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "StatusInfo.TLabel",
        background=DARK_PALETTE["canvas"],
        foreground=DARK_PALETTE["accent_alt"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "SurfaceStatus.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["accent_alt"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "GuardBadge.TLabel",
        background=DARK_PALETTE["surface_alt"],
        foreground=DARK_PALETTE["accent_alt"],
        font=("Segoe UI", 10, "bold"),
        padding=(12, 4),
    )
    style.configure(
        "Guidance.TLabelframe",
        background=DARK_PALETTE["surface"],
        bordercolor=DARK_PALETTE["border"],
        relief="solid",
        borderwidth=1,
        padding=(12, 10),
    )
    style.configure(
        "Guidance.TLabelframe.Label",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text"],
        font=("Segoe UI", 11, "bold"),
    )
    style.configure(
        "Section.TLabelframe",
        background=DARK_PALETTE["surface"],
        bordercolor=DARK_PALETTE["border"],
        relief="solid",
        borderwidth=1,
        padding=(16, 12),
    )
    style.configure(
        "Section.TLabelframe.Label",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["accent_alt"],
        font=("Segoe UI", 11, "bold"),
    )
    style.configure(
        "DashboardCard.TFrame",
        background=DARK_PALETTE["surface"],
        bordercolor=DARK_PALETTE["border"],
        relief="solid",
        borderwidth=1,
        padding=(18, 16),
    )
    style.configure(
        "ChartContainer.TFrame",
        background=DARK_PALETTE["surface"],
        bordercolor=DARK_PALETTE["border_muted"],
        relief="solid",
        borderwidth=1,
        padding=(18, 16),
    )
    style.configure(
        "CardHeading.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text_muted"],
        font=("Segoe UI", 11, "bold"),
    )
    style.configure(
        "MetricValue.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["accent"],
        font=("Segoe UI", 22, "bold"),
    )
    style.configure(
        "MetricCaption.TLabel",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text_muted"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "Accent.TButton",
        background=DARK_PALETTE["accent"],
        foreground=DARK_PALETTE["canvas"],
        bordercolor=DARK_PALETTE["accent"],
        focusthickness=3,
        focuscolor=DARK_PALETTE["accent_alt"],
        padding=(16, 10),
    )
    style.configure(
        "Neutral.TButton",
        background=DARK_PALETTE["surface_alt"],
        foreground=DARK_PALETTE["text"],
        bordercolor=DARK_PALETTE["border"],
        focusthickness=3,
        focuscolor=DARK_PALETTE["accent_alt"],
        padding=(14, 10),
    )
    style.map(
        "Accent.TButton",
        background=[
            ("pressed", DARK_PALETTE["accent_active"]),
            ("active", DARK_PALETTE["accent_hover"]),
            ("disabled", DARK_PALETTE["border_muted"]),
        ],
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
            ("pressed", DARK_PALETTE["canvas"]),
            ("active", DARK_PALETTE["canvas"]),
        ],
    )
    style.map(
        "Neutral.TButton",
        background=[
            ("pressed", DARK_PALETTE["surface_muted"]),
            ("active", DARK_PALETTE["surface_alt"]),
            ("disabled", DARK_PALETTE["border_muted"]),
        ],
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
        ],
    )
    style.configure(
        "Input.TEntry",
        fieldbackground=DARK_PALETTE["surface_alt"],
        foreground=DARK_PALETTE["text"],
        bordercolor=DARK_PALETTE["border"],
        insertcolor=DARK_PALETTE["text"],
    )
    style.map(
        "Input.TEntry",
        fieldbackground=[
            ("readonly", DARK_PALETTE["surface"]),
            ("disabled", DARK_PALETTE["border_muted"]),
        ],
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
        ],
    )
    style.configure(
        "Input.TCombobox",
        fieldbackground=DARK_PALETTE["surface_alt"],
        background=DARK_PALETTE["surface_alt"],
        foreground=DARK_PALETTE["text"],
        bordercolor=DARK_PALETTE["border"],
    )
    style.map(
        "Input.TCombobox",
        fieldbackground=[
            ("readonly", DARK_PALETTE["surface_alt"]),
            ("disabled", DARK_PALETTE["border_muted"]),
        ],
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
        ],
    )
    style.configure(
        "Input.TSpinbox",
        fieldbackground=DARK_PALETTE["surface_alt"],
        foreground=DARK_PALETTE["text"],
        bordercolor=DARK_PALETTE["border"],
    )
    style.map(
        "Input.TSpinbox",
        fieldbackground=[
            ("disabled", DARK_PALETTE["border_muted"]),
        ],
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
        ],
    )
    style.configure(
        "Input.TRadiobutton",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text"],
    )
    style.map(
        "Input.TRadiobutton",
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
        ],
    )
    style.configure(
        "Input.TCheckbutton",
        background=DARK_PALETTE["surface"],
        foreground=DARK_PALETTE["text"],
    )
    style.map(
        "Input.TCheckbutton",
        foreground=[
            ("disabled", DARK_PALETTE["text_muted"]),
        ],
    )
    style.configure(
        "Accent.Horizontal.TProgressbar",
        background=DARK_PALETTE["accent"],
        troughcolor=DARK_PALETTE["surface"],
        bordercolor=DARK_PALETTE["border"],
    )

    root.option_add("*TCombobox*Listbox.background", DARK_PALETTE["surface"])
    root.option_add("*TCombobox*Listbox.foreground", DARK_PALETTE["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", DARK_PALETTE["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", DARK_PALETTE["canvas"])

    container = ttk.Frame(root, padding=16, style="AppContainer.TFrame")
    container.pack(fill=tk.BOTH, expand=True)

    header = ttk.Frame(container, style="DashboardBackground.TFrame")
    header.pack(fill=tk.X, pady=(0, 12))

    header_left = ttk.Frame(header, style="DashboardBackground.TFrame")
    header_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(
        header_left, text="Project X · Manual Trading Copilot", style="Header.TLabel"
    ).pack(anchor=tk.W)
    ttk.Label(
        header_left,
        text="Follow the guided workflow from credentials to Topstep-compliant trade plans.",
        style="SubHeader.TLabel",
    ).pack(anchor=tk.W)

    if web_handle is not None:
        ttk.Button(
            header,
            text="Open Web Console",
            style="Accent.TButton",
            command=web_handle.open,
        ).pack(side=tk.RIGHT)
    elif web_message:
        ttk.Label(header, text=web_message, style="Muted.TLabel").pack(
            side=tk.RIGHT, anchor=tk.E
        )

    guidance_card = ttk.Labelframe(
        container, text="Mission Checklist", style="Guidance.TLabelframe"
    )
    guidance_card.pack(fill=tk.X, pady=(0, 12))

    step_label = ttk.Label(guidance_card, style="Step.TLabel")
    step_label.pack(anchor=tk.W)
    progress = ttk.Progressbar(
        guidance_card,
        maximum=1,
        mode="determinate",
        length=220,
        style="Accent.Horizontal.TProgressbar",
    )
    progress.pack(anchor=tk.W, pady=(8, 0))

    def handle_tab_change(index: int, name: str, guidance: str) -> None:
        step_label.config(text=f"{guidance}\n→ Current focus: {name} tab")
        progress.configure(value=min(index, float(progress.cget("maximum"))))

    notebook = ToptekApp(container, configs, paths, on_tab_change=handle_tab_change)
    notebook.pack(fill=tk.BOTH, expand=True)
    tab_count = len(notebook.tabs())
    if tab_count > 1:
        progress.configure(maximum=float(tab_count - 1))
    notebook.initialise_guidance()

    def _on_close() -> None:
        if web_handle is not None:
            web_handle.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    try:
        root.mainloop()
    finally:
        if web_handle is not None:
            web_handle.stop()


__all__ = ["launch_app", "ToptekApp"]
