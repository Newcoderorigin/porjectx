"""Elite HUD Tkinter bootstrap for Toptek."""

from __future__ import annotations

import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, TOP, X
from ttkbootstrap.toast import ToastNotification

from core import utils

from . import widgets


class CommandBus:
    """Simple publish/subscribe command bus for the GUI."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[..., None]]] = defaultdict(list)

    def send_command(self, name: str, /, **kwargs: Any) -> None:
        for callback in list(self._subscribers.get(name, [])):
            callback(**kwargs)

    def subscribe_to(self, signal: str, callback: Callable[..., None]) -> None:
        self._subscribers.setdefault(signal, []).append(callback)


class ToastManager:
    """Helper for showing toast notifications via ttkbootstrap."""

    def __init__(self, master: tk.Misc) -> None:
        self.master = master

    def _show(self, title: str, message: str, bootstyle: str) -> None:
        ToastNotification(
            title=title,
            message=message,
            duration=4000,
            bootstyle=bootstyle,
            position=("ne"),
            master=self.master,
        ).show()

    def info(self, message: str) -> None:
        self._show("Toptek", message, "info")

    def success(self, message: str) -> None:
        self._show("Toptek", message, "success")

    def warning(self, message: str) -> None:
        self._show("Toptek", message, "warning")

    def error(self, message: str) -> None:
        self._show("Toptek", message, "danger")


class GuideOverlay:
    """Translucent overlay that renders contextual coach-mark bubbles."""

    def __init__(self, master: tk.Misc) -> None:
        self.master = master
        self._layer = tb.Toplevel(master)
        self._layer.withdraw()
        self._layer.overrideredirect(True)
        self._layer.attributes("-topmost", True)
        self._layer.attributes("-alpha", 0.0)
        self._text = tk.StringVar(value="")
        self._frame = tb.Frame(self._layer, padding=12, bootstyle="info")
        self._label = tb.Label(
            self._frame,
            textvariable=self._text,
            bootstyle="inverse-info",
            wraplength=260,
            justify=tk.LEFT,
        )
        self._frame.pack()
        self._label.pack()
        self._dismiss_callback: Callable[[], None] | None = None
        self._layer.bind("<Button-1>", self._on_click)

    def show(self, widget: tk.Widget, text: str, *, on_dismiss: Callable[[], None] | None = None) -> None:
        widget.update_idletasks()
        x = widget.winfo_rootx() + widget.winfo_width() + 16
        y = widget.winfo_rooty()
        self._layer.geometry(f"+{x}+{y}")
        self._text.set(text)
        self._dismiss_callback = on_dismiss
        self._layer.deiconify()
        self._layer.attributes("-alpha", 0.92)

    def hide(self) -> None:
        self._layer.withdraw()
        self._layer.attributes("-alpha", 0.0)
        self._dismiss_callback = None

    def _on_click(self, *_: Any) -> None:
        if self._dismiss_callback:
            self._dismiss_callback()
        else:
            self.hide()


@dataclass
class StatusState:
    """Track headline status line values."""

    mode: tk.StringVar
    account: tk.StringVar
    symbol: tk.StringVar
    timeframe: tk.StringVar


class AppFrame(tb.Frame):
    """Elite HUD frame combining navigation, rail, and tab content."""

    def __init__(
        self,
        master: tb.Window,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        bus: CommandBus,
        first_run: bool = False,
    ) -> None:
        super().__init__(master, padding=0, bootstyle="dark")
        self.configs = configs
        self.paths = paths
        self.bus = bus
        self.toast = ToastManager(master)
        self.overlay = GuideOverlay(master)
        self.status_state = StatusState(
            mode=tk.StringVar(value="Paper"),
            account=tk.StringVar(value="—"),
            symbol=tk.StringVar(value="—"),
            timeframe=tk.StringVar(value=configs["app"].get("default_timeframe", "5m")),
        )
        self.status_summary = tk.StringVar()
        self._bind_status_updates()
        self.guide_drawer: widgets.GuideDrawer | None = None
        self.coach: widgets.CoachMarks | None = None
        self._build_layout()
        self._wire_bus()
        if first_run:
            master.after(800, self._start_coach_marks)

    def _bind_status_updates(self) -> None:
        for var in (self.status_state.mode, self.status_state.account, self.status_state.symbol, self.status_state.timeframe):
            var.trace_add("write", lambda *_: self._update_status())
        self._update_status()

    def _update_status(self) -> None:
        text = (
            f"Mode: {self.status_state.mode.get()} • "
            f"Account: {self.status_state.account.get()} • "
            f"Symbol: {self.status_state.symbol.get()} • "
            f"Timeframe: {self.status_state.timeframe.get()}"
        )
        self.status_summary.set(text)

    def _build_layout(self) -> None:
        self.pack(fill=BOTH, expand=True)
        container = tb.Panedwindow(self, orient=tk.HORIZONTAL, bootstyle="dark")
        container.pack(fill=BOTH, expand=True)

        self.command_rail = tb.Frame(container, padding=(12, 16), bootstyle="secondary")
        container.add(self.command_rail, weight=0)

        central = tb.Frame(container, padding=0, bootstyle="dark")
        container.add(central, weight=1)

        self.guide_drawer = widgets.GuideDrawer(container, width=260)
        container.add(self.guide_drawer, weight=0)

        self._build_command_rail()

        self.notebook = tb.Notebook(central, bootstyle="dark")
        self.notebook.pack(fill=BOTH, expand=True, padx=12, pady=(12, 6))

        bottom = tb.Frame(central, padding=(12, 6))
        bottom.pack(fill=X, side=TOP)
        tb.Label(bottom, textvariable=self.status_summary, bootstyle="secondary", font=("Consolas", 10)).pack(anchor="w")

        self.tabs: Dict[str, widgets.EliteTab] = {}
        for name, tab_cls in (
            ("Login", widgets.LoginTab),
            ("Research", widgets.ResearchTab),
            ("Train", widgets.TrainTab),
            ("Backtest", widgets.BacktestTab),
            ("Trade", widgets.TradeTab),
        ):
            tab = tab_cls(
                self.notebook,
                self.configs,
                self.paths,
                bus=self.bus,
                toast=self.toast,
                overlay=self.overlay,
                guide_drawer=self.guide_drawer,
                status=self.status_state,
            )
            self.notebook.add(tab, text=name)
            self.tabs[name] = tab

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._on_tab_changed()
        self.coach = widgets.CoachMarks(self.master, self.overlay)

    def _build_command_rail(self) -> None:
        title = tb.Label(
            self.command_rail,
            text="COMMAND",
            bootstyle="inverse-secondary",
            font=("BankGothic Md BT", 11),
        )
        title.pack(anchor="center", pady=(0, 12))

        buttons = [
            ("🔑", "Login", "Login"),
            ("🔎", "Fetch Bars", "Research"),
            ("📊", "Train", "Train"),
            ("🧪", "Backtest", "Backtest"),
            ("🟢/🔴", "Paper/Live", "Trade"),
        ]

        for icon, text, target in buttons:
            btn = tb.Button(
                self.command_rail,
                text=f"{icon}\n{text}",
                width=12,
                bootstyle="info-outline",
                command=lambda t=target: self._focus_tab(t),
            )
            btn.pack(pady=6, fill=X)

    def _focus_tab(self, name: str) -> None:
        tab = self.tabs.get(name)
        if not tab:
            return
        self.notebook.select(tab)
        tab.handle_command("primary")

    def _on_tab_changed(self, *_: Any) -> None:
        current = self.notebook.select()
        tab = self.notebook.nametowidget(current)
        if isinstance(tab, widgets.EliteTab) and self.guide_drawer is not None:
            self.guide_drawer.set_steps(tab.guide_steps)

    def _wire_bus(self) -> None:
        self.bus.subscribe_to("status:update", self._handle_status_update)

    def _handle_status_update(self, **payload: Any) -> None:
        if "mode" in payload:
            self.status_state.mode.set(payload["mode"])
        if "account" in payload:
            self.status_state.account.set(payload["account"])
        if "symbol" in payload:
            self.status_state.symbol.set(payload["symbol"])
        if "timeframe" in payload:
            self.status_state.timeframe.set(payload["timeframe"])

    def _start_coach_marks(self) -> None:
        if self.coach is None:
            return
        steps = []
        login = self.tabs.get("Login")
        research = self.tabs.get("Research")
        train = self.tabs.get("Train")
        backtest = self.tabs.get("Backtest")
        trade = self.tabs.get("Trade")
        if login:
            steps.append(("Enter API key", login.coach_targets.get("api_key")))
            steps.append(("Save credentials", login.coach_targets.get("save")))
            steps.append(("Login", login.coach_targets.get("login")))
        if research:
            steps.append(("Search symbol", research.coach_targets.get("search")))
            steps.append(("Preview bars", research.coach_targets.get("preview")))
        if train:
            steps.append(("Run train", train.coach_targets.get("train")))
        if backtest:
            steps.append(("Run backtest", backtest.coach_targets.get("run")))
        if trade:
            steps.append(("Start paper mode", trade.coach_targets.get("paper")))
        filtered_steps = [(text, widget) for text, widget in steps if widget is not None]
        if not filtered_steps:
            return
        self.coach.configure(filtered_steps)
        self.coach.start()


_BUS: CommandBus | None = None


def send_command(name: str, /, **kwargs: Any) -> None:
    """Send a GUI command to registered subscribers."""

    if _BUS is not None:
        _BUS.send_command(name, **kwargs)


def subscribe_to(signal: str, callback: Callable[..., None]) -> None:
    """Subscribe to GUI command notifications."""

    if _BUS is not None:
        _BUS.subscribe_to(signal, callback)


def launch_app(*, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths, first_run: bool = False) -> None:
    """Initialise and start the Elite Toptek Tkinter main loop."""

    global _BUS
    window = tb.Window(themename="cyborg")
    window.title("Toptek Elite HUD")
    window.geometry("1280x840")
    style = window.style
    style.configure("TNotebook", background="#10141b")
    style.configure("TFrame", background="#10141b")
    style.configure("info", foreground="#00E0FF")
    bus = CommandBus()
    _BUS = bus
    app = AppFrame(window, configs, paths, bus=bus, first_run=first_run)
    app.pack(fill=BOTH, expand=True)
    window.mainloop()


__all__ = [
    "launch_app",
    "send_command",
    "subscribe_to",
    "AppFrame",
    "CommandBus",
    "ToastManager",
    "GuideOverlay",
]
