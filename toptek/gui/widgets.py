"""Tkinter tab implementations for the Toptek GUI."""

from __future__ import annotations

import math
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, TypeVar, cast

import numpy as np

from core import backtest, features, model, utils
from core.data import sample_dataframe
from core.utils import json_dumps
from toptek.features import build_features
from toptek.risk import GuardReport, RiskEngine
from toptek.replay import ReplayBar, ReplaySimulator

from . import DARK_PALETTE, TEXT_WIDGET_DEFAULTS
try:  # pragma: no cover - fallback for legacy installs
    from .builder import invoke_tab_builder, MissingTabBuilderError  # type: ignore
except Exception:  # pragma: no cover - defensive compatibility
    class MissingTabBuilderError(RuntimeError):
        """Fallback error raised when a tab builder is missing."""

        def __init__(self, tab_name: str, attr: str = "_build") -> None:
            self.tab_name = tab_name
            self.attr = attr
            super().__init__(
                f"{tab_name} is missing a callable '{attr}()' layout builder."
            )

    def invoke_tab_builder(tab: object, *, attr: str = "_build") -> None:
        builder = getattr(tab, attr, None)
        if not callable(builder):
            raise MissingTabBuilderError(tab.__class__.__name__, attr)
        builder()
from .tradingview import TradingViewDefaults, TradingViewRouter


T = TypeVar("T")


class BaseTab(ttk.Frame):
    """Base class providing convenience utilities for tabs."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, style="DashboardBackground.TFrame")
        self.configs = configs
        self._ui_config = configs.get("ui", {})
        self.paths = paths
        self.logger = utils.build_logger(self.__class__.__name__)
        self.tv_router = tv_router

    def style_text_widget(self, widget: tk.Text) -> None:
        """Apply the shared dark theme to ``tk.Text`` widgets."""

        widget.configure(**TEXT_WIDGET_DEFAULTS)

    def log_event(self, message: str, *, level: str = "info") -> None:
        """Log an event using the tab's logger.

        Args:
            message: Event description.
            level: Logging level name to dispatch (defaults to ``info``).
        """

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def update_section(self, section: str, updates: Dict[str, object]) -> None:
        """Update the configuration state for *section* with *updates*."""

        self.configs.setdefault(section, {}).update(updates)

    def ui_setting(self, *keys: str, default: T) -> T:
        """Retrieve a nested UI configuration value with a fallback."""

        data: Any = self._ui_config
        for key in keys:
            if not isinstance(data, dict):
                return default
            data = data.get(key)
            if data is None:
                return default
        return cast(T, data)

    def tradingview_defaults(self) -> TradingViewDefaults:
        base = (
            self.tv_router.defaults()
            if self.tv_router and self.tv_router.enabled
            else TradingViewDefaults(
                symbol="ES=F", interval="5m", theme="dark", locale="en"
            )
        )
        stored = self.configs.get("tradingview")
        if isinstance(stored, dict):
            symbol = str(stored.get("symbol", base.symbol) or base.symbol)
            interval = str(stored.get("interval", base.interval) or base.interval)
            theme = str(stored.get("theme", base.theme) or base.theme)
            locale = str(stored.get("locale", base.locale) or base.locale)
            return TradingViewDefaults(
                symbol=symbol,
                interval=interval,
                theme=theme,
                locale=locale,
            )
        return base

    def _open_tradingview(
        self,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        theme: str | None = None,
        locale: str | None = None,
    ) -> str | None:
        if not self.tv_router or not self.tv_router.enabled:
            messagebox.showinfo(
                "TradingView disabled",
                "Enable the TradingView integration in config/app.yml to launch charts.",
            )
            return None
        defaults = self.tradingview_defaults()
        resolved_symbol = symbol or defaults.symbol
        resolved_interval = interval or defaults.interval
        resolved_theme = theme or defaults.theme
        resolved_locale = locale or defaults.locale
        try:
            url = self.tv_router.launch(
                symbol=resolved_symbol,
                interval=resolved_interval,
                theme=resolved_theme,
                locale=resolved_locale,
            )
        except RuntimeError as exc:
            messagebox.showwarning("TradingView", str(exc))
            return None
        payload = {
            "symbol": resolved_symbol,
            "interval": resolved_interval,
            "theme": resolved_theme,
            "locale": resolved_locale,
            "url": url,
        }
        self.update_section("tradingview", payload)
        self.log_event(f"TradingView launch → {url}")
        return url


class LiveChart(ttk.Frame):
    """Lightweight canvas-based price chart for replay sessions."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        max_points: int,
        decimals: int,
    ) -> None:
        super().__init__(master, style="ChartContainer.TFrame")
        self._max_points = max(10, int(max_points))
        self._decimals = max(0, int(decimals))
        self._values: deque[float] = deque(maxlen=self._max_points)
        self._timestamps: deque[datetime] = deque(maxlen=self._max_points)
        self._line_id: int | None = None
        self._canvas = tk.Canvas(
            self,
            background=DARK_PALETTE["surface_alt"],
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._canvas.bind("<Configure>", lambda _: self._render())

    def clear(self) -> None:
        self._values.clear()
        self._timestamps.clear()
        if self._line_id is not None:
            self._canvas.delete(self._line_id)
            self._line_id = None
        self._canvas.delete("axes")

    def push(self, bar: ReplayBar) -> None:
        price = self._extract_price(bar)
        if price is None:
            return
        self._values.append(price)
        self._timestamps.append(bar.timestamp)
        self._render()

    def last_price(self) -> float | None:
        return self._values[-1] if self._values else None

    def _extract_price(self, bar: ReplayBar) -> float | None:
        for key in ("close", "settle", "last", "price", "mid"):
            value = bar.data.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _render(self) -> None:
        if len(self._values) < 2:
            if self._line_id is not None:
                self._canvas.delete(self._line_id)
                self._line_id = None
            self._canvas.delete("axes")
            return
        width = max(self._canvas.winfo_width(), 32)
        height = max(self._canvas.winfo_height(), 32)
        values = list(self._values)
        low = min(values)
        high = max(values)
        if math.isclose(low, high):
            buffer = abs(low) * 0.001 or 1.0
            low -= buffer
            high += buffer
        padding = 14
        usable_height = max(height - padding * 2, 1)
        usable_width = max(width - padding * 2, 1)
        step = usable_width / (len(values) - 1)
        coords: list[float] = []
        for idx, value in enumerate(values):
            x = padding + idx * step
            ratio = (value - low) / (high - low)
            y = height - padding - ratio * usable_height
            coords.extend((x, y))
        if self._line_id is None:
            self._line_id = self._canvas.create_line(
                *coords,
                fill=DARK_PALETTE["accent"],
                width=2,
                tags=("series",),
            )
        else:
            self._canvas.coords(self._line_id, *coords)
        self._canvas.delete("axes")
        baseline = DARK_PALETTE["border_muted"]
        self._canvas.create_line(
            padding,
            height - padding,
            width - padding,
            height - padding,
            fill=baseline,
            width=1,
            tags=("axes",),
        )
        self._canvas.create_text(
            padding,
            padding,
            anchor=tk.NW,
            fill=DARK_PALETTE["text_muted"],
            text=f"{values[0]:.{self._decimals}f}",
            tags=("axes",),
        )
        self._canvas.create_text(
            width - padding,
            padding,
            anchor=tk.NE,
            fill=DARK_PALETTE["text_muted"],
            text=f"{values[-1]:.{self._decimals}f}",
            tags=("axes",),
        )


class DashboardTab(BaseTab):
    """Mission control overview with themed dashboard cards."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        self.workflow_value = tk.StringVar()
        self.workflow_caption = tk.StringVar()
        self.credentials_value = tk.StringVar()
        self.credentials_caption = tk.StringVar()
        self.training_value = tk.StringVar()
        self.training_caption = tk.StringVar()
        self.chart_summary = tk.StringVar()
        super().__init__(master, configs, paths)
        invoke_tab_builder(self)
        super().__init__(master, configs, paths, tv_router=tv_router)
        self._build()
        self._refresh_metrics()

    def _build(self) -> None:
        cards = ttk.Frame(self, style="DashboardBackground.TFrame")
        cards.pack(fill=tk.X, padx=10, pady=(12, 6))
        for column in range(3):
            cards.columnconfigure(column, weight=1, uniform="dashboard")

        workflow_card = ttk.Frame(cards, style="DashboardCard.TFrame")
        workflow_card.grid(row=0, column=0, padx=(0, 12), sticky="nsew")
        ttk.Label(workflow_card, text="Workflow", style="CardHeading.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            workflow_card, textvariable=self.workflow_value, style="MetricValue.TLabel"
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(
            workflow_card,
            textvariable=self.workflow_caption,
            style="MetricCaption.TLabel",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 0))

        credentials_card = ttk.Frame(cards, style="DashboardCard.TFrame")
        credentials_card.grid(row=0, column=1, padx=12, sticky="nsew")
        ttk.Label(
            credentials_card, text="Credentials", style="CardHeading.TLabel"
        ).pack(anchor=tk.W)
        ttk.Label(
            credentials_card,
            textvariable=self.credentials_value,
            style="MetricValue.TLabel",
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(
            credentials_card,
            textvariable=self.credentials_caption,
            style="MetricCaption.TLabel",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 0))

        training_card = ttk.Frame(cards, style="DashboardCard.TFrame")
        training_card.grid(row=0, column=2, padx=(12, 0), sticky="nsew")
        ttk.Label(training_card, text="Models", style="CardHeading.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            training_card, textvariable=self.training_value, style="MetricValue.TLabel"
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(
            training_card,
            textvariable=self.training_caption,
            style="MetricCaption.TLabel",
            wraplength=220,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 0))

        chart_frame = ttk.Frame(self, style="ChartContainer.TFrame")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 12))
        ttk.Label(chart_frame, text="Signal quality", style="CardHeading.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            chart_frame,
            textvariable=self.chart_summary,
            style="MetricCaption.TLabel",
            wraplength=760,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

    def on_activated(self) -> None:
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        sections = ("login", "research", "training", "backtest", "trade")
        completed = sum(1 for name in sections if self.configs.get(name))
        self.workflow_value.set(f"{completed}/{len(sections)}")
        if completed == len(sections):
            self.workflow_caption.set("Full mission ready. Proceed to guard checks.")
        else:
            remaining = len(sections) - completed
            self.workflow_caption.set(
                f"{remaining} step(s) outstanding in the mission checklist."
            )

        login_state = self.configs.get("login", {})
        verified = bool(login_state.get("verified"))
        saved = bool(login_state.get("saved"))
        if verified:
            self.credentials_value.set("Verified")
            self.credentials_caption.set(
                "Keys validated locally. Data requests unlocked."
            )
        elif saved:
            self.credentials_value.set("Saved")
            self.credentials_caption.set(
                "Run verification to confirm API reachability."
            )
        else:
            self.credentials_value.set("Pending")
            self.credentials_caption.set("Store sandbox credentials in the Login tab.")

        training_state = self.configs.get("training", {})
        model_name = training_state.get("model")
        if model_name:
            self.training_value.set(str(model_name).capitalize())
            threshold_value = self._coerce_float(training_state.get("threshold"))
            if threshold_value is not None:
                self.training_caption.set(
                    f"Decision threshold at {threshold_value:.2f}. Backtest next."
                )
            else:
                self.training_caption.set(
                    "Model artefact refreshed. Validate via backtest."
                )
        else:
            self.training_value.set("Untrained")
            self.training_caption.set("Run Train tab to fit a baseline classifier.")

        backtest_state = self.configs.get("backtest", {})
        expectancy = self._coerce_float(backtest_state.get("expectancy"))
        sharpe = self._coerce_float(backtest_state.get("sharpe"))
        if expectancy is not None and sharpe is not None:
            self.chart_summary.set(
                (
                    f"Latest backtest expectancy {expectancy:+.2f} with Sharpe {sharpe:.2f}. "
                    "Tune risk parameters before trading."
                )
            )
        else:
            self.chart_summary.set(
                "No simulations yet. Run the Backtest tab to evaluate expectancy before manual execution."
            )

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        """Best-effort float conversion used for dashboard metrics."""

        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None


class LoginTab(BaseTab):
    """Login tab that manages .env configuration."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, configs, paths)
        invoke_tab_builder(self)
        super().__init__(master, configs, paths, tv_router=tv_router)
        self._build()

    def _build(self) -> None:
        intro = ttk.LabelFrame(
            self,
            text="Step 1 · Secure your environment",
            style="Section.TLabelframe",
        )
        intro.pack(fill=tk.X, padx=10, pady=(12, 6))
        ttk.Label(
            intro,
            text=(
                "Paste sandbox credentials or API keys. Nothing leaves your machine. "
                "Use the guided Save + Verify buttons to confirm readiness before moving on."
            ),
            wraplength=520,
            justify=tk.LEFT,
            style="Surface.TLabel",
        ).pack(anchor=tk.W)

        form = ttk.Frame(self, style="DashboardBackground.TFrame")
        form.pack(padx=10, pady=6, fill=tk.X)
        self.vars = {
            "PX_BASE_URL": tk.StringVar(value=self._env_value("PX_BASE_URL")),
            "PX_MARKET_HUB": tk.StringVar(value=self._env_value("PX_MARKET_HUB")),
            "PX_USER_HUB": tk.StringVar(value=self._env_value("PX_USER_HUB")),
            "PX_USERNAME": tk.StringVar(value=self._env_value("PX_USERNAME")),
            "PX_API_KEY": tk.StringVar(value=self._env_value("PX_API_KEY")),
        }
        for row, (label, var) in enumerate(self.vars.items()):
            ttk.Label(form, text=label, style="Body.TLabel").grid(
                row=row, column=0, sticky=tk.W, padx=4, pady=4
            )
            ttk.Entry(form, textvariable=var, width=60, style="Input.TEntry").grid(
                row=row, column=1, padx=4, pady=4
            )
        actions = ttk.Frame(self, style="DashboardBackground.TFrame")
        actions.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Button(
            actions, text="Save .env", style="Accent.TButton", command=self._save_env
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            actions,
            text="Verify entries",
            style="Neutral.TButton",
            command=self._verify_env,
        ).pack(side=tk.LEFT)
        login_idle = self.ui_setting(
            "status", "login", "idle", default="Awaiting verification"
        )
        self.status = ttk.Label(actions, text=login_idle, style="StatusInfo.TLabel")
        self.status.pack(side=tk.LEFT, padx=12)

    def _env_value(self, key: str) -> str:
        return os.environ.get(key, "")

    def _save_env(self) -> None:
        env_path = self.paths.root / ".env"
        with env_path.open("w", encoding="utf-8") as handle:
            for key, var in self.vars.items():
                handle.write(f"{key}={var.get()}\n")
        messagebox.showinfo("Settings", f"Saved credentials to {env_path}")
        self.update_section("login", {"saved": True, "verified": False})
        saved_msg = self.ui_setting(
            "status",
            "login",
            "saved",
            default="Saved. Run verification to confirm access.",
        )
        self.status.config(text=saved_msg, foreground=DARK_PALETTE["warning"])

    def _verify_env(self) -> None:
        missing = [key for key, var in self.vars.items() if not var.get().strip()]
        if missing:
            details = ", ".join(missing)
            self.update_section("login", {"verified": False})
            self.status.config(
                text=f"Missing: {details}", foreground=DARK_PALETTE["danger"]
            )
            messagebox.showwarning("Verification", f"Provide values for: {details}")
            return
        self.update_section("login", {"saved": True, "verified": True})
        verified_msg = self.ui_setting(
            "status",
            "login",
            "verified",
            default="All keys present. Proceed to Research ▶",
        )
        self.status.config(text=verified_msg, foreground=DARK_PALETTE["success"])
        messagebox.showinfo(
            "Verification",
            "Environment entries look complete. Continue to the next tab.",
        )


class ResearchTab(BaseTab):
    """Research tab to preview sample data."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, configs, paths)
        invoke_tab_builder(self)
        super().__init__(master, configs, paths, tv_router=tv_router)
        self._build()

    def _build(self) -> None:
        self._tv_status: ttk.Label | None = None
        controls = ttk.LabelFrame(
            self,
            text="Step 2 · Research console",
            style="Section.TLabelframe",
        )
        controls.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(
            controls,
            text="1) Choose your focus market and timeframe. 2) Pull sample data to inspect structure and features.",
            wraplength=520,
            justify=tk.LEFT,
            style="Surface.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))

        shell_symbol = self.ui_setting("shell", "symbol", default="ES=F")
        shell_interval = self.ui_setting("shell", "interval", default="5m")
        shell_bars = int(self.ui_setting("shell", "research_bars", default=240))
        self.symbol_var = tk.StringVar(value=shell_symbol)
        self.timeframe_var = tk.StringVar(value=shell_interval)
        self.bars_var = tk.IntVar(value=shell_bars)

        ttk.Label(controls, text="Symbol", style="Surface.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6)
        )
        ttk.Entry(
            controls, textvariable=self.symbol_var, width=12, style="Input.TEntry"
        ).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(controls, text="Timeframe", style="Surface.TLabel").grid(
            row=1, column=2, sticky=tk.W, padx=(12, 6)
        )
        ttk.Combobox(
            controls,
            textvariable=self.timeframe_var,
            values=("1m", "5m", "15m", "1h", "4h", "1d"),
            state="readonly",
            width=8,
            style="Input.TCombobox",
        ).grid(row=1, column=3, sticky=tk.W)

        ttk.Label(controls, text="Bars", style="Surface.TLabel").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0)
        )
        ttk.Spinbox(
            controls,
            from_=60,
            to=1000,
            increment=60,
            textvariable=self.bars_var,
            width=10,
            style="Input.TSpinbox",
        ).grid(row=2, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Button(
            controls,
            text="Load sample bars",
            style="Accent.TButton",
            command=self._load_sample,
        ).grid(row=2, column=3, padx=(12, 0), pady=(6, 0))

        controls.columnconfigure(1, weight=1)

        if self.tv_router and self.tv_router.is_tab_enabled("research"):
            ttk.Button(
                controls,
                text="Open TradingView chart (Ctrl+Shift+T)",
                style="Neutral.TButton",
                command=self._open_research_tradingview,
            ).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(12, 0))
            defaults = self.tradingview_defaults()
            hint = (
                f"Launches TradingView with {defaults.symbol} at {defaults.interval} "
                "unless you override the symbol/timeframe above."
            )
            self._tv_status = ttk.Label(
                controls,
                text=hint,
                style="MetricCaption.TLabel",
                wraplength=520,
                justify=tk.LEFT,
            )
            self._tv_status.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

        self.text = tk.Text(self, height=18)
        self.style_text_widget(self.text)
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))
        self.summary = ttk.Label(
            self, anchor=tk.W, justify=tk.LEFT, style="Body.TLabel"
        )
        self.summary.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _load_sample(self) -> None:
        try:
            bars = int(self.bars_var.get())
        except (TypeError, ValueError):
            bars = 240
        bars = max(60, min(bars, 1000))
        df = sample_dataframe(bars)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, df.tail(12).to_string())
        if self._tv_status is not None:
            symbol = self.symbol_var.get()
            interval = self.timeframe_var.get()
            self._tv_status.config(
                text=(
                    f"Latest sample pulled for {symbol} ({interval}). "
                    "Use the TradingView button to mirror the selection."
                )
            )

    def _open_research_tradingview(self) -> None:
        symbol = str(self.symbol_var.get())
        interval = str(self.timeframe_var.get())
        url = self._open_tradingview(symbol=symbol, interval=interval)
        if url and self._tv_status is not None:
            self._tv_status.config(
                text=f"Opened TradingView for {symbol} ({interval}). URL copied to logs."
            )

        feat_map = features.compute_features(df)
        latest = -1
        atr = float(
            np.nan_to_num(feat_map.get("atr_14", np.array([0.0]))[latest], nan=0.0)
        )
        rsi = float(
            np.nan_to_num(feat_map.get("rsi_14", np.array([50.0]))[latest], nan=50.0)
        )
        vol = float(
            np.nan_to_num(
                feat_map.get("volatility_close", np.array([0.0]))[latest], nan=0.0
            )
        )
        trend = (
            "uptrend"
            if df["close"].tail(30).mean() > df["close"].tail(90).mean()
            else "down/sideways"
        )
        self.summary.config(
            text=(
                f"Symbol {self.symbol_var.get()} · {self.timeframe_var.get()} — ATR14 {atr:.2f} · RSI14 {rsi:.1f} · "
                f"20-bar vol {vol:.4f}\nRegime hint: {trend}. Move to Train when the setup looks promising."
            )
        )
        self.update_section(
            "research",
            {
                "symbol": self.symbol_var.get(),
                "timeframe": self.timeframe_var.get(),
                "bars": int(self.bars_var.get()),
                "atr": atr,
                "rsi": rsi,
                "volatility": vol,
                "trend": trend,
            },
        )


class TrainTab(BaseTab):
    """Training tab for running local models."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, configs, paths)
        invoke_tab_builder(self)
        super().__init__(master, configs, paths, tv_router=tv_router)
        self._build()

    def _build(self) -> None:
        config = ttk.LabelFrame(
            self,
            text="Step 3 · Model lab",
            style="Section.TLabelframe",
        )
        config.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(
            config,
            text="Select a model, choose lookback and optionally calibrate probabilities before saving the artefact.",
            wraplength=520,
            justify=tk.LEFT,
            style="Surface.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W)

        default_model = self.ui_setting("shell", "model", default="logistic")
        default_calibrate = bool(self.ui_setting("shell", "calibrate", default=True))
        default_lookback = int(self.ui_setting("shell", "lookback_bars", default=480))
        self.model_type = tk.StringVar(value=default_model)
        self.calibrate_var = tk.BooleanVar(value=default_calibrate)
        self.lookback_var = tk.IntVar(value=default_lookback)

        ttk.Label(config, text="Model", style="Surface.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Radiobutton(
            config,
            text="Logistic",
            value="logistic",
            variable=self.model_type,
            style="Input.TRadiobutton",
        ).grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Radiobutton(
            config,
            text="Gradient Boosting",
            value="gbm",
            variable=self.model_type,
            style="Input.TRadiobutton",
        ).grid(row=1, column=2, sticky=tk.W, pady=(8, 0))

        ttk.Label(config, text="Lookback bars", style="Surface.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Spinbox(
            config,
            from_=240,
            to=2000,
            increment=120,
            textvariable=self.lookback_var,
            width=10,
            style="Input.TSpinbox",
        ).grid(row=2, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(
            config,
            text="Calibrate probabilities",
            variable=self.calibrate_var,
            style="Input.TCheckbutton",
        ).grid(row=2, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Button(
            config,
            text="Train + Score",
            style="Accent.TButton",
            command=self._train_model,
        ).grid(row=2, column=3, padx=(12, 0), pady=(8, 0))

        config.columnconfigure(1, weight=1)

        self.output = tk.Text(self, height=12)
        self.style_text_widget(self.output)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))
        training_idle = self.ui_setting(
            "status", "training", "idle", default="Awaiting training run"
        )
        self.status = ttk.Label(
            self, text=training_idle, anchor=tk.W, style="StatusInfo.TLabel"
        )
        self.status.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _train_model(self) -> None:
        try:
            lookback = int(self.lookback_var.get())
        except (TypeError, ValueError):
            lookback = 480
        lookback = max(240, min(lookback, 2000))
        df = sample_dataframe(lookback)
        try:
            bundle = build_features(df, cache_dir=self.paths.cache)
        except ValueError as exc:
            self.status.config(
                text="Training aborted: feature pipeline returned no usable rows.",
                foreground=DARK_PALETTE["danger"],
            )
            messagebox.showwarning(
                "Training warning",
                (
                    "Training halted because the unified feature pipeline discarded all rows.\n"
                    f"Details: {exc}"
                ),
            )
            self.log_event(f"Feature pipeline failure: {exc}", level="error")
            return

        X = bundle.X
        y = bundle.y

        dropped = int(bundle.meta.get("dropped_rows", 0))
        if dropped:
            self.log_event(
                f"Feature pipeline dropped {dropped} rows before training",
                level="warning",
            )
        elif not len(bundle.meta.get("mask", [])):
            self.log_event(
                "Feature pipeline returned data without mask metadata", level="warning"
            )

        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            self.status.config(
                text="Training aborted: target labels lack class diversity.",
                foreground=DARK_PALETTE["danger"],
            )
            messagebox.showwarning(
                "Training warning",
                (
                    "Training requires at least two classes after cleaning the dataset.\n"
                    "Collect more data or adjust preprocessing to obtain both up and down samples."
                ),
            )
            return

        try:
            result = model.train_classifier(
                X, y, model_type=self.model_type.get(), models_dir=self.paths.models
            )
        except ValueError as exc:
            self.status.config(
                text="Training failed due to invalid feature matrix. Review warnings.",
                foreground=DARK_PALETTE["danger"],
            )
            messagebox.showwarning(
                "Training warning",
                (
                    "The classifier could not be trained with the current dataset.\n\n"
                    f"Details: {exc}"
                ),
            )
            self.log_event(f"Model training failed: {exc}", level="error")
            return
        # Ensure this is always defined for downstream payload/logging
        preprocessing = (getattr(result, "preprocessing", None) or {}).copy()

        calibrate_report = "skipped"
        calibration_detail: str | None = None
        calibration_failed = False
        if self.calibrate_var.get() and len(X) > 60:
            cal_size = max(60, int(len(X) * 0.2))
            X_cal = X[-cal_size:]
            y_cal = y[-cal_size:]
            calibrate_kwargs = {}
            if result.retained_columns is not None:
                calibrate_kwargs["feature_mask"] = result.retained_columns
                if result.original_feature_count is not None:
                    calibrate_kwargs["original_feature_count"] = (
                        result.original_feature_count
                    )
            try:
                calibrated_path = model.calibrate_classifier(
                    result.model_path,
                    (X_cal, y_cal),
                    **calibrate_kwargs,
                )
            except (ValueError, RuntimeError) as exc:
                calibrate_report = f"calibration failed: {exc}"
                calibration_detail = calibrate_report
                calibration_failed = True
                self.log_event(
                    f"Calibration failed for {result.model_path.name}: {exc}",
                    level="warning",
                )
                self.status.config(
                    text="Calibration skipped due to data quality. Review logs for details.",
                    foreground=DARK_PALETTE["danger"],
                )
                messagebox.showwarning(
                    "Calibration warning",
                    (
                        "Probability calibration failed with the current sample. "
                        "The model artefact remains uncalibrated.\n\n"
                        f"Details: {exc}"
                    ),
                )
            else:
                calibrate_report = f"calibrated → {calibrated_path.name}"
                calibration_detail = calibrate_report
                self.log_event(
                    f"Calibration completed for {result.model_path.name} → {calibrated_path.name}",
                    level="info",
                )
        calibrate_value = calibrate_report if not calibration_failed else "skipped"
        self.output.delete("1.0", tk.END)
        payload = {
            "model": self.model_type.get(),
            "metrics": result.metrics,
            "threshold": result.threshold,
            "preprocessing": preprocessing,
            "retained_columns": (
                list(result.retained_columns)
                if result.retained_columns is not None
                else None
            ),
            "original_feature_count": result.original_feature_count,
            "calibration": calibrate_value,
            "calibration_detail": calibration_detail,
        }
        self.output.insert(tk.END, json_dumps(payload))
        self.update_section("training", payload)
        if not calibration_failed:
            success_msg = self.ui_setting(
                "status",
                "training",
                "success",
                default="Model artefact refreshed. Continue to Backtest ▶",
            )
            self.status.config(
                text=success_msg,
                foreground=DARK_PALETTE["accent_alt"],
            )


class BacktestTab(BaseTab):
    """Backtesting tab with a simple equity curve display."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, configs, paths)
        invoke_tab_builder(self)
        super().__init__(master, configs, paths, tv_router=tv_router)
        self._build()

    def _build(self) -> None:
        controls = ttk.LabelFrame(
            self,
            text="Step 4 · Backtest",
            style="Section.TLabelframe",
        )
        controls.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(
            controls,
            text="Stress test expectancy against synthetic regimes before taking ideas live.",
            wraplength=520,
            justify=tk.LEFT,
            style="Surface.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W)

        default_sample = int(self.ui_setting("shell", "simulation_bars", default=720))
        default_strategy = self.ui_setting("shell", "playbook", default="momentum")
        self.sample_var = tk.IntVar(value=default_sample)
        self.strategy_var = tk.StringVar(value=default_strategy)

        ttk.Label(controls, text="Sample bars", style="Surface.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Spinbox(
            controls,
            from_=240,
            to=5000,
            increment=240,
            textvariable=self.sample_var,
            width=10,
            style="Input.TSpinbox",
        ).grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Label(controls, text="Playbook", style="Surface.TLabel").grid(
            row=1, column=2, sticky=tk.W, pady=(8, 0)
        )
        ttk.Combobox(
            controls,
            textvariable=self.strategy_var,
            values=("momentum", "mean_reversion"),
            state="readonly",
            width=16,
            style="Input.TCombobox",
        ).grid(row=1, column=3, sticky=tk.W, pady=(8, 0))
        ttk.Button(
            controls,
            text="Run sample backtest",
            style="Accent.TButton",
            command=self._run_backtest,
        ).grid(row=2, column=3, padx=(12, 0), pady=(8, 0))

        controls.columnconfigure(1, weight=1)

        self.output = tk.Text(self, height=14)
        self.style_text_widget(self.output)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))
        backtest_idle = self.ui_setting(
            "status", "backtest", "idle", default="No simulations yet"
        )
        self.status = ttk.Label(
            self, text=backtest_idle, anchor=tk.W, style="StatusInfo.TLabel"
        )
        self.status.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _run_backtest(self) -> None:
        try:
            sample = int(self.sample_var.get())
        except (TypeError, ValueError):
            sample = 720
        sample = max(240, min(sample, 5000))
        df = sample_dataframe(sample)
        returns = np.log(df["close"]).diff().fillna(0).to_numpy()
        if self.strategy_var.get() == "momentum":
            signals = (returns > 0).astype(int)
            playbook = "Momentum bias — follow strength"
        else:
            signals = (returns < 0).astype(int)
            playbook = "Mean reversion — fade spikes"
        result = backtest.run_backtest(returns, signals)
        payload = {
            "hit_rate": result.hit_rate,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "expectancy": result.expectancy,
            "playbook": playbook,
        }
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json_dumps(payload))
        success_msg = self.ui_setting(
            "status",
            "backtest",
            "success",
            default="Sim complete. If expectancy holds, draft a manual trade plan ▶",
        )
        self.status.config(text=success_msg, foreground=DARK_PALETTE["accent_alt"])
        self.update_section("backtest", payload)


class ReplayTab(BaseTab):
    """Replay tab wiring the simulator feed into the live chart."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        self.file_var = tk.StringVar()
        self.format_var = tk.StringVar(value="auto")
        self.speed_var = tk.StringVar(value="1.0")
        self.seek_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar()
        self._queue: Queue[ReplayBar] = Queue()
        self._poll_job: str | None = None
        self._simulator: ReplaySimulator | None = None
        self._processed = 0
        self._total = 0
        self._suspend_seek = False
        self._user_scrubbing = False
        self._poll_interval = 100
        self._chart: LiveChart | None = None
        self._price_decimals = 2
        super().__init__(master, configs, paths, tv_router=tv_router)
        self._price_decimals = int(
            self.ui_setting("chart", "price_decimals", default=2)
        )
        default_status = self.ui_setting(
            "status",
            "replay",
            "idle",
            default="Load a dataset to begin playback.",
        )
        self.status_var.set(default_status)
        default_dataset = (self.paths.cache / "replay.parquet").resolve()
        self.file_var.set(str(default_dataset))
        self._poll_interval = max(
            16, int(1000 / int(self.ui_setting("chart", "fps", default=12)))
        )
        invoke_tab_builder(self)

class TradingViewTab(BaseTab):
    """Dedicated tab exposing TradingView launch controls."""

    _HOTKEY = "<Control-Shift-T>"

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, configs, paths, tv_router=tv_router)
        self.symbol_var = tk.StringVar()
        self.interval_var = tk.StringVar()
        self.theme_var = tk.StringVar()
        self.locale_var = tk.StringVar()
        self._status: ttk.Label | None = None
        self._hotkey_bound = False
        self._build()
        self._bind_hotkey()

    def destroy(self) -> None:  # pragma: no cover - Tk lifecycle
        self._unbind_hotkey()
        super().destroy()

    def _bind_hotkey(self) -> None:
        if self._hotkey_bound:
            return
        self.bind_all(self._HOTKEY, self._handle_hotkey, add="+")
        self._hotkey_bound = True

    def _unbind_hotkey(self) -> None:
        if not self._hotkey_bound:
            return
        self.unbind_all(self._HOTKEY)
        self._hotkey_bound = False

    def _build(self) -> None:
        defaults = self.tradingview_defaults()
        self.symbol_var.set(defaults.symbol)
        self.interval_var.set(defaults.interval)
        self.theme_var.set(defaults.theme)
        self.locale_var.set(defaults.locale)

        frame = ttk.LabelFrame(
            self,
            text="TradingView launchpad",
            style="Section.TLabelframe",
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(12, 12))

        if not self.tv_router or not self.tv_router.enabled:
            ttk.Label(
                frame,
                text=(
                    "Set tv.enabled: true in config/app.yml or TOPTEK_TV_ENABLED=1 in "
                    ".env to activate TradingView integration."
                ),
                wraplength=540,
                justify=tk.LEFT,
                style="MetricCaption.TLabel",
            ).pack(anchor=tk.W, pady=6)
            return

        control_grid = ttk.Frame(frame, style="DashboardBackground.TFrame")
        control_grid.pack(fill=tk.X, pady=(6, 12))
        ttk.Label(control_grid, text="Symbol", style="Surface.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 6)
        )
        ttk.Entry(
            control_grid,
            textvariable=self.symbol_var,
            width=16,
            style="Input.TEntry",
        ).grid(row=0, column=1, sticky=tk.W, pady=(0, 6), padx=(6, 12))

        ttk.Label(control_grid, text="Interval", style="Surface.TLabel").grid(
            row=0, column=2, sticky=tk.W, pady=(0, 6)
        )
        ttk.Entry(
            control_grid,
            textvariable=self.interval_var,
            width=10,
            style="Input.TEntry",
        ).grid(row=0, column=3, sticky=tk.W, pady=(0, 6), padx=(6, 0))

        ttk.Label(control_grid, text="Theme", style="Surface.TLabel").grid(
            row=1, column=0, sticky=tk.W
        )
        ttk.Combobox(
            control_grid,
            textvariable=self.theme_var,
            values=("dark", "light"),
            state="readonly",
            width=12,
            style="Input.TCombobox",
        ).grid(row=1, column=1, sticky=tk.W, padx=(6, 12))

        ttk.Label(control_grid, text="Locale", style="Surface.TLabel").grid(
            row=1, column=2, sticky=tk.W
        )
        ttk.Entry(
            control_grid,
            textvariable=self.locale_var,
            width=10,
            style="Input.TEntry",
        ).grid(row=1, column=3, sticky=tk.W, padx=(6, 0))

        favourites = self.tv_router.favorites if self.tv_router else []
        if favourites:
            ttk.Label(
                control_grid, text="Favorites", style="Surface.TLabel"
            ).grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
            fav_combo = ttk.Combobox(
                control_grid,
                values=[item["label"] for item in favourites],
                state="readonly",
                width=28,
                style="Input.TCombobox",
            )
            fav_combo.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=(6, 0), pady=(8, 0))

            def _apply_favorite(_: tk.Event) -> None:
                selection = fav_combo.current()
                if selection < 0:
                    return
                entry = favourites[selection]
                self.symbol_var.set(entry["symbol"])
                self.interval_var.set(entry["interval"])

            fav_combo.bind("<<ComboboxSelected>>", _apply_favorite, add="+")

        ttk.Button(
            frame,
            text="Launch TradingView (Ctrl+Shift+T)",
            style="Accent.TButton",
            command=self._open_tradingview_tab,
        ).pack(anchor=tk.W, padx=4, pady=(6, 0))

        ttk.Label(
            frame,
            text=(
                "TradingView data is provided under their Terms of Service. Always "
                "display the attribution footer and confirm symbols manually before "
                "placing orders."
            ),
            wraplength=560,
            justify=tk.LEFT,
            style="MetricCaption.TLabel",
        ).pack(anchor=tk.W, padx=4, pady=(8, 0))

        ttk.Label(
            frame,
            text=(
                "Tip: Use the Trading tab or Research tab buttons to keep symbol "
                "and interval selections synced across the workflow."
            ),
            wraplength=560,
            justify=tk.LEFT,
            style="MetricCaption.TLabel",
        ).pack(anchor=tk.W, padx=4, pady=(4, 0))

        self._status = ttk.Label(
            frame,
            text="Press the launch button or Ctrl+Shift+T to open TradingView.",
            style="StatusInfo.TLabel",
        )
        self._status.pack(anchor=tk.W, padx=4, pady=(10, 0))

    def _open_tradingview_tab(self) -> None:
        url = self._open_tradingview(
            symbol=self.symbol_var.get(),
            interval=self.interval_var.get(),
            theme=self.theme_var.get(),
            locale=self.locale_var.get(),
        )
        if url and self._status is not None:
            self._status.config(
                text=f"TradingView launched. Review the browser window at {url}."
            )

    def _handle_hotkey(self, _: tk.Event) -> str:
        self._open_tradingview_tab()
        return "break"

    def destroy(self) -> None:
        self._cancel_poll()
        self._stop_simulator()
        super().destroy()

    def _build(self) -> None:
        controls = ttk.LabelFrame(
            self,
            text="Step 5 · Replay",
            style="Section.TLabelframe",
        )
        controls.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(controls, text="Dataset", style="Surface.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        path_entry = ttk.Entry(
            controls,
            textvariable=self.file_var,
            width=46,
            style="Input.TEntry",
        )
        path_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=(8, 8))
        ttk.Button(
            controls,
            text="Browse",
            style="Neutral.TButton",
            command=self._browse_dataset,
        ).grid(row=0, column=4, padx=(0, 0))

        ttk.Label(controls, text="Format", style="Surface.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        format_combo = ttk.Combobox(
            controls,
            textvariable=self.format_var,
            values=("auto", "csv", "parquet"),
            state="readonly",
            width=12,
            style="Input.TCombobox",
        )
        format_combo.grid(row=1, column=1, sticky=tk.W, padx=(8, 12), pady=(8, 0))

        ttk.Label(controls, text="Speed (×)", style="Surface.TLabel").grid(
            row=1, column=2, sticky=tk.W, pady=(8, 0)
        )
        speed_combo = ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=("0.25", "0.5", "1.0", "1.5", "2.0", "3.0", "4.0"),
            width=10,
            style="Input.TCombobox",
        )
        speed_combo.grid(row=1, column=3, sticky=tk.W, pady=(8, 0))
        speed_combo.bind("<<ComboboxSelected>>", self._set_speed)
        speed_combo.bind("<FocusOut>", self._set_speed)
        speed_combo.bind("<Return>", self._set_speed)

        ttk.Button(
            controls, text="Start", style="Accent.TButton", command=self._start_replay
        ).grid(row=1, column=4, padx=(12, 0), pady=(8, 0))

        ttk.Button(
            controls,
            text="Pause",
            style="Neutral.TButton",
            command=self._pause_replay,
        ).grid(row=2, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Button(
            controls,
            text="Resume",
            style="Neutral.TButton",
            command=self._resume_replay,
        ).grid(row=2, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Button(
            controls,
            text="Stop",
            style="Neutral.TButton",
            command=self._stop_replay,
        ).grid(row=2, column=3, sticky=tk.W, pady=(8, 0))

        ttk.Label(controls, text="Position", style="Surface.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=(10, 0)
        )
        self.seek_scale = ttk.Scale(
            controls,
            from_=0,
            to=100,
            variable=self.seek_var,
        )
        self.seek_scale.grid(
            row=3, column=1, columnspan=4, sticky=tk.EW, padx=(8, 0), pady=(10, 0)
        )
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_start)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)

        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=1)

        chart_container = ttk.Frame(self, style="DashboardBackground.TFrame")
        chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 6))
        self._chart = LiveChart(
            chart_container,
            max_points=int(self.ui_setting("chart", "max_points", default=180)),
            decimals=self._price_decimals,
        )
        self._chart.pack(fill=tk.BOTH, expand=True)

        self.output = tk.Text(self, height=10)
        self.style_text_widget(self.output)
        self.output.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 4))
        self.output.insert(
            tk.END,
            "Select a dataset and click Start to stream the recording through the chart.",
        )

        self.status = ttk.Label(
            self,
            textvariable=self.status_var,
            style="StatusInfo.TLabel",
            anchor=tk.W,
        )
        self.status.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _browse_dataset(self) -> None:
        current = Path(self.file_var.get()).expanduser()
        initial_dir = current.parent if current.exists() else self.paths.root
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select replay dataset",
            initialdir=initial_dir,
            filetypes=(
                ("Parquet", "*.parquet *.pq"),
                ("CSV", "*.csv *.txt"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self.file_var.set(selected)

    def _start_replay(self) -> None:
        path = Path(self.file_var.get()).expanduser()
        fmt = (self.format_var.get() or "auto").strip().lower()
        speed = self._current_speed()
        if not path.exists():
            messagebox.showerror("Replay", f"Dataset not found: {path}")
            return
        self._stop_simulator()
        try:
            simulator = ReplaySimulator.from_path(path, fmt=fmt, speed=speed)
        except Exception as exc:
            self.logger.error("Replay start failed", exc_info=exc)
            messagebox.showerror("Replay", f"Unable to start replay: {exc}")
            return
        self._simulator = simulator
        self._queue = Queue()
        simulator.clear_listeners()
        simulator.add_listener(self._on_bar)
        self._processed = 0
        self._total = simulator.total_bars
        self._update_seek_slider(0.0)
        if self._chart:
            self._chart.clear()
        self.output.delete("1.0", tk.END)
        buffering = self.ui_setting(
            "status", "replay", "buffering", default="Preparing replay dataset..."
        )
        self.status_var.set(buffering)
        self.update_section(
            "replay",
            {
                "dataset": str(path),
                "format": fmt,
                "speed": speed,
                "total_bars": self._total,
            },
        )
        simulator.set_speed(speed)
        simulator.start()
        self._schedule_poll()

    def _pause_replay(self) -> None:
        if not self._simulator:
            return
        self._simulator.pause()
        paused = self.ui_setting("status", "replay", "paused", default="Replay paused.")
        self.status_var.set(paused)

    def _resume_replay(self) -> None:
        if not self._simulator:
            self._start_replay()
            return
        self._simulator.resume()
        if not self._simulator.running:
            self._simulator.start()
        playing = self.ui_setting(
            "status", "replay", "playing", default="Streaming simulator feed."
        )
        self.status_var.set(playing)
        self._schedule_poll()

    def _stop_replay(self) -> None:
        self._stop_simulator()

    def _stop_simulator(self) -> None:
        simulator = self._simulator
        if simulator is not None:
            simulator.clear_listeners()
            simulator.stop()
        self._simulator = None
        self._queue = Queue()
        self._cancel_poll()
        self._processed = 0
        self._total = 0
        if self._chart:
            self._chart.clear()
        idle = self.ui_setting(
            "status", "replay", "idle", default="Load a dataset to begin playback."
        )
        self.status_var.set(idle)

    def _on_bar(self, bar: ReplayBar) -> None:
        self._queue.put(bar)

    def _schedule_poll(self) -> None:
        if self._poll_job is None:
            self._poll_job = self.after(self._poll_interval, self._poll_queue)

    def _cancel_poll(self) -> None:
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    def _poll_queue(self) -> None:
        self._poll_job = None
        updated = False
        while True:
            try:
                bar = self._queue.get_nowait()
            except Empty:
                break
            self._process_bar(bar)
            updated = True
        simulator = self._simulator
        if simulator and (simulator.running or not self._queue.empty()):
            self._schedule_poll()
        elif updated and self._total:
            complete = self.ui_setting(
                "status",
                "replay",
                "complete",
                default="Reached end of recording.",
            )
            self.status_var.set(complete)

    def _process_bar(self, bar: ReplayBar) -> None:
        self._processed = max(self._processed, bar.index + 1)
        if self._chart:
            self._chart.push(bar)
        price = self._chart.last_price() if self._chart else None
        playing = self.ui_setting(
            "status", "replay", "playing", default="Streaming simulator feed."
        )
        if price is not None:
            summary = (
                f"{bar.timestamp:%Y-%m-%d %H:%M:%S} · {playing}"
                f" Close {price:.{self._price_decimals}f}"
            )
        else:
            summary = f"{bar.timestamp:%Y-%m-%d %H:%M:%S} · {playing}"
        self.status_var.set(summary)
        ratio = self._processed / self._total if self._total else 0.0
        self._update_seek_slider(ratio * 100.0)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json_dumps(bar.to_dict()))
        self.update_section(
            "replay",
            {
                "last_timestamp": bar.timestamp.isoformat(),
                "last_index": bar.index,
                "progress": self._processed,
            },
        )

    def _set_speed(self, *_: object) -> None:
        speed = self._current_speed()
        if self._simulator:
            try:
                self._simulator.set_speed(speed)
            except ValueError:
                return
        self.update_section("replay", {"speed": speed})

    def _current_speed(self) -> float:
        try:
            value = float(self.speed_var.get())
        except (TypeError, ValueError):
            value = 1.0
        return max(0.1, value)

    def _on_seek_start(self, _: tk.Event) -> None:
        self._user_scrubbing = True

    def _on_seek_release(self, _: tk.Event) -> None:
        self._user_scrubbing = False
        self._apply_seek()

    def _apply_seek(self) -> None:
        if self._suspend_seek or not self._simulator or not self._total:
            return
        target = max(0.0, min(100.0, self.seek_var.get())) / 100.0
        index = int(target * (self._total - 1))
        try:
            self._simulator.seek(index)
        except (TypeError, ValueError) as exc:
            self.logger.error("Replay seek failed", exc_info=exc)
            return
        self._queue = Queue()
        self._processed = index
        buffering = self.ui_setting(
            "status", "replay", "buffering", default="Preparing replay dataset..."
        )
        self.status_var.set(buffering)
        if self._chart:
            self._chart.clear()

    def _update_seek_slider(self, percent: float) -> None:
        if self._user_scrubbing:
            return
        self._suspend_seek = True
        self.seek_var.set(max(0.0, min(100.0, percent)))
        self._suspend_seek = False


class TradeTab(BaseTab):
    """Trade tab that wires the guard engine into the execution workflow."""

    _PANIC_BINDING = "<Control-p>"

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        tv_router: TradingViewRouter | None = None,
    ) -> None:
        super().__init__(master, configs, paths, tv_router=tv_router)
        self.configs.setdefault("trade", {})
        guard_pending = self.ui_setting(
            "status", "guard", "pending", default="Topstep Guard: pending review"
        )
        initial_mode = str(self.configs["trade"].get("mode", "PAPER")).upper()
        self._guard_pending_copy = guard_pending
        self.trading_mode = tk.StringVar(master=self, value=initial_mode)
        self.guard_status = tk.StringVar(
            master=self,
            value=self._compose_guard_caption(guard_pending, initial_mode),
        )
        self.guard_label: ttk.Label | None = None
        self._risk_engine = RiskEngine.from_policy()
        self._guard_report: GuardReport | None = None
        self._panic_bound = False
        self.update_section("trade", {"mode": initial_mode})
        self._bind_panic()
        invoke_tab_builder(self)

    def destroy(self) -> None:  # pragma: no cover - Tk handles lifecycle in UI
        self._unbind_panic()
        super().destroy()

    def _bind_panic(self) -> None:
        if self._panic_bound:
            return
        self.bind_all(self._PANIC_BINDING, self._handle_panic, add="+")
        self._panic_bound = True

    def _unbind_panic(self) -> None:
        if not self._panic_bound:
            return
        self.unbind_all(self._PANIC_BINDING)
        self._panic_bound = False

    def _build(self) -> None:
        self._tv_status: ttk.Label | None = None
        intro = ttk.LabelFrame(
            self,
            text="Step 6 · Execution guard",
            style="Section.TLabelframe",
        )
        intro.pack(fill=tk.X, padx=10, pady=(12, 6))
        ttk.Label(
            intro,
            text=(
                "Final pre-flight checks before you place manual orders. Refresh the guard summary to confirm "
                "position limits, drawdown caps, and cooldown status."
            ),
            wraplength=520,
            justify=tk.LEFT,
            style="Surface.TLabel",
        ).pack(anchor=tk.W)

        status_row = ttk.Frame(intro, style="DashboardBackground.TFrame")
        status_row.pack(fill=tk.X, pady=(10, 6))

        self.guard_label = ttk.Label(
            status_row,
            textvariable=self.guard_status,
            style="GuardBadge.TLabel",
        )
        self.guard_label.pack(anchor=tk.W, side=tk.LEFT, padx=(0, 12))

        mode_frame = ttk.Frame(status_row, style="DashboardBackground.TFrame")
        mode_frame.pack(side=tk.LEFT)
        ttk.Label(
            mode_frame,
            text="Trading mode",
            style="Surface.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        ttk.Radiobutton(
            mode_frame,
            text="Paper (simulated)",
            value="PAPER",
            variable=self.trading_mode,
            command=lambda: self._set_mode("PAPER"),
            style="Input.TRadiobutton",
        ).grid(row=1, column=0, padx=(0, 8), sticky=tk.W)
        ttk.Radiobutton(
            mode_frame,
            text="Live (connected)",
            value="LIVE",
            variable=self.trading_mode,
            command=lambda: self._set_mode("LIVE"),
            style="Input.TRadiobutton",
        ).grid(row=1, column=1, sticky=tk.W)

        controls = ttk.Frame(intro, style="DashboardBackground.TFrame")
        controls.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(
            controls,
            text="Refresh Topstep guard",
            style="Accent.TButton",
            command=lambda: self._refresh_guard(show_modal=True),
        ).pack(side=tk.LEFT)
        ttk.Button(
            controls,
            text="Panic to paper (Ctrl+P)",
            style="Neutral.TButton",
            command=self._panic_to_paper,
        ).pack(side=tk.LEFT, padx=(8, 0))

        if self.tv_router and self.tv_router.is_tab_enabled("trade"):
            ttk.Button(
                controls,
                text="Open TradingView (Ctrl+Shift+T)",
                style="Neutral.TButton",
                command=self._open_trade_tradingview,
            ).pack(side=tk.LEFT, padx=(8, 0))
            self._tv_status = ttk.Label(
                self,
                text="Use TradingView to verify price context before executing.",
                style="MetricCaption.TLabel",
                wraplength=540,
                justify=tk.LEFT,
            )
            self._tv_status.pack(fill=tk.X, padx=12, pady=(0, 6))

        self.output = tk.Text(self, height=12)
        self.style_text_widget(self.output)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 12))
        guard_intro = self.ui_setting(
            "status",
            "guard",
            "intro",
            default="Manual execution only. Awaiting guard refresh...",
        )
        self.output.insert(
            tk.END,
            (
                f"{guard_intro}\n"
                "Use insights from earlier tabs to justify every trade — "
                "and always log rationale."
            ),
        )

    def _panic_to_paper(self) -> None:
        self._apply_panic()

    def _handle_panic(self, _: tk.Event | None) -> str:
        self._apply_panic()
        return "break"

    def _apply_panic(self) -> None:
        mode_before = self.trading_mode.get()
        if mode_before != "PAPER":
            self.trading_mode.set("PAPER")
        panic_at = datetime.now(timezone.utc).isoformat()
        self.update_section("trade", {"mode": "PAPER", "panic_at": panic_at})
        self._update_guard_badge(self._guard_report, "PAPER")
        self.log_event("Panic trigger fired → forced PAPER mode")

    def _set_mode(self, mode: str) -> None:
        current = self.trading_mode.get()
        mode_upper = mode.upper()
        if mode_upper == current:
            return
        if mode_upper == "LIVE":
            confirmation = messagebox.askyesno(
                "Switch to live execution",
                (
                    "You are about to enable LIVE mode. Confirm that risk checks are complete and "
                    "brokers are connected before continuing."
                ),
                icon="warning",
                default=messagebox.NO,
            )
            if not confirmation:
                self.trading_mode.set(current)
                return
        self.trading_mode.set(mode_upper)
        self.update_section("trade", {"mode": mode_upper})
        self._update_guard_badge(self._guard_report, mode_upper)

    def _refresh_guard(self, *, show_modal: bool) -> GuardReport:
        risk_config = self.configs.get("risk", {})
        profile = self._risk_engine.build_profile(risk_config)
        report = self._risk_engine.evaluate(
            profile,
            account_balance=cast(float | None, risk_config.get("account_balance")),
            atr=cast(float | None, risk_config.get("atr")),
            tick_value=cast(float | None, risk_config.get("tick_value")),
            risk_per_trade=cast(float | None, risk_config.get("risk_per_trade")),
        )
        self._guard_report = report
        mode = self.trading_mode.get()
        self._update_guard_badge(report, mode)
        payload: Dict[str, object] = {
            "policy": dict(self._risk_engine.policy_metadata),
            "report": report.to_dict(),
            "mode": mode,
        }
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json_dumps(payload))
        self.update_section("trade", payload)
        if show_modal:
            self._show_guard_dialog(report)
        return report

    def _open_trade_tradingview(self) -> None:
        defaults = self.tradingview_defaults()
        url = self._open_tradingview(symbol=defaults.symbol, interval=defaults.interval)
        if url and self._tv_status is not None:
            self._tv_status.config(
                text=(
                    f"TradingView launched with {defaults.symbol} ({defaults.interval}). "
                    "Confirm guards before placing orders."
                )
            )

    def _show_guard_dialog(self, report: GuardReport) -> None:
        lines = [
            f"Suggested contracts: {report.suggested_contracts}",
            f"Account balance assumption: ${report.account_balance:,.2f}",
        ]
        for rule in report.rules:
            lines.append(f"{rule.title}: {rule.message}")
        guard_message = "Topstep guard assessment completed.\n\n" + "\n".join(lines)
        if report.status == "OK":
            messagebox.showinfo("Topstep Guard", guard_message)
            return
        warning_suffix = self.ui_setting(
            "status",
            "guard",
            "defensive_warning",
            default="DEFENSIVE_MODE active. Stand down and review your journal before trading.",
        )
        messagebox.showwarning("Topstep Guard", f"{guard_message}\n\n{warning_suffix}")

    def _update_guard_badge(self, report: GuardReport | None, mode: str) -> None:
        mode_upper = mode.upper()
        if report is None:
            caption = self._compose_guard_caption(self._guard_pending_copy, mode_upper)
            colour = DARK_PALETTE["accent_alt"]
        else:
            caption = self._compose_guard_caption(
                f"Topstep Guard: {report.status}", mode_upper
            )
            colour = (
                DARK_PALETTE["success"]
                if report.status == "OK"
                else DARK_PALETTE["danger"]
            )
        self.guard_status.set(caption)
        if self.guard_label is not None:
            self.guard_label.configure(foreground=colour)

    def _compose_guard_caption(self, guard_text: str, mode: str) -> str:
        label = guard_text.strip()
        mode_upper = mode.upper()
        if "mode" in label.lower():
            return label
        return f"{label} · Mode {mode_upper}"


__all__ = [
    "DashboardTab",
    "LoginTab",
    "ResearchTab",
    "TrainTab",
    "BacktestTab",
    "ReplayTab",
    "TradingViewTab",
    "TradeTab",
]
