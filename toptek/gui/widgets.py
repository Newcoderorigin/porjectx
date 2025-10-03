"""Elite HUD widgets and tab implementations for the Toptek GUI."""

from __future__ import annotations

import os
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import ttkbootstrap as tb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap.constants import BOTH, END, LEFT, RIGHT, TOP, X, Y

from core import backtest, features, model, risk, utils
from core.data import sample_dataframe


MONO_FONT = ("Consolas", 12)
ACCENT_A = "#00E0FF"
ACCENT_B = "#39FF14"
BACKGROUND = "#10141b"


class HUDCard(tb.Frame):
    """Glassy HUD tile displaying a primary numeric value."""

    def __init__(self, master: tk.Misc, title: str, *, icon: str | None = None) -> None:
        super().__init__(master, padding=12, bootstyle="secondary")
        self.configure(borderwidth=1)
        self.title_var = tk.StringVar(value=title)
        self.value_var = tk.StringVar(value="—")
        self.subvalue_var = tk.StringVar(value="")
        header = tb.Frame(self)
        header.pack(fill=X)
        if icon:
            tb.Label(header, text=icon, font=("Segoe UI Symbol", 14), bootstyle="inverse-secondary").pack(side=LEFT)
        tb.Label(
            header,
            textvariable=self.title_var,
            font=("Eurostile", 11),
            bootstyle="inverse-secondary",
        ).pack(side=LEFT, padx=(4, 0))
        tb.Label(
            self,
            textvariable=self.value_var,
            font=("Consolas", 20, "bold"),
            bootstyle="info",
        ).pack(anchor="w", pady=(8, 0))
        tb.Label(
            self,
            textvariable=self.subvalue_var,
            font=("Consolas", 10),
            bootstyle="secondary",
        ).pack(anchor="w")

    def update(self, value: str, *, subvalue: str = "") -> None:
        self.value_var.set(value)
        self.subvalue_var.set(subvalue)


class MetricRow(tb.Frame):
    """Row showing a left-aligned metric name with right-aligned value."""

    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(master)
        tb.Label(self, text=label, bootstyle="secondary", font=("Eurostile", 10)).pack(side=LEFT)
        self.value_var = tk.StringVar(value="—")
        tb.Label(self, textvariable=self.value_var, bootstyle="inverse-secondary", font=MONO_FONT).pack(side=RIGHT)

    def set(self, value: str) -> None:
        self.value_var.set(value)


class HelpIcon(tb.Label):
    """Clickable help icon that shows a popover with instructions."""

    def __init__(self, master: tk.Misc, message: str) -> None:
        super().__init__(master, text="?", cursor="question_arrow", bootstyle="info")
        self.message = message
        self._popover: tb.Toplevel | None = None
        self.bind("<Button-1>", self._show)

    def _show(self, *_: Any) -> None:
        if self._popover is not None:
            self._popover.destroy()
        self._popover = tb.Toplevel(self)
        self._popover.overrideredirect(True)
        self._popover.attributes("-topmost", True)
        x = self.winfo_rootx() + 20
        y = self.winfo_rooty() + 20
        self._popover.geometry(f"+{x}+{y}")
        frame = tb.Frame(self._popover, padding=8, bootstyle="info")
        frame.pack(fill=BOTH, expand=True)
        tb.Label(frame, text=self.message, wraplength=240, justify=LEFT, bootstyle="inverse-info").pack()
        frame.after(4000, self._popover.destroy)


class GuideDrawer(tb.Frame):
    """Collapsible checklist panel displayed on the right edge."""

    def __init__(self, master: tk.Misc, *, width: int = 240) -> None:
        super().__init__(master, padding=12, bootstyle="secondary")
        self.master = master
        self.width = width
        self.visible = True
        self._steps: List[str] = []
        header = tb.Frame(self)
        header.pack(fill=X)
        tb.Label(header, text="GUIDE", bootstyle="inverse-secondary", font=("Eurostile", 11)).pack(side=LEFT)
        self.toggle_btn = tb.Button(header, text="⮜", width=3, bootstyle="secondary", command=self.toggle)
        self.toggle_btn.pack(side=RIGHT)
        self.list_frame = tb.Frame(self, padding=(0, 8))
        self.list_frame.pack(fill=BOTH, expand=True)

    def toggle(self) -> None:
        self.visible = not self.visible
        if self.visible:
            self.toggle_btn.configure(text="⮜")
            self.list_frame.pack(fill=BOTH, expand=True)
            self.configure(width=self.width)
        else:
            self.toggle_btn.configure(text="⮞")
            self.list_frame.forget()
            self.configure(width=24)

    def set_steps(self, steps: Sequence[str]) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        self._steps = list(steps)
        for idx, step in enumerate(self._steps, start=1):
            tb.Label(
                self.list_frame,
                text=f"{idx}. {step}",
                wraplength=self.width - 24,
                justify=LEFT,
                bootstyle="secondary",
                font=("Eurostile", 10),
            ).pack(anchor="w", pady=4)


class CoachMarks:
    """Orchestrates sequential coach marks using a :class:`GuideOverlay`."""

    def __init__(self, master: tk.Misc, overlay: Any) -> None:
        self.master = master
        self.overlay = overlay
        self.steps: List[tuple[str, tk.Widget]] = []
        self._index = -1

    def configure(self, steps: Sequence[tuple[str, tk.Widget]]) -> None:
        self.steps = list(steps)
        self._index = -1

    def start(self) -> None:
        self._index = -1
        self._advance()

    def _advance(self) -> None:
        self._index += 1
        if self._index >= len(self.steps):
            self.overlay.hide()
            return
        text, widget = self.steps[self._index]
        if widget is None:
            self._advance()
            return
        self.overlay.show(widget, f"{text}\n(click to continue)", on_dismiss=self._advance)


class EliteTab(tb.Frame):
    """Base class for all HUD tabs providing shared helpers."""

    guide_steps: Sequence[str] = ()

    def __init__(
        self,
        master: tb.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
        *,
        bus: Any,
        toast: Any,
        overlay: Any,
        guide_drawer: GuideDrawer,
        status: Any,
    ) -> None:
        super().__init__(master, padding=16)
        self.configs = configs
        self.paths = paths
        self.bus = bus
        self.toast = toast
        self.overlay = overlay
        self.guide_drawer = guide_drawer
        self.status = status
        self.coach_targets: Dict[str, tk.Widget] = {}

    def handle_command(self, command: str) -> None:  # noqa: D401 - doc inherited
        """React to command-rail shortcuts. Default focuses the tab."""
        self.focus_set()

    def show_details_popup(self, title: str, message: str) -> None:
        popup = tb.Toplevel(self)
        popup.title(title)
        popup.geometry("520x340")
        text = tk.Text(popup, wrap="word", font=("Consolas", 10))
        text.insert("1.0", message)
        text.configure(state="disabled")
        text.pack(fill=BOTH, expand=True)
        tb.Button(popup, text="Close", command=popup.destroy).pack(pady=8)


class LoginTab(EliteTab):
    """Login tab that manages environment credentials and auth."""

    guide_steps = ["Paste keys", "Save .env", "Login", "Select account"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._build()

    def _build(self) -> None:
        frame = tb.Labelframe(self, text="ProjectX Credentials", padding=16)
        frame.pack(fill=X, pady=(0, 16))
        fields = [
            "PX_BASE_URL",
            "PX_MARKET_HUB",
            "PX_USER_HUB",
            "PX_USERNAME",
            "PX_API_KEY",
        ]
        self.vars = {}
        for row, field in enumerate(fields):
            tb.Label(frame, text=field, bootstyle="secondary").grid(row=row, column=0, sticky="w", pady=6)
            var = tk.StringVar(value=os.environ.get(field, ""))
            entry = tb.Entry(frame, textvariable=var, width=48)
            entry.grid(row=row, column=1, sticky="ew", pady=6)
            frame.columnconfigure(1, weight=1)
            self.vars[field] = var
            if field == "PX_API_KEY":
                self.coach_targets["api_key"] = entry
        btns = tb.Frame(frame)
        btns.grid(row=len(fields), column=0, columnspan=2, pady=(12, 0))
        save_btn = tb.Button(btns, text="Save .env", bootstyle="success", command=self._save_env)
        save_btn.pack(side=LEFT, padx=(0, 6))
        self.coach_targets["save"] = save_btn
        self.login_btn = tb.Button(btns, text="Login", bootstyle="primary", command=self._login)
        self.login_btn.pack(side=LEFT)
        self.coach_targets["login"] = self.login_btn

        self.account_var = tk.StringVar(value="")
        account_frame = tb.Labelframe(self, text="Account", padding=16)
        account_frame.pack(fill=X)
        tb.Label(account_frame, text="Accounts", bootstyle="secondary").pack(side=LEFT)
        self.account_combo = tb.Combobox(account_frame, textvariable=self.account_var, width=32, state="readonly")
        self.account_combo.pack(side=LEFT, padx=8)
        self.account_combo.bind("<<ComboboxSelected>>", self._on_account_selected)

    def _save_env(self) -> None:
        env_path = self.paths.root / ".env"
        with env_path.open("w", encoding="utf-8") as handle:
            for key, var in self.vars.items():
                handle.write(f"{key}={var.get()}\n")
        self.toast.success(f"Credentials saved to {env_path}")

    def _login(self) -> None:
        self.login_btn.configure(state="disabled", text="Logging in…")
        self.after(600, self._complete_login)

    def _complete_login(self) -> None:
        self.login_btn.configure(state="normal", text="Login")
        accounts = ["SIM-001", "SIM-002"]
        self.account_combo.configure(values=accounts)
        if accounts:
            self.account_combo.current(0)
            self.status.account.set(accounts[0])
        self.toast.success("Connected to ProjectX demo gateway")
        self.bus.send_command("status:update", account=self.status.account.get())

    def _on_account_selected(self, *_: Any) -> None:
        self.status.account.set(self.account_var.get())
        self.bus.send_command("status:update", account=self.account_var.get())


class ResearchTab(EliteTab):
    """Research tab for contract discovery and bar previews."""

    guide_steps = ["Search symbol", "Select result", "Preview bars"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._build()

    def _build(self) -> None:
        top = tb.Frame(self)
        top.pack(fill=X)
        tb.Label(top, text="Symbol", bootstyle="secondary").pack(side=LEFT)
        self.search_var = tk.StringVar()
        search_entry = tb.Entry(top, textvariable=self.search_var, width=18)
        search_entry.pack(side=LEFT, padx=6)
        self.coach_targets["search"] = search_entry
        tb.Button(top, text="Search", bootstyle="primary", command=self._search).pack(side=LEFT)
        self.timeframe_var = tk.StringVar(value="5m")
        tb.Label(top, text="Timeframe", bootstyle="secondary").pack(side=LEFT, padx=(16, 4))
        self.timeframe_combo = tb.Combobox(top, values=["1m", "5m", "15m", "1h", "1d"], textvariable=self.timeframe_var, width=6, state="readonly")
        self.timeframe_combo.pack(side=LEFT)
        HelpIcon(top, "Search contracts via ProjectX or offline futures list.").pack(side=LEFT, padx=(6, 0))

        columns = ("symbol", "name", "exchange", "asset")
        self.tree = tb.Treeview(self, columns=columns, show="headings", height=6)
        for col in columns:
            self.tree.heading(col, text=col.upper())
            self.tree.column(col, stretch=True, width=130)
        self.tree.pack(fill=X, pady=12)
        self.tree.bind("<Double-1>", self._on_select)

        action_bar = tb.Frame(self)
        action_bar.pack(fill=X)
        preview_btn = tb.Button(action_bar, text="Preview Bars", bootstyle="info", command=self._preview)
        preview_btn.pack(side=LEFT)
        self.coach_targets["preview"] = preview_btn

        dash = tb.Frame(self)
        dash.pack(fill=X, pady=(16, 8))
        self.card_last = HUDCard(dash, "Last Close", icon="💡")
        self.card_last.pack(side=LEFT, padx=6)
        self.card_range = HUDCard(dash, "Day Range", icon="📈")
        self.card_range.pack(side=LEFT, padx=6)
        self.card_atr = HUDCard(dash, "ATR20", icon="📐")
        self.card_atr.pack(side=LEFT, padx=6)
        self.card_session = HUDCard(dash, "Session", icon="⏱")
        self.card_session.pack(side=LEFT, padx=6)

        chart_frame = tb.Frame(self)
        chart_frame.pack(fill=BOTH, expand=True)
        self.figure = Figure(figsize=(6, 3), facecolor=BACKGROUND)
        self.ax_price = self.figure.add_subplot(211)
        self.ax_volume = self.figure.add_subplot(212)
        self.ax_price.set_facecolor(BACKGROUND)
        self.ax_volume.set_facecolor(BACKGROUND)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self.guide_drawer.set_steps(self.guide_steps)

    def _search(self) -> None:
        term = self.search_var.get().strip().upper()
        self.tree.delete(*self.tree.get_children())
        if not term:
            return
        sample = [
            {"symbol": "ESZ5", "name": "E-Mini S&P 500", "exchange": "CME", "asset": "Index"},
            {"symbol": "NQZ5", "name": "E-Mini Nasdaq-100", "exchange": "CME", "asset": "Index"},
        ]
        matches = [row for row in sample if term in row["symbol"] or term in row["name"].upper()]
        for row in matches:
            self.tree.insert("", END, values=(row["symbol"], row["name"], row["exchange"], row["asset"]))
        if matches:
            self.status.symbol.set(matches[0]["symbol"])
            self.bus.send_command("status:update", symbol=matches[0]["symbol"])

    def _on_select(self, *_: Any) -> None:
        selection = self.tree.item(self.tree.selection() or "")
        if selection and selection.get("values"):
            symbol = selection["values"][0]
            self.status.symbol.set(symbol)
            self.bus.send_command("status:update", symbol=symbol)

    def _preview(self) -> None:
        df = sample_dataframe(180)
        df = df.tail(100)
        close = df["close"]
        volume = df["volume"]
        self.ax_price.clear()
        self.ax_volume.clear()
        self.ax_price.plot(close.index, close.values, color=ACCENT_A)
        self.ax_price.set_title("Mini candles")
        self.ax_volume.bar(volume.index, volume.values, color=ACCENT_B)
        self.ax_volume.set_title("Volume")
        self.figure.autofmt_xdate()
        self.canvas.draw_idle()
        self.card_last.update(f"{close.iloc[-1]:.2f}")
        self.card_range.update(f"{df['high'].iloc[-1]-df['low'].iloc[-1]:.2f}")
        atr = features.compute_features(df)["atr_14"].iloc[-1]
        self.card_atr.update(f"{atr:.2f}")
        minutes_left = max(0, 60 - datetime.utcnow().minute)
        self.card_session.update(f"{minutes_left}m left")
        self.toast.info("Loaded preview bars")
        self.bus.send_command("status:update", timeframe=self.timeframe_var.get())


class TrainTab(EliteTab):
    """Training tab for running local models and calibration."""

    guide_steps = ["Pick features", "Train", "Calibrate", "Save"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_result: model.TrainResult | None = None
        self.last_training_data: tuple[pd.DataFrame, pd.Series] | None = None
        self.last_calibration: model.CalibrationResult | None = None
        self._build()

    def _build(self) -> None:
        config_features = self.configs.get("features", {}).get("feature_set", [])
        self.feature_set_var = tk.StringVar(value=", ".join(config_features) or "default")
        control = tb.Labelframe(self, text="Training Setup", padding=16)
        control.pack(fill=X)

        tb.Label(control, text="Feature set", bootstyle="secondary").grid(row=0, column=0, sticky="w")
        self.features_entry = tb.Entry(control, textvariable=self.feature_set_var, width=60)
        self.features_entry.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        control.columnconfigure(1, weight=1)

        tb.Label(control, text="Horizon", bootstyle="secondary").grid(row=1, column=0, sticky="w")
        self.horizon_var = tk.IntVar(value=5)
        tb.Spinbox(control, from_=1, to=60, textvariable=self.horizon_var, width=6).grid(row=1, column=1, sticky="w", pady=4)

        tb.Label(control, text="Model type", bootstyle="secondary").grid(row=2, column=0, sticky="w")
        self.model_type_var = tk.StringVar(value="logistic")
        model_combo = tb.Combobox(
            control,
            textvariable=self.model_type_var,
            values=("logistic", "gbm"),
            state="readonly",
            width=12,
        )
        model_combo.grid(row=2, column=1, sticky="w", pady=4)
        HelpIcon(control, "Choose between logistic regression or gradient boosting.").grid(row=2, column=2, padx=6)

        tb.Label(control, text="Probability threshold", bootstyle="secondary").grid(row=3, column=0, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.65)
        threshold_entry = tb.Entry(control, textvariable=self.threshold_var, width=8)
        threshold_entry.grid(row=3, column=1, sticky="w", pady=4)
        HelpIcon(control, "Signals trigger when probability exceeds this value.").grid(row=3, column=2, padx=6)
        self.coach_targets["train"] = None  # placeholder updated later

        buttons = tb.Frame(control)
        buttons.grid(row=4, column=0, columnspan=3, pady=(12, 0), sticky="w")
        self.train_btn = tb.Button(buttons, text="Run Train", bootstyle="primary", command=self._train_model)
        self.train_btn.pack(side=LEFT)
        self.coach_targets["train"] = self.train_btn
        tb.Button(buttons, text="Calibrate", bootstyle="info", command=self._calibrate).pack(side=LEFT, padx=6)
        tb.Button(buttons, text="Save Model", bootstyle="success", command=self._save_model).pack(side=LEFT)
        HelpIcon(buttons, "Calibration aligns predicted probabilities with outcomes.").pack(side=LEFT, padx=6)

        self.metrics_box = tb.Labelframe(self, text="Metrics", padding=12)
        self.metrics_box.pack(fill=BOTH, expand=True, pady=(16, 0))
        self.metric_accuracy = MetricRow(self.metrics_box, "Accuracy")
        self.metric_accuracy.pack(fill=X)
        self.metric_auc = MetricRow(self.metrics_box, "ROC-AUC")
        self.metric_auc.pack(fill=X)
        self.metric_samples = MetricRow(self.metrics_box, "Samples/Features")
        self.metric_samples.pack(fill=X)
        self.metric_brier = MetricRow(self.metrics_box, "Brier score")
        self.metric_brier.pack(fill=X)

    def _train_model(self) -> None:
        try:
            df = sample_dataframe(400)
            feature_frame = features.compute_features(df)
            forward_return = df["close"].pct_change().shift(-1)
            y_series = (forward_return.loc[feature_frame.index] > 0).astype(int)
            aligned = pd.concat({"X": feature_frame, "y": y_series}, axis=1).dropna()
            X = aligned["X"].copy()
            y = aligned["y"].copy()
            self.last_training_data = (X, y)
            self.last_calibration = None
            result = model.train_classifier(
                X,
                y,
                models_dir=self.paths.models,
                threshold=self.threshold_var.get(),
                model_type=self.model_type_var.get(),
            )
        except Exception as exc:  # pragma: no cover - GUI feedback path
            import traceback

            self.toast.error("Training failed (see details)")
            self.show_details_popup("Training Error", traceback.format_exc())
            return
        self.last_result = result
        self.metric_accuracy.set(f"{result.metrics['accuracy']:.3f}")
        self.metric_auc.set(f"{result.metrics['roc_auc']:.3f}")
        self.metric_samples.set(
            f"{result.metrics['n_samples']} samples / {result.metrics['n_features']} features"
        )
        self.metric_brier.set(f"{result.metrics['brier_score']:.3f}")
        bins = "\n".join(
            f"Pred {pred:.2f} → Obs {obs:.2f}" for pred, obs in result.calibration_curve
        )
        if bins:
            self.show_details_popup("Calibration (hold-out)", bins)
        self.toast.success("Model trained successfully")

    def _calibrate(self) -> None:
        if not self.last_result or not self.last_training_data:
            self.toast.warning("Train a model first")
            return
        X, y = self.last_training_data
        try:
            calib = model.calibrate_classifier(
                X,
                y,
                base_model_path=self.last_result.model_path,
                models_dir=self.paths.models,
            )
        except Exception:
            import traceback

            self.toast.error("Calibration failed (see details)")
            self.show_details_popup("Calibration Error", traceback.format_exc())
            return
        self.last_calibration = calib
        self.metric_brier.set(f"{calib.metrics['brier_score']:.3f}")
        details = "\n".join(
            f"Pred {pred:.2f} → Obs {obs:.2f}" for pred, obs in calib.calibration_curve
        )
        self.show_details_popup("Calibration curve", details)
        self.toast.success(f"Calibrated model saved to {calib.model_path.name}")

    def _save_model(self) -> None:
        if not self.last_result:
            self.toast.warning("Train a model first")
            return
        self.toast.success(f"Model saved to {self.last_result.model_path}")


class BacktestTab(EliteTab):
    """Backtesting tab with metrics and equity curve display."""

    guide_steps = ["Load model", "Run", "Review metrics"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.equity_fig: Figure | None = None
        self._build()

    def _build(self) -> None:
        control = tb.Labelframe(self, text="Backtest", padding=16)
        control.pack(fill=X)
        tb.Label(control, text="Model", bootstyle="secondary").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_combo = tb.Combobox(control, textvariable=self.model_var, width=40, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        tb.Button(control, text="Refresh", bootstyle="secondary", command=self._refresh_models).grid(row=0, column=2, padx=6)
        control.columnconfigure(1, weight=1)

        tb.Label(control, text="Start", bootstyle="secondary").grid(row=1, column=0, sticky="w")
        self.start_var = tk.StringVar(value=(datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"))
        tb.Entry(control, textvariable=self.start_var, width=12).grid(row=1, column=1, sticky="w", pady=4)

        tb.Label(control, text="End", bootstyle="secondary").grid(row=1, column=2, sticky="w")
        self.end_var = tk.StringVar(value=datetime.utcnow().strftime("%Y-%m-%d"))
        tb.Entry(control, textvariable=self.end_var, width=12).grid(row=1, column=3, sticky="w", pady=4)

        tb.Label(control, text="Fees", bootstyle="secondary").grid(row=2, column=0, sticky="w")
        self.fee_var = tk.DoubleVar(value=1.25)
        tb.Entry(control, textvariable=self.fee_var, width=8).grid(row=2, column=1, sticky="w")

        self.run_btn = tb.Button(control, text="Run Backtest", bootstyle="primary", command=self._run_backtest)
        self.run_btn.grid(row=3, column=0, columnspan=4, pady=(12, 0))
        self.coach_targets["run"] = self.run_btn

        metrics_frame = tb.Frame(self)
        metrics_frame.pack(fill=X, pady=(16, 8))
        self.hit_card = HUDCard(metrics_frame, "Hit-Rate", icon="🎯")
        self.hit_card.pack(side=LEFT, padx=6)
        self.sharpe_card = HUDCard(metrics_frame, "Sharpe", icon="⚡")
        self.sharpe_card.pack(side=LEFT, padx=6)
        self.dd_card = HUDCard(metrics_frame, "Max DD", icon="📉")
        self.dd_card.pack(side=LEFT, padx=6)
        self.exp_card = HUDCard(metrics_frame, "Expectancy", icon="Σ")
        self.exp_card.pack(side=LEFT, padx=6)

        chart_frame = tb.Frame(self)
        chart_frame.pack(fill=BOTH, expand=True)
        self.equity_fig = Figure(figsize=(6, 3), facecolor=BACKGROUND)
        self.ax_equity = self.equity_fig.add_subplot(111)
        self.ax_equity.set_facecolor(BACKGROUND)
        self.equity_canvas = FigureCanvasTkAgg(self.equity_fig, master=chart_frame)
        self.equity_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self._refresh_models()
        self.guide_drawer.set_steps(self.guide_steps)

    def _refresh_models(self) -> None:
        paths = list(Path(self.paths.models).glob("*.pkl"))
        labels = [p.name for p in paths]
        self.model_combo.configure(values=labels)
        if labels:
            self.model_combo.current(0)

    def _run_backtest(self) -> None:
        if not self.model_var.get():
            self.toast.warning("Select a model first")
            return
        df = sample_dataframe(400)
        returns = np.log(df["close"]).diff().fillna(0).to_numpy()
        signals = (returns > 0).astype(int)
        result = backtest.run_backtest(returns, signals, fee_per_trade=self.fee_var.get())
        self.hit_card.update(f"{result.hit_rate:.2%}")
        self.sharpe_card.update(f"{result.sharpe:.2f}")
        self.dd_card.update(f"{result.max_drawdown:.2f}")
        self.exp_card.update(f"{result.expectancy:.4f}")
        self.ax_equity.clear()
        self.ax_equity.plot(result.equity_curve, color=ACCENT_A)
        self.ax_equity.set_title("Equity Curve")
        self.equity_canvas.draw_idle()
        self.toast.info("Backtest complete")


class TradeTab(EliteTab):
    """Trade tab for monitoring orders, positions, and toggling paper/live."""

    guide_steps = ["Select model", "Start Paper", "Monitor"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mode_var = tk.StringVar(value="Paper")
        self.mode_state = tk.BooleanVar(value=False)
        self._build()

    def _build(self) -> None:
        header = tb.Frame(self)
        header.pack(fill=X)
        tb.Label(header, text="Mode", bootstyle="secondary").pack(side=LEFT)
        self.mode_toggle = tb.Checkbutton(
            header,
            textvariable=self.mode_var,
            variable=self.mode_state,
            bootstyle="success-toolbutton",
            command=self._toggle_mode,
        )
        self.mode_toggle.pack(side=LEFT, padx=6)
        self.coach_targets["paper"] = self.mode_toggle

        self.signal_card = HUDCard(self, "Last Signal", icon="📡")
        self.signal_card.pack(side=TOP, anchor="w", pady=(12, 6))

        order_frame = tb.Labelframe(self, text="Place Order", padding=12)
        order_frame.pack(fill=X)
        self.side_var = tk.StringVar(value="BUY")
        tb.Combobox(order_frame, values=["BUY", "SELL"], textvariable=self.side_var, width=6, state="readonly").grid(row=0, column=0, padx=4, pady=4)
        self.qty_var = tk.IntVar(value=1)
        tb.Spinbox(order_frame, from_=1, to=10, textvariable=self.qty_var, width=5).grid(row=0, column=1, padx=4, pady=4)
        self.type_var = tk.StringVar(value="LIMIT")
        tb.Combobox(order_frame, values=["MARKET", "LIMIT", "STOP"], textvariable=self.type_var, width=8, state="readonly").grid(row=0, column=2, padx=4, pady=4)
        self.price_var = tk.DoubleVar(value=0.0)
        tb.Entry(order_frame, textvariable=self.price_var, width=10).grid(row=0, column=3, padx=4, pady=4)
        tb.Button(order_frame, text="Submit", bootstyle="primary", command=self._submit_order).grid(row=0, column=4, padx=4)

        tables = tb.Frame(self)
        tables.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.orders_table = self._build_table(tables, "Open Orders")
        self.positions_table = self._build_table(tables, "Positions")
        self.trades_table = self._build_table(tables, "Recent Trades")

        self.guide_drawer.set_steps(self.guide_steps)

    def _build_table(self, master: tk.Misc, title: str) -> tb.Treeview:
        frame = tb.Labelframe(master, text=title, padding=8)
        frame.pack(fill=BOTH, expand=True, side=LEFT, padx=4)
        tree = tb.Treeview(frame, columns=("col1", "col2", "col3"), show="headings", height=6)
        for col in ("col1", "col2", "col3"):
            tree.heading(col, text=col.upper())
            tree.column(col, width=90, stretch=True)
        tree.pack(fill=BOTH, expand=True)
        return tree

    def _toggle_mode(self) -> None:
        mode = "Live" if self.mode_state.get() else "Paper"
        self.mode_var.set(mode)
        self.status.mode.set(mode)
        self.bus.send_command("status:update", mode=mode)
        self.toast.info(f"Switched to {mode} mode")

    def _submit_order(self) -> None:
        size = self.qty_var.get()
        risk_profile = risk.RiskProfile(
            max_position_size=self.configs["risk"].get("max_position_size", 1),
            max_daily_loss=self.configs["risk"].get("max_daily_loss", 1000),
            restricted_hours=self.configs["risk"].get("restricted_trading_hours", []),
            atr_multiplier_stop=self.configs["risk"].get("atr_multiplier_stop", 2.0),
            cooldown_losses=self.configs["risk"].get("cooldown_losses", 2),
            cooldown_minutes=self.configs["risk"].get("cooldown_minutes", 30),
        )
        if size > risk_profile.max_position_size:
            self.toast.error("Order breaches Topstep max contracts")
            return
        self.toast.success("Order submitted (simulated)")


__all__ = [
    "HUDCard",
    "MetricRow",
    "HelpIcon",
    "GuideDrawer",
    "CoachMarks",
    "EliteTab",
    "LoginTab",
    "ResearchTab",
    "TrainTab",
    "BacktestTab",
    "TradeTab",
]
