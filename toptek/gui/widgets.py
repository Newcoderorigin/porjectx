"""Tkinter tab implementations for the Toptek GUI."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, TypeVar, cast

import numpy as np

from core import backtest, features, model, risk, utils
from toptek.features import build_features
from core.data import sample_dataframe
from core.utils import json_dumps

from . import DARK_PALETTE, TEXT_WIDGET_DEFAULTS


T = TypeVar("T")


class BaseTab(ttk.Frame):
    """Base class providing convenience utilities for tabs."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
    ) -> None:
        super().__init__(master, style="DashboardBackground.TFrame")
        self.configs = configs
        self._ui_config = configs.get("ui", {})
        self.paths = paths
        self.logger = utils.build_logger(self.__class__.__name__)

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


class DashboardTab(BaseTab):
    """Mission control overview with themed dashboard cards."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
    ) -> None:
        self.workflow_value = tk.StringVar()
        self.workflow_caption = tk.StringVar()
        self.credentials_value = tk.StringVar()
        self.credentials_caption = tk.StringVar()
        self.training_value = tk.StringVar()
        self.training_caption = tk.StringVar()
        self.chart_summary = tk.StringVar()
        super().__init__(master, configs, paths)
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
    ) -> None:
        super().__init__(master, configs, paths)
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
    ) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
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
    ) -> None:
        super().__init__(master, configs, paths)
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
    ) -> None:
        super().__init__(master, configs, paths)
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


class TradeTab(BaseTab):
    """Trade tab placeholder for polling order/position data."""

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: utils.AppPaths,
    ) -> None:
        super().__init__(master, configs, paths)
        guard_pending = self.ui_setting(
            "status", "guard", "pending", default="Topstep Guard: pending review"
        )
        self.guard_status = tk.StringVar(master=self, value=guard_pending)
        self.guard_label: ttk.Label | None = None
        self._build()

    def _build(self) -> None:
        intro = ttk.LabelFrame(
            self,
            text="Step 5 · Execution guard",
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

        self.guard_label = ttk.Label(
            intro, textvariable=self.guard_status, style="SurfaceStatus.TLabel"
        )
        self.guard_label.pack(anchor=tk.W, pady=(8, 0))

        ttk.Button(
            self,
            text="Refresh Topstep guard",
            style="Accent.TButton",
            command=self._show_risk,
        ).pack(pady=(6, 0))
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
            f"{guard_intro}\nUse insights from earlier tabs to justify every trade — and always log rationale.",
        )

    def _show_risk(self) -> None:
        """Refresh the Topstep guard summary and surface a contextual dialog.

        The check recalculates a sample position size using the configured
        Topstep profile, updates the guard label colouring, and displays either
        an informational or warning dialog depending on whether the guard
        remains in ``OK`` or moves into ``DEFENSIVE_MODE``.
        """
        profile = risk.RiskProfile(
            max_position_size=self.configs["risk"].get("max_position_size", 1),
            max_daily_loss=self.configs["risk"].get("max_daily_loss", 1000),
            restricted_hours=self.configs["risk"].get("restricted_trading_hours", []),
            atr_multiplier_stop=self.configs["risk"].get("atr_multiplier_stop", 2.0),
            cooldown_losses=self.configs["risk"].get("cooldown_losses", 2),
            cooldown_minutes=self.configs["risk"].get("cooldown_minutes", 30),
        )
        sample_size = risk.position_size(
            50000, profile, atr=3.5, tick_value=12.5, risk_per_trade=0.01
        )
        guard = "OK" if sample_size > 0 else "DEFENSIVE_MODE"
        self.guard_status.set(f"Topstep Guard: {guard}")
        if self.guard_label is not None:
            colour = (
                DARK_PALETTE["success"] if guard == "OK" else DARK_PALETTE["danger"]
            )
            self.guard_label.configure(foreground=colour)
        payload = {
            "profile": profile.__dict__,
            "suggested_contracts": sample_size,
            "account_balance_assumed": 50000,
            "cooldown_policy": {
                "losses": profile.cooldown_losses,
                "minutes": profile.cooldown_minutes,
            },
            "topstep_guard": guard,
            "next_steps": "If guard shows DEFENSIVE_MODE, stand down and review journal before trading.",
        }
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json_dumps(payload))
        self.update_section("trade", payload)

        guard_message = (
            "Topstep guard assessment completed.\n\n"
            f"Suggested contracts: {sample_size}.\n"
            f"Daily loss cap: ${profile.max_daily_loss}.\n"
            "Cooldown policy: "
            f"{profile.cooldown_losses} losses → wait {profile.cooldown_minutes} minutes."
        )

        if guard == "OK":
            messagebox.showinfo("Topstep Guard", guard_message)
        else:
            warning_suffix = self.ui_setting(
                "status",
                "guard",
                "defensive_warning",
                default="DEFENSIVE_MODE active. Stand down and review your journal before trading.",
            )
            warning_message = f"{guard_message}\n\n{warning_suffix}"
            messagebox.showwarning("Topstep Guard", warning_message)


__all__ = [
    "DashboardTab",
    "LoginTab",
    "ResearchTab",
    "TrainTab",
    "BacktestTab",
    "TradeTab",
]
