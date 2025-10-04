"""Tkinter tab implementations for the Toptek GUI."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict

import numpy as np

from core import backtest, features, model, risk, utils
from core.data import sample_dataframe
from core.utils import json_dumps


class BaseTab(ttk.Frame):
    """Base class providing convenience utilities for tabs."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master)
        self.configs = configs
        self.paths = paths
        self.logger = utils.build_logger(self.__class__.__name__)

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


class LoginTab(BaseTab):
    """Login tab that manages .env configuration."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        intro = ttk.LabelFrame(self, text="Step 1 · Secure your environment", padding=12)
        intro.pack(fill=tk.X, padx=10, pady=(12, 6))
        ttk.Label(
            intro,
            text=(
                "Paste sandbox credentials or API keys. Nothing leaves your machine. "
                "Use the guided Save + Verify buttons to confirm readiness before moving on."
            ),
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        form = ttk.Frame(self)
        form.pack(padx=10, pady=6, fill=tk.X)
        self.vars = {
            "PX_BASE_URL": tk.StringVar(value=self._env_value("PX_BASE_URL")),
            "PX_MARKET_HUB": tk.StringVar(value=self._env_value("PX_MARKET_HUB")),
            "PX_USER_HUB": tk.StringVar(value=self._env_value("PX_USER_HUB")),
            "PX_USERNAME": tk.StringVar(value=self._env_value("PX_USERNAME")),
            "PX_API_KEY": tk.StringVar(value=self._env_value("PX_API_KEY")),
        }
        for row, (label, var) in enumerate(self.vars.items()):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky=tk.W, padx=4, pady=4)
            ttk.Entry(form, textvariable=var, width=60).grid(row=row, column=1, padx=4, pady=4)
        actions = ttk.Frame(self)
        actions.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Button(actions, text="Save .env", command=self._save_env).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Verify entries", command=self._verify_env).pack(side=tk.LEFT)
        self.status = ttk.Label(actions, text="Awaiting verification", foreground="#1d4ed8")
        self.status.pack(side=tk.LEFT, padx=12)

    def _env_value(self, key: str) -> str:
        return os.environ.get(key, "")

    def _save_env(self) -> None:
        env_path = self.paths.root / ".env"
        with env_path.open("w", encoding="utf-8") as handle:
            for key, var in self.vars.items():
                handle.write(f"{key}={var.get()}\n")
        messagebox.showinfo("Settings", f"Saved credentials to {env_path}")
        self.status.config(text="Saved. Run verification to confirm access.", foreground="#166534")

    def _verify_env(self) -> None:
        missing = [key for key, var in self.vars.items() if not var.get().strip()]
        if missing:
            details = ", ".join(missing)
            self.status.config(text=f"Missing: {details}", foreground="#b91c1c")
            messagebox.showwarning("Verification", f"Provide values for: {details}")
            return
        self.status.config(text="All keys present. Proceed to Research ▶", foreground="#15803d")
        messagebox.showinfo("Verification", "Environment entries look complete. Continue to the next tab.")


class ResearchTab(BaseTab):
    """Research tab to preview sample data."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        controls = ttk.LabelFrame(self, text="Step 2 · Research console", padding=12)
        controls.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(
            controls,
            text="1) Choose your focus market and timeframe. 2) Pull sample data to inspect structure and features.",
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))

        self.symbol_var = tk.StringVar(value="ES=F")
        self.timeframe_var = tk.StringVar(value="5m")
        self.bars_var = tk.IntVar(value=240)

        ttk.Label(controls, text="Symbol").grid(row=1, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Entry(controls, textvariable=self.symbol_var, width=12).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(controls, text="Timeframe").grid(row=1, column=2, sticky=tk.W, padx=(12, 6))
        ttk.Combobox(
            controls,
            textvariable=self.timeframe_var,
            values=("1m", "5m", "15m", "1h", "4h", "1d"),
            state="readonly",
            width=8,
        ).grid(row=1, column=3, sticky=tk.W)

        ttk.Label(controls, text="Bars").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        ttk.Spinbox(controls, from_=60, to=1000, increment=60, textvariable=self.bars_var, width=10).grid(
            row=2, column=1, sticky=tk.W, pady=(6, 0)
        )
        ttk.Button(controls, text="Load sample bars", command=self._load_sample).grid(row=2, column=3, padx=(12, 0), pady=(6, 0))

        controls.columnconfigure(1, weight=1)

        self.text = tk.Text(self, height=18)
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))
        self.summary = ttk.Label(self, anchor=tk.W, justify=tk.LEFT)
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
        atr = float(np.nan_to_num(feat_map.get("atr_14", np.array([0.0]))[latest], nan=0.0))
        rsi = float(np.nan_to_num(feat_map.get("rsi_14", np.array([50.0]))[latest], nan=50.0))
        vol = float(np.nan_to_num(feat_map.get("volatility_close", np.array([0.0]))[latest], nan=0.0))
        trend = "uptrend" if df["close"].tail(30).mean() > df["close"].tail(90).mean() else "down/sideways"
        self.summary.config(
            text=(
                f"Symbol {self.symbol_var.get()} · {self.timeframe_var.get()} — ATR14 {atr:.2f} · RSI14 {rsi:.1f} · "
                f"20-bar vol {vol:.4f}\nRegime hint: {trend}. Move to Train when the setup looks promising."
            )
        )


class TrainTab(BaseTab):
    """Training tab for running local models."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        config = ttk.LabelFrame(self, text="Step 3 · Model lab", padding=12)
        config.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(
            config,
            text="Select a model, choose lookback and optionally calibrate probabilities before saving the artefact.",
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W)

        self.model_type = tk.StringVar(value="logistic")
        self.calibrate_var = tk.BooleanVar(value=True)
        self.lookback_var = tk.IntVar(value=480)

        ttk.Label(config, text="Model").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Radiobutton(config, text="Logistic", value="logistic", variable=self.model_type).grid(
            row=1, column=1, sticky=tk.W, pady=(8, 0)
        )
        ttk.Radiobutton(config, text="Gradient Boosting", value="gbm", variable=self.model_type).grid(
            row=1, column=2, sticky=tk.W, pady=(8, 0)
        )

        ttk.Label(config, text="Lookback bars").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Spinbox(config, from_=240, to=2000, increment=120, textvariable=self.lookback_var, width=10).grid(
            row=2, column=1, sticky=tk.W, pady=(8, 0)
        )
        ttk.Checkbutton(config, text="Calibrate probabilities", variable=self.calibrate_var).grid(
            row=2, column=2, sticky=tk.W, pady=(8, 0)
        )
        ttk.Button(config, text="Train + Score", command=self._train_model).grid(row=2, column=3, padx=(12, 0), pady=(8, 0))

        config.columnconfigure(1, weight=1)

        self.output = tk.Text(self, height=12)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))
        self.status = ttk.Label(self, text="Awaiting training run", anchor=tk.W)
        self.status.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _train_model(self) -> None:
        try:
            lookback = int(self.lookback_var.get())
        except (TypeError, ValueError):
            lookback = 480
        lookback = max(240, min(lookback, 2000))
        df = sample_dataframe(lookback)
        feat_map = features.compute_features(df)
        X = np.column_stack(list(feat_map.values()))
        y = (np.diff(df["close"], prepend=df["close"].iloc[0]) > 0).astype(int)

        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            self.status.config(
                text="Training aborted: target labels lack class diversity.", foreground="#b91c1c"
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
                foreground="#b91c1c",
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
        preprocessing = result.preprocessing or {}
        prep_notes = []
        if preprocessing.get("imputed_cells"):
            prep_notes.append(
                f"imputed {preprocessing['imputed_cells']} feature values"
            )
        if preprocessing.get("dropped_rows"):
            prep_notes.append(
                f"dropped {preprocessing['dropped_rows']} all-NaN rows"
            )
        if preprocessing.get("dropped_columns"):
            prep_notes.append(
                f"removed {preprocessing['dropped_columns']} empty columns"
            )
        if result.retained_columns is not None and result.original_feature_count is not None:
            prep_notes.append(
                f"retained {len(result.retained_columns)} of {result.original_feature_count} feature columns"
            )
        if prep_notes:
            self.log_event(
                "Preprocessing summary: " + ", ".join(prep_notes),
                level="info",
            )
        calibrate_report = "skipped"
        calibration_failed = False
        if self.calibrate_var.get() and len(X) > 60:
            cal_size = max(60, int(len(X) * 0.2))
            X_cal = X[-cal_size:]
            y_cal = y[-cal_size:]
            calibrate_kwargs = {}
            if result.retained_columns is not None:
                calibrate_kwargs["feature_mask"] = result.retained_columns
                if result.original_feature_count is not None:
                    calibrate_kwargs["original_feature_count"] = result.original_feature_count
            try:
                calibrated_path = model.calibrate_classifier(
                    result.model_path,
                    (X_cal, y_cal),
                    **calibrate_kwargs,
                )
            except (ValueError, RuntimeError) as exc:
                calibrate_report = f"calibration failed: {exc}"
                calibration_failed = True
                self.log_event(
                    f"Calibration failed for {result.model_path.name}: {exc}",
                    level="warning",
                )
                self.status.config(
                    text="Calibration skipped due to data quality. Review logs for details.",
                    foreground="#b91c1c",
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
                self.log_event(
                    f"Calibration completed for {result.model_path.name} → {calibrated_path.name}",
                    level="info",
                )
        self.output.delete("1.0", tk.END)
        payload = {
            "model": self.model_type.get(),
            "metrics": result.metrics,
            "threshold": result.threshold,
            "preprocessing": preprocessing,
            "retained_columns": list(result.retained_columns) if result.retained_columns is not None else None,
            "original_feature_count": result.original_feature_count,
            "calibration": calibrate_report,
        }
        self.output.insert(tk.END, json_dumps(payload))
        self.update_section("training", payload)
        if not calibration_failed:
            self.status.config(text="Model artefact refreshed. Continue to Backtest ▶", foreground="")


class BacktestTab(BaseTab):
    """Backtesting tab with a simple equity curve display."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        controls = ttk.LabelFrame(self, text="Step 4 · Backtest", padding=12)
        controls.pack(fill=tk.X, padx=10, pady=(12, 6))

        ttk.Label(
            controls,
            text="Stress test expectancy against synthetic regimes before taking ideas live.",
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W)

        self.sample_var = tk.IntVar(value=720)
        self.strategy_var = tk.StringVar(value="momentum")

        ttk.Label(controls, text="Sample bars").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Spinbox(controls, from_=240, to=5000, increment=240, textvariable=self.sample_var, width=10).grid(
            row=1, column=1, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(controls, text="Playbook").grid(row=1, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.strategy_var,
            values=("momentum", "mean_reversion"),
            state="readonly",
            width=16,
        ).grid(row=1, column=3, sticky=tk.W, pady=(8, 0))
        ttk.Button(controls, text="Run sample backtest", command=self._run_backtest).grid(
            row=2, column=3, padx=(12, 0), pady=(8, 0)
        )

        controls.columnconfigure(1, weight=1)

        self.output = tk.Text(self, height=14)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))
        self.status = ttk.Label(self, text="No simulations yet", anchor=tk.W)
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
        self.status.config(text="Sim complete. If expectancy holds, draft a manual trade plan ▶")


class TradeTab(BaseTab):
    """Trade tab placeholder for polling order/position data."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        self.guard_status = tk.StringVar(value="Topstep Guard: pending review")
        self.guard_label: ttk.Label | None = None
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        intro = ttk.LabelFrame(self, text="Step 5 · Execution guard", padding=12)
        intro.pack(fill=tk.X, padx=10, pady=(12, 6))
        ttk.Label(
            intro,
            text=(
                "Final pre-flight checks before you place manual orders. Refresh the guard summary to confirm "
                "position limits, drawdown caps, and cooldown status."
            ),
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        self.guard_label = ttk.Label(intro, textvariable=self.guard_status, foreground="#1d4ed8")
        self.guard_label.pack(anchor=tk.W, pady=(8, 0))

        ttk.Button(self, text="Refresh Topstep guard", command=self._show_risk).pack(pady=(6, 0))
        self.output = tk.Text(self, height=12)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 12))
        self.output.insert(
            tk.END,
            "Manual execution only. Awaiting guard refresh...\n"
            "Use insights from earlier tabs to justify every trade — and always log rationale.",
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
        sample_size = risk.position_size(50000, profile, atr=3.5, tick_value=12.5, risk_per_trade=0.01)
        guard = "OK" if sample_size > 0 else "DEFENSIVE_MODE"
        self.guard_status.set(f"Topstep Guard: {guard}")
        if self.guard_label is not None:
            colour = "#15803d" if guard == "OK" else "#b91c1c"
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
            warning_message = (
                f"{guard_message}\n\nDEFENSIVE_MODE active. Stand down and review your journal before trading."
            )
            messagebox.showwarning("Topstep Guard", warning_message)


__all__ = [
    "LoginTab",
    "ResearchTab",
    "TrainTab",
    "BacktestTab",
    "TradeTab",
]
