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


class LoginTab(BaseTab):
    """Login tab that manages .env configuration."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        form = ttk.Frame(self)
        form.pack(padx=10, pady=10, fill=tk.X)
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
        ttk.Button(form, text="Save .env", command=self._save_env).grid(row=len(self.vars), column=0, columnspan=2, pady=10)

    def _env_value(self, key: str) -> str:
        return os.environ.get(key, "")

    def _save_env(self) -> None:
        env_path = self.paths.root / ".env"
        with env_path.open("w", encoding="utf-8") as handle:
            for key, var in self.vars.items():
                handle.write(f"{key}={var.get()}\n")
        messagebox.showinfo("Settings", f"Saved credentials to {env_path}")


class ResearchTab(BaseTab):
    """Research tab to preview sample data."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        ttk.Button(self, text="Load sample bars", command=self._load_sample).pack(pady=10)
        self.text = tk.Text(self, height=25)
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _load_sample(self) -> None:
        df = sample_dataframe(120)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, df.tail(10).to_string())


class TrainTab(BaseTab):
    """Training tab for running local models."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        ttk.Button(self, text="Train logistic model", command=self._train_model).pack(pady=10)
        self.output = tk.Text(self, height=10)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _train_model(self) -> None:
        df = sample_dataframe()
        feat_map = features.compute_features(df)
        X = np.column_stack(list(feat_map.values()))
        y = (np.diff(df["close"], prepend=df["close"].iloc[0]) > 0).astype(int)
        result = model.train_classifier(X, y, models_dir=self.paths.models)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json_dumps(result.metrics))


class BacktestTab(BaseTab):
    """Backtesting tab with a simple equity curve display."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        ttk.Button(self, text="Run sample backtest", command=self._run_backtest).pack(pady=10)
        self.output = tk.Text(self, height=15)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _run_backtest(self) -> None:
        df = sample_dataframe()
        returns = np.log(df["close"]).diff().fillna(0).to_numpy()
        signals = (returns > 0).astype(int)
        result = backtest.run_backtest(returns, signals)
        payload = {
            "hit_rate": result.hit_rate,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "expectancy": result.expectancy,
        }
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, json_dumps(payload))


class TradeTab(BaseTab):
    """Trade tab placeholder for polling order/position data."""

    def __init__(self, master: ttk.Notebook, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
        super().__init__(master, configs, paths)
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Trade management is available once API credentials are configured.").pack(pady=10)
        ttk.Button(self, text="Show risk guard", command=self._show_risk).pack(pady=5)

    def _show_risk(self) -> None:
        profile = risk.RiskProfile(
            max_position_size=self.configs["risk"].get("max_position_size", 1),
            max_daily_loss=self.configs["risk"].get("max_daily_loss", 1000),
            restricted_hours=self.configs["risk"].get("restricted_trading_hours", []),
            atr_multiplier_stop=self.configs["risk"].get("atr_multiplier_stop", 2.0),
            cooldown_losses=self.configs["risk"].get("cooldown_losses", 2),
            cooldown_minutes=self.configs["risk"].get("cooldown_minutes", 30),
        )
        messagebox.showinfo("Risk", json_dumps(profile.__dict__))


__all__ = [
    "LoginTab",
    "ResearchTab",
    "TrainTab",
    "BacktestTab",
    "TradeTab",
]
