"""Integration tests covering CLI and GUI training reuse of the feature bundle."""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from toptek.core import utils
from toptek.features import FeatureBundle
from toptek.gui.widgets import TrainTab
from toptek.main import run_cli


def _sample_dataframe(rows: int = 128) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(100.0, 105.0, rows)
    noise = np.sin(np.linspace(0, np.pi, rows))
    return pd.DataFrame(
        {
            "open": base + noise * 0.1,
            "high": base + 0.5 + noise * 0.2,
            "low": base - 0.5 + noise * 0.2,
            "close": base + noise * 0.05,
            "volume": np.linspace(1000, 2000, rows),
        },
        index=idx,
    )


def test_cli_training_consumes_feature_bundle(monkeypatch, tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)

    bundle = FeatureBundle(
        X=np.full((96, 3), fill_value=0.5, dtype=float),
        y=np.tile(np.array([0, 1], dtype=np.int8), 48),
        meta={
            "dropped_rows": 0,
            "mask": [1] * 96,
            "feature_names": ["atr_14", "ema_fast", "ema_slow"],
            "valid_index": ["2024-01-01T00:00:00Z"],
            "cache_key": "dummy",
        },
    )

    captured: dict[str, object] = {}

    def fake_build_features(df, *, cache_dir, engine="pandas"):
        captured["cache_dir"] = cache_dir
        captured["rows"] = len(df)
        return bundle

    def fake_train_classifier(X, y, **kwargs):
        captured["train_X"] = X
        captured["train_y"] = y
        return SimpleNamespace(
            model_path=tmp_path / "models" / "logistic.pkl",
            metrics={"accuracy": 0.75},
            threshold=0.5,
        )

    monkeypatch.setattr("toptek.main.build_features", fake_build_features)
    monkeypatch.setattr("toptek.main.model.train_classifier", fake_train_classifier)
    monkeypatch.setattr(
        "toptek.main.data.sample_dataframe", lambda: _sample_dataframe(140)
    )

    args = argparse.Namespace(
        cli="train",
        model="logistic",
        symbol="ES",
        timeframe="5m",
        lookback="90d",
        start=None,
    )
    configs = {"risk": {}, "app": {}, "features": {}}
    paths = utils.AppPaths(
        root=tmp_path,
        cache=tmp_path / "cache",
        models=tmp_path / "models",
        logs=tmp_path / "logs",
        reports=tmp_path / "reports",
    )

    run_cli(args, configs, paths)

    assert captured["cache_dir"] == paths.cache
    assert captured["train_X"] is bundle.X
    assert captured["train_y"] is bundle.y
    assert "Feature pipeline dropped" not in caplog.text


def test_train_tab_uses_feature_bundle(monkeypatch, tmp_path, caplog) -> None:
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")

    root.withdraw()

    bundle = FeatureBundle(
        X=np.arange(120, dtype=float).reshape(60, 2),
        y=np.tile(np.array([0, 1], dtype=np.int8), 30),
        meta={
            "dropped_rows": 0,
            "mask": [1] * 60,
            "feature_names": ["atr_14", "ema_fast"],
            "valid_index": ["2024-01-01T00:00:00Z"],
            "cache_key": "dummy",
        },
    )

    captured: dict[str, object] = {}

    def fake_build_features(df, *, cache_dir, engine="pandas"):
        captured["cache_dir"] = cache_dir
        captured["rows"] = len(df)
        return bundle

    def fake_train_classifier(X, y, **kwargs):
        captured["train_X"] = X
        captured["train_y"] = y
        return SimpleNamespace(
            model_path=tmp_path / "models" / "bundle.pkl",
            metrics={"accuracy": 0.8},
            threshold=0.5,
            preprocessing={},
            retained_columns=None,
            original_feature_count=2,
        )

    monkeypatch.setattr("toptek.gui.widgets.build_features", fake_build_features)
    monkeypatch.setattr(
        "toptek.gui.widgets.sample_dataframe", lambda rows: _sample_dataframe(rows)
    )
    monkeypatch.setattr(
        "toptek.gui.widgets.model.train_classifier", fake_train_classifier
    )
    monkeypatch.setattr("tkinter.messagebox.showwarning", lambda *args, **kwargs: None)
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *args, **kwargs: None)

    notebook = ttk.Notebook(root)
    notebook.pack()

    configs: dict[str, dict[str, object]] = {}
    paths = utils.AppPaths(
        root=tmp_path,
        cache=tmp_path / "cache",
        models=tmp_path / "models",
        logs=tmp_path / "logs",
        reports=tmp_path / "reports",
    )

    tab = TrainTab(notebook, configs, paths)
    tab.calibrate_var.set(False)

    caplog.set_level(logging.WARNING)
    tab._train_model()

    assert captured["cache_dir"] == paths.cache
    assert captured["train_X"] is bundle.X
    assert captured["train_y"] is bundle.y
    assert "Feature pipeline dropped" not in caplog.text

    root.destroy()
