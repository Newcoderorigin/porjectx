"""CLI-focused tests that do not require heavy numerical dependencies."""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from datetime import datetime

from types import SimpleNamespace

from toptek.core import utils


def test_run_cli_filters_dataframe_when_start_provided(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_filter(df, start):
        captured["df"] = df
        captured["start"] = start
        return "filtered_df"

    def fake_build_features(df, *, cache_dir, engine="pandas"):
        captured["bundle_df"] = df
        return SimpleNamespace(X=[[0.0]], y=[0, 1], meta={})

    def fake_train_classifier(X, y, **kwargs):
        return SimpleNamespace(metrics={}, threshold=0.5)

    fake_numpy = types.ModuleType("numpy")

    class _Result:
        size = 2

    fake_numpy.unique = lambda _arr: _Result()
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_features = types.ModuleType("toptek.features")
    fake_features.build_features = fake_build_features
    monkeypatch.setitem(sys.modules, "toptek.features", fake_features)

    fake_model = types.ModuleType("toptek.core.model")
    fake_model.train_classifier = fake_train_classifier
    monkeypatch.setitem(sys.modules, "toptek.core.model", fake_model)

    fake_backtest = types.ModuleType("toptek.core.backtest")
    fake_backtest.run_backtest = lambda returns, signals: SimpleNamespace(
        hit_rate=0.0, sharpe=0.0, max_drawdown=0.0, expectancy=0.0
    )
    monkeypatch.setitem(sys.modules, "toptek.core.backtest", fake_backtest)

    fake_risk = types.ModuleType("toptek.core.risk")

    class _RiskProfile:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_risk.RiskProfile = _RiskProfile
    monkeypatch.setitem(sys.modules, "toptek.core.risk", fake_risk)

    fake_data = types.ModuleType("toptek.core.data")
    fake_data.sample_dataframe = lambda rows=500: "raw_df"
    monkeypatch.setitem(sys.modules, "toptek.core.data", fake_data)

    main_module = importlib.import_module("toptek.main")

    monkeypatch.setattr(main_module, "_filter_dataframe_by_start", fake_filter)

    args = argparse.Namespace(
        cli="train",
        model="logistic",
        symbol="ES",
        timeframe="5m",
        lookback=50,
        start=datetime(2024, 1, 1),
    )
    configs = {"risk": {}, "app": {}, "features": {}}
    paths = utils.AppPaths(root=tmp_path, cache=tmp_path / "cache", models=tmp_path / "models")

    main_module.run_cli(args, configs, paths)

    assert captured["df"] == "raw_df"
    assert captured["bundle_df"] == "filtered_df"
    assert captured["start"].year == 2024
