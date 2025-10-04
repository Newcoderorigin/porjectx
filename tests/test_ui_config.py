from __future__ import annotations
from pathlib import Path

import pytest

from core import ui_config


def test_load_ui_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "ui.yml"
    path.write_text("{}\n", encoding="utf-8")
    cfg = ui_config.load_ui_config(path, env={})
    assert cfg.shell.symbol == "ES=F"
    assert cfg.shell.interval == "5m"
    assert cfg.shell.research_bars == 240
    assert cfg.chart.fps == 12
    assert cfg.status.login.idle == "Awaiting verification"
    assert cfg.appearance.theme == "dark"


def test_load_ui_config_env_overrides(tmp_path: Path) -> None:
    path = tmp_path / "ui.yml"
    path.write_text(
        "shell:\n  symbol: ES=F\n  calibrate: true\nchart:\n  fps: 8\n",
        encoding="utf-8",
    )
    env = {
        "TOPTEK_UI_SYMBOL": "NQ=F",
        "TOPTEK_UI_CALIBRATE": "false",
        "TOPTEK_UI_LOOKBACK_BARS": "960",
        "TOPTEK_UI_FPS": "24",
        "TOPTEK_UI_THEME": "dark",
    }
    cfg = ui_config.load_ui_config(path, env=env)
    assert cfg.shell.symbol == "NQ=F"
    assert cfg.shell.calibrate is False
    assert cfg.shell.lookback_bars == 960
    assert cfg.chart.fps == 24
    assert cfg.appearance.theme == "dark"


def test_load_ui_config_validation(tmp_path: Path) -> None:
    path = tmp_path / "ui.yml"
    path.write_text("chart:\n  fps: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ui_config.load_ui_config(path, env={})
