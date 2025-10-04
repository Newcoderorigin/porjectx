from pathlib import Path

import pytest

from toptek.ai_server import config


def test_load_settings_respects_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "sys", type("S", (), {"version_info": (3, 11, 0)})())
    cfg = tmp_path / "ai.yml"
    cfg.write_text(
        """
base_url: "http://localhost:9999/v1"
port: 9999
auto_start: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://example.com/v1")
    monkeypatch.setenv("LMSTUDIO_PORT", "4321")
    monkeypatch.setenv("LMSTUDIO_MODEL", "demo-model")
    monkeypatch.setenv("AI_DEFAULT_ROLE", "system role")

    settings = config.load_settings(cfg)

    assert settings.base_url == "http://example.com/v1"
    assert settings.port == 4321
    assert settings.default_model == "demo-model"
    assert settings.default_role == "system role"
    assert settings.auto_start is False


def test_python_version_guard(monkeypatch):
    class DummySys:
        version_info = (3, 12, 0)

    monkeypatch.setattr(config, "sys", DummySys)
    with pytest.raises(RuntimeError):
        config.load_settings(Path("missing"))
