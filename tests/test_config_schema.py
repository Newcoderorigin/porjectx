from __future__ import annotations

from pathlib import Path

from toptek.core import ui_config


def test_ui_config_includes_lmstudio_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "ui.yml"
    config_path.write_text("{}\n", encoding="utf-8")

    cfg = ui_config.load_ui_config(config_path, env={})

    assert cfg.lmstudio.enabled is True
    assert cfg.lmstudio.base_url == "http://localhost:1234/v1"
    assert cfg.lmstudio.model == "llama-3.1-8b-instruct"
    assert cfg.lmstudio.max_tokens == 512
    assert cfg.as_dict()["lmstudio"]["temperature"] == 0.0


def test_repository_ui_config_matches_schema() -> None:
    project_cfg = ui_config.load_ui_config(Path("configs/ui.yml"), env={})
    lmstudio = project_cfg.lmstudio
    assert lmstudio.enabled is True
    assert lmstudio.timeout_s == 30
    assert "Autostealth Evolution" in lmstudio.system_prompt
