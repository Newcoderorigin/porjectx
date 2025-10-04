"""Configuration schema checks for UI additions."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from toptek.core import utils

CONFIG_PATH = Path("configs/ui.yml")


REQUIRED_LM_KEYS = {
    "enabled",
    "base_url",
    "api_key",
    "model",
    "system_prompt",
    "max_tokens",
    "temperature",
    "top_p",
    "timeout_s",
}

REQUIRED_FUTURES_KEYS = {
    "enabled",
    "default_symbol",
    "default_interval",
    "sources",
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_lmstudio_block_present() -> None:
    config = load_config()
    section = config.get("lmstudio")
    assert isinstance(section, dict)
    assert REQUIRED_LM_KEYS.issubset(section.keys())


def test_futures_research_block_present() -> None:
    config = load_config()
    section = config.get("futures_research")
    assert isinstance(section, dict)
    assert REQUIRED_FUTURES_KEYS.issubset(section.keys())


def test_futures_utils_helper_coverage(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_logger = utils.logging.getLogger("coverage")
    base_logger.handlers.clear()
    logger = utils.build_logger("coverage", level="info")
    assert logger.name == "coverage"

    sample_yaml = tmp_path / "sample.yml"
    sample_yaml.write_text("value: 1\n", encoding="utf-8")
    parsed = utils.load_yaml(sample_yaml)
    assert parsed == {"value": 1}
    assert utils.load_yaml(tmp_path / "missing.yml") == {}

    paths = utils.AppPaths(root=tmp_path, cache=tmp_path / "cache", models=tmp_path / "models")
    utils.ensure_directories(paths)
    assert paths.cache.exists() and paths.models.exists()

    stamp = utils.timestamp()
    payload = utils.json_dumps({"when": stamp})
    assert "when" in payload

    assert utils.env_or_default("_UNSET_ENV_VAR_", "fallback") == "fallback"

    derived_paths = utils.build_paths(tmp_path, {"cache_directory": "cache", "models_directory": "models"})
    assert derived_paths.cache == paths.cache
    assert derived_paths.models == paths.models

    assert utils._version_tuple("1.2.3") == (1, 2, 3)
    assert utils._version_tuple("release-2") == (0,)
    assert utils._version_tuple("1..2") == (1, 2)
    assert utils._compare_versions((1, 0), (1, 0)) == 0
    assert utils._compare_versions((1, 0), (2, 0)) == -1
    assert utils._compare_versions((2, 0), (1, 0)) == 1
    assert utils._spec_matches("1.2.3", ">=1.0,<2.0")
    assert not utils._spec_matches("1.0.0", ">1.0.0")
    assert utils._spec_matches("1.0.0", " ,>=0")
    assert utils._spec_matches("1.0.0", ">= ")

    utils.build_logger("coverage", level="debug")

    monkeypatch.setattr(utils.metadata, "version", lambda name: "1.0.0")
    utils.assert_numeric_stack({"pytest": ">=0"})

    def failing_version(name: str) -> str:
        if name == "missing":
            raise utils.PackageNotFoundError
        return "0.5.0"

    monkeypatch.setattr(utils.metadata, "version", failing_version)
    with pytest.raises(RuntimeError):
        utils.assert_numeric_stack({"missing": ">=1.0"})
    with pytest.raises(RuntimeError):
        utils.assert_numeric_stack({"pytest": ">=1.0"})
