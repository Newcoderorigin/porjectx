from __future__ import annotations

import logging
from pathlib import Path

import pytest

from toptek.core import utils


def test_build_logger_configures_stream_handler(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logger = utils.build_logger("toptek.test", level="debug")
    assert logger.level == logging.DEBUG
    logger.debug("hello")
    assert caplog.records[-1].message == "hello"
    second = utils.build_logger("toptek.test", level="info")
    assert second.handlers == logger.handlers


def test_load_yaml_and_build_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_text = "cache_directory: cache\nmodels_directory: models\n"
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(config_text, encoding="utf-8")
    data = utils.load_yaml(yaml_path)
    assert data["cache_directory"] == "cache"
    paths = utils.build_paths(tmp_path, data)
    assert paths.cache == tmp_path / "cache"
    assert paths.models == tmp_path / "models"

    utils.ensure_directories(paths)
    assert paths.cache.exists()
    assert paths.models.exists()

    missing_path = tmp_path / "missing.yaml"
    assert utils.load_yaml(missing_path) == {}


def test_timestamp_and_json_dumps() -> None:
    stamp = utils.timestamp()
    assert stamp.tzinfo is not None
    payload = {"ts": stamp}
    dumped = utils.json_dumps(payload)
    assert "ts" in dumped


def test_env_or_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_KEY", raising=False)
    assert utils.env_or_default("TEST_KEY", "fallback") == "fallback"
    monkeypatch.setenv("TEST_KEY", "value")
    assert utils.env_or_default("TEST_KEY", "fallback") == "value"


@pytest.mark.parametrize(
    "value,expected",
    [("1.2.3", (1, 2, 3)), ("1.2b0", (1, 2)), ("invalid", (0,)), ("1..2", (1, 2))],
)
def test_version_tuple_parsing(value: str, expected: tuple[int, ...]) -> None:
    assert utils._version_tuple(value) == expected


@pytest.mark.parametrize(
    "installed,expected",
    [("1.0.0", 0), ("0.9.0", -1), ("1.1.0", 1)],
)
def test_compare_versions(installed: tuple[int, ...] | str, expected: int) -> None:
    if isinstance(installed, str):
        installed_tuple = utils._version_tuple(installed)
    else:
        installed_tuple = installed
    assert utils._compare_versions(installed_tuple, (1, 0, 0)) == expected


@pytest.mark.parametrize(
    "spec,installed,valid",
    [
        (">=1.0,<2.0", "1.5.0", True),
        (">=1.0,<1.5", "1.6.0", False),
        ("==1.0.0", "1.0.0", True),
        ("!=1.0.0", "1.0.0", False),
    ],
)
def test_spec_matches(spec: str, installed: str, valid: bool) -> None:
    assert utils._spec_matches(installed, spec) is valid


def test_spec_matches_handles_empty_segments() -> None:
    assert utils._spec_matches("1.0.0", ">=1.0.0,,<=2.0.0")
    assert utils._spec_matches("1.0.0", ">= ")
