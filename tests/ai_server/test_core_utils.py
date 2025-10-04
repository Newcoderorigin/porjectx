import logging

import pytest

from toptek.core import utils


def test_build_logger_reuses_handler(monkeypatch):
    fresh = utils.build_logger("ai-server-branch")
    assert fresh.handlers
    logger = utils.build_logger("ai-server-test", level="debug")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG
    handler_count = len(logger.handlers)
    again = utils.build_logger("ai-server-test")
    assert len(again.handlers) == handler_count


def test_load_yaml_and_ensure_directories(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("foo: bar", encoding="utf-8")
    data = utils.load_yaml(yaml_path)
    assert data == {"foo": "bar"}

    paths = utils.AppPaths(
        root=tmp_path, cache=tmp_path / "cache", models=tmp_path / "models"
    )
    utils.ensure_directories(paths)
    assert paths.cache.exists()
    assert paths.models.exists()


def test_timestamp_and_json_dump(monkeypatch):
    ts = utils.timestamp()
    assert ts.tzinfo is utils.DEFAULT_TIMEZONE
    payload = utils.json_dumps({"now": ts})
    assert "now" in payload


def test_env_or_default(monkeypatch):
    monkeypatch.setenv("SAMPLE_KEY", "present")
    assert utils.env_or_default("SAMPLE_KEY", "fallback") == "present"
    assert utils.env_or_default("OTHER_KEY", "fallback") == "fallback"


def test_build_paths(tmp_path):
    config = {"cache_directory": "cache-dir", "models_directory": "models-dir"}
    paths = utils.build_paths(tmp_path, config)
    assert paths.cache == tmp_path / "cache-dir"
    assert paths.models == tmp_path / "models-dir"


def test_version_tuple_and_compare():
    assert utils._version_tuple("1.2.3") == (1, 2, 3)
    assert utils._version_tuple("1.2b0") == (1, 2)
    assert utils._version_tuple("1..2") == (1, 2)
    assert utils._version_tuple("abc") == (0,)
    assert utils._compare_versions((1, 2), (1, 2, 0)) == 0
    assert utils._compare_versions((1, 1), (1, 2)) == -1
    assert utils._compare_versions((2,), (1, 5)) == 1


def test_spec_matches_variants():
    assert utils._spec_matches("1.2.3", ">=1.0,<2.0")
    assert not utils._spec_matches("1.0.0", ">1.0")
    assert utils._spec_matches("1.2.3", "==1.2.3")
    assert not utils._spec_matches("1.2.3", "!=1.2.3")


def test_assert_numeric_stack_pass(monkeypatch):
    recorded = {}

    def fake_version(name: str) -> str:
        recorded[name] = True
        return "1.3.2" if name == "scikit-learn" else "1.9.0"

    monkeypatch.setattr(utils.metadata, "version", fake_version)
    utils.assert_numeric_stack({"scikit-learn": "==1.3.2", "numpy": ">=1.0"})
    assert recorded["scikit-learn"] is True


def test_assert_numeric_stack_failure(monkeypatch):
    def fake_version(name: str) -> str:
        if name == "numpy":
            raise utils.PackageNotFoundError
        return "0.0.1"

    monkeypatch.setattr(utils.metadata, "version", fake_version)
    with pytest.raises(RuntimeError) as excinfo:
        utils.assert_numeric_stack({"numpy": ">=1.0", "scipy": ">=1.0"})
    assert "Missing packages" in str(excinfo.value)


def test_load_yaml_missing(tmp_path):
    missing = tmp_path / "absent.yaml"
    assert utils.load_yaml(missing) == {}


def test_spec_matches_with_empty_clause():
    assert utils._spec_matches("1.2.3", ">=1.0, ,<2.0")
    assert utils._spec_matches("1.2.3", ">=")


def test_assert_numeric_stack_mismatch(monkeypatch):
    monkeypatch.setattr(utils.metadata, "version", lambda name: "0.0.1")
    with pytest.raises(RuntimeError) as excinfo:
        utils.assert_numeric_stack({"numpy": ">=1.0", "scipy": ">=1.0"})
    assert "Version mismatches" in str(excinfo.value)
