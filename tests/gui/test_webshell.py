from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from urllib.request import urlopen

import pytest
import toptek.core as core_package

sys.modules.setdefault("core", core_package)

from toptek.core import utils
import toptek.core.utils as utils_module
from toptek.gui.webshell import find_web_build, launch_web_frontend


def _paths(tmp_path: Path) -> utils.AppPaths:
    return utils.AppPaths(root=tmp_path, cache=tmp_path / "cache", models=tmp_path / "models")


def test_find_web_build_missing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    module = importlib.reload(utils_module)
    module.ensure_directories(paths)
    assert module.load_yaml(tmp_path / "missing.yml") == {}
    (tmp_path / "config.yml").write_text("answer: 42", encoding="utf-8")
    data = module.load_yaml(tmp_path / "config.yml")
    assert data["answer"] == 42
    logging.getLogger("test-webshell-unique").handlers.clear()
    module.build_logger("test-webshell-unique").info("logger wired")
    built_paths = module.build_paths(tmp_path, {})
    assert built_paths.cache.name == "cache"
    logging.getLogger("test-webshell-another").handlers.clear()
    logger = module.build_logger("test-webshell-another")
    assert logger.handlers and logger.handlers[0].formatter is not None
    assert module.env_or_default("TOPTEK_MISSING", "fallback") == "fallback"
    assert "answer" in module.json_dumps(data)
    assert module.timestamp().tzinfo is not None
    assert find_web_build(paths) is None


def test_find_web_build_locates_dist(tmp_path: Path) -> None:
    dist = tmp_path / "toptek" / "ui" / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    paths = _paths(tmp_path)
    assert find_web_build(paths) == dist


def test_launch_web_frontend_serves_assets(tmp_path: Path) -> None:
    dist = tmp_path / "toptek" / "ui" / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>Toptek</body></html>", encoding="utf-8")
    paths = _paths(tmp_path)
    handle = launch_web_frontend(paths, auto_open=False)
    assert handle is not None
    try:
        with urlopen(handle.url, timeout=2) as response:
            body = response.read().decode("utf-8")
        assert "Toptek" in body
    finally:
        handle.stop()


def test_spec_helpers_cover_version_logic() -> None:
    module = importlib.reload(utils_module)
    assert module._version_tuple("1.2.3") == (1, 2, 3)
    assert module._version_tuple("1.2b") == (1, 2)
    assert module._version_tuple("1..2") == (1, 2)
    assert module._version_tuple("abc") == (0,)
    assert module._compare_versions((1, 2, 0), (1, 2, 0)) == 0
    assert module._compare_versions((1, 2, 0), (1, 3, 0)) == -1
    assert module._compare_versions((2, 0), (1, 9, 9)) == 1
    assert module._spec_matches("1.2.0", ">=1.1,<1.3")
    assert not module._spec_matches("1.0.0", ">=1.1")
    assert not module._spec_matches("1.0.0", "==1.1.0")
    assert not module._spec_matches("1.0.0", "!=1.0.0")
    assert not module._spec_matches("1.2.0", "<=1.1.0")
    assert not module._spec_matches("1.0.0", ">1.0.0")
    assert module._spec_matches("1.0.0", ",>=1.0")
    assert module._spec_matches("1.0.0", ">=,")
    assert "numpy" in module.STACK_REQUIREMENTS


def test_numeric_stack_guard_reports_missing() -> None:
    module = importlib.reload(utils_module)
    with pytest.raises(RuntimeError) as exc:
        module.assert_numeric_stack({"fakepkg": ">=1.0"})
    assert "fakepkg" in str(exc.value)
