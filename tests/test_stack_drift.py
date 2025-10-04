from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from typing import Dict

import pytest

from toptek.core import utils


def test_assert_numeric_stack_passes_with_matching_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions: Dict[str, str] = {"numpy": "1.26.4", "scipy": "1.10.1"}

    def _fake_version(name: str) -> str:
        return versions[name]

    monkeypatch.setattr(utils.metadata, "version", _fake_version)

    utils.assert_numeric_stack({"numpy": ">=1.26,<1.27", "scipy": ">=1.10,<1.11"})


def test_assert_numeric_stack_raises_with_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements = {"numpy": ">=1.26,<1.27", "scipy": ">=1.10,<1.11"}

    def _fake_version(name: str) -> str:
        if name == "numpy":
            return "1.25.0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(utils.metadata, "version", _fake_version)

    with pytest.raises(RuntimeError) as excinfo:
        utils.assert_numeric_stack(requirements)

    message = str(excinfo.value)
    assert "Incompatible numeric stack" in message
    assert "Missing packages" in message
    assert "numpy" in message
