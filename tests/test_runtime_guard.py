from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from toptek import runtime_guard


def test_warn_if_unsupported_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_guard,
        "sys",
        SimpleNamespace(
            version_info=SimpleNamespace(major=3, minor=12, micro=0, releaselevel="final", serial=0)
        ),
    )
    with pytest.warns(RuntimeWarning):
        runtime_guard.warn_if_unsupported()


def test_warn_if_unsupported_noop_for_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_guard,
        "sys",
        SimpleNamespace(
            version_info=SimpleNamespace(major=3, minor=11, micro=9, releaselevel="final", serial=0)
        ),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        runtime_guard.warn_if_unsupported()
    assert not captured
