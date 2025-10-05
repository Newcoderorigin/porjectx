"""Tests for the defensive tab builder helpers."""

from __future__ import annotations

import pytest

from toptek.gui.builder import MissingTabBuilderError, invoke_tab_builder


def test_invoke_tab_builder_calls_callable() -> None:
    calls: list[bool] = []

    class Dummy:
        def _build(self) -> None:
            calls.append(True)

    invoke_tab_builder(Dummy())
    assert calls == [True]


def test_invoke_tab_builder_raises_when_missing() -> None:
    class Dummy:
        pass

    with pytest.raises(MissingTabBuilderError) as excinfo:
        invoke_tab_builder(Dummy())
    assert "Dummy" in str(excinfo.value)


def test_invoke_tab_builder_raises_when_not_callable() -> None:
    class Dummy:
        _build = None

    with pytest.raises(MissingTabBuilderError):
        invoke_tab_builder(Dummy())
