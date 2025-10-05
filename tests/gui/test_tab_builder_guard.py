"""Tests for the defensive tab builder helpers."""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")
from tkinter import ttk

from toptek.gui.builder import invoke_tab_builder


def _make_root() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    return root


def test_invoke_tab_builder_calls_callable() -> None:
    root = _make_root()

    class Dummy(ttk.Frame):
        def __init__(self, master: tk.Misc) -> None:
            super().__init__(master)
            self.called = False

        def _build(self) -> None:
            self.called = True

    tab = Dummy(root)
    try:
        result = invoke_tab_builder(tab)
        assert tab.called is True
        assert result is None
    finally:
        root.destroy()


def test_invoke_tab_builder_renders_placeholder_when_missing() -> None:
    root = _make_root()

    class Dummy(ttk.Frame):
        pass

    tab = Dummy(root)
    try:
        placeholder = invoke_tab_builder(tab)
        assert placeholder is not None
        assert isinstance(placeholder, ttk.Frame)
        assert placeholder.master is tab
        assert placeholder.winfo_children()
    finally:
        root.destroy()


def test_invoke_tab_builder_placeholder_when_not_callable() -> None:
    root = _make_root()

    class Dummy(ttk.Frame):
        _build = None

    tab = Dummy(root)
    try:
        placeholder = invoke_tab_builder(tab)
        assert placeholder is not None
        assert placeholder.master is tab
    finally:
        root.destroy()
