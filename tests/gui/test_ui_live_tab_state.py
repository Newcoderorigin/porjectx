"""Tk-aware smoke tests for the UI LiveTab widget."""

from __future__ import annotations

from typing import Any

import pytest

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

from toptek.ui.live_tab import LiveTab  # noqa: E402


@pytest.fixture
def tk_root() -> Any:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def test_live_tab_chat_log_disabled_initially(tk_root: Any) -> None:
    notebook = ttk.Notebook(tk_root)
    notebook.pack()

    class DummyClient:
        def chat(self, messages):  # pragma: no cover - not used in test
            raise AssertionError("chat should not be called")

    tab = LiveTab(notebook, {"model": "test-model"}, client=DummyClient())

    assert tab.chat_log["state"] == "disabled"
