from __future__ import annotations

import os
import tkinter as tk

import pytest

from toptek.ui.widgets import ConfidenceRing


@pytest.mark.skipif(tk.TkVersion < 8.0, reason="Tk not available")
@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no X display available")
def test_confidence_ring_updates() -> None:
    root = tk.Tk()
    root.withdraw()
    widget = ConfidenceRing(root)
    widget.update_from_payload({"p": 0.65, "coverage": 0.4, "ev": 0.12})
    assert widget.winfo_exists()
    root.destroy()
