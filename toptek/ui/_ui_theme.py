"""Bridge module providing Tk window helpers for the UI entry point."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from toptek._ui_theme import apply_base_spacing as _apply_base_spacing
from toptek._ui_theme import get_window as _get_window

_CURRENT_ROOT: Optional[tk.Misc] = None


def get_window(theme: str | None):
    """Return a themed root window and memoise it for style defaults."""

    global _CURRENT_ROOT
    root = _get_window(theme)
    _CURRENT_ROOT = root
    return root


def maybe_apply_style_defaults() -> None:
    """Apply baseline style defaults if ttkbootstrap is unavailable."""

    if _CURRENT_ROOT is None:
        return
    _apply_base_spacing(_CURRENT_ROOT)
