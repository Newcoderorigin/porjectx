"""Helpers for working with optional ttkbootstrap theming."""

from __future__ import annotations

import tkinter as tk
from typing import Mapping

from toptek.gui import DARK_PALETTE

try:  # pragma: no cover - optional dependency
    import ttkbootstrap as _ttkbootstrap
except Exception:  # pragma: no cover - graceful fallback when absent
    _ttkbootstrap = None

BOOTSTRAP_AVAILABLE: bool = _ttkbootstrap is not None

_THEME_TOKEN_MAP: Mapping[str, str] = {
    "dark": "superhero",
    "light": "flatly",
}


def _resolve_theme_name(theme: str | None) -> str | None:
    if not theme:
        return None
    canonical = theme.strip().lower()
    return _THEME_TOKEN_MAP.get(canonical, theme)


def get_window(theme: str | None):
    """Return a themed root window using ttkbootstrap when available."""

    resolved = _resolve_theme_name(theme)
    if _ttkbootstrap is not None:
        themename = resolved or "superhero"
        return _ttkbootstrap.Window(themename=themename)

    root = tk.Tk()
    root.configure(background=DARK_PALETTE["canvas"])
    return root


def apply_base_spacing(root: tk.Misc) -> None:
    """Apply baseline spacing tweaks for classic Tk deployments."""

    if _ttkbootstrap is not None:
        return

    root.option_add("*TCombobox*Listbox.background", DARK_PALETTE["surface"])
    root.option_add("*TCombobox*Listbox.foreground", DARK_PALETTE["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", DARK_PALETTE["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", DARK_PALETTE["canvas"])
