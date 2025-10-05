"""Helpers for working with optional ttkbootstrap theming."""

from __future__ import annotations

import tkinter as tk
from functools import lru_cache
from typing import Mapping

from toptek.gui import DARK_PALETTE

_THEME_TOKEN_MAP: Mapping[str, str] = {
    "dark": "superhero",
    "light": "flatly",
}


def _resolve_theme_name(theme: str | None) -> str | None:
    if not theme:
        return None
    canonical = theme.strip().lower()
    return _THEME_TOKEN_MAP.get(canonical, theme)


@lru_cache(maxsize=1)
def _import_ttkbootstrap():  # pragma: no cover - optional dependency
    try:
        import ttkbootstrap  # type: ignore
    except Exception:
        return None
    return ttkbootstrap


def get_window(theme: str | None):
    """Return a themed root window using ttkbootstrap when available."""

    resolved = _resolve_theme_name(theme)
    ttkbootstrap = _import_ttkbootstrap()
    if ttkbootstrap is not None:
        themename = resolved or "superhero"
        return ttkbootstrap.Window(themename=themename)

    root = tk.Tk()
    root.configure(background=DARK_PALETTE["canvas"])
    return root


def apply_base_spacing(root: tk.Misc) -> None:
    """Apply baseline spacing tweaks for classic Tk deployments."""

    if _import_ttkbootstrap() is not None:
        return

    root.option_add("*TCombobox*Listbox.background", DARK_PALETTE["surface"])
    root.option_add("*TCombobox*Listbox.foreground", DARK_PALETTE["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", DARK_PALETTE["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", DARK_PALETTE["canvas"])
