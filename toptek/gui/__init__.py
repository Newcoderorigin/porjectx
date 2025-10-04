"""GUI modules for Toptek."""

from __future__ import annotations

from typing import Any, Dict

DARK_PALETTE: Dict[str, str] = {
    "canvas": "#0b1120",
    "surface": "#111827",
    "surface_alt": "#1e293b",
    "surface_muted": "#18243a",
    "border": "#1f2937",
    "border_muted": "#243047",
    "accent": "#8b5cf6",
    "accent_hover": "#a855f7",
    "accent_active": "#7c3aed",
    "accent_alt": "#38bdf8",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "success": "#22c55e",
    "warning": "#f97316",
    "danger": "#f87171",
}

TEXT_WIDGET_DEFAULTS: Dict[str, Any] = {
    "background": DARK_PALETTE["surface_alt"],
    "foreground": DARK_PALETTE["text"],
    "insertbackground": DARK_PALETTE["accent"],
    "highlightthickness": 0,
    "bd": 0,
    "selectbackground": DARK_PALETTE["accent"],
    "selectforeground": DARK_PALETTE["canvas"],
}

__all__ = ["DARK_PALETTE", "TEXT_WIDGET_DEFAULTS"]

try:  # Re-export optional Live tab when available
    from .live_tab import LiveTab  # type: ignore F401
except ModuleNotFoundError:  # pragma: no cover - legacy deployments
    LiveTab = None  # type: ignore
else:
    __all__.append("LiveTab")
