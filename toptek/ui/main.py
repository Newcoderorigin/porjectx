"""Graphical entry point responsible for constructing the Tk root."""

from __future__ import annotations

from typing import Dict

from toptek.core import utils
from toptek.gui.app import launch_app
from toptek.ui._ui_theme import get_window, maybe_apply_style_defaults


def _resolve_theme(configs: Dict[str, Dict[str, object]]) -> str | None:
    ui_config = configs.get("ui", {})
    if not isinstance(ui_config, dict):
        return None
    appearance = ui_config.get("appearance", {})
    if not isinstance(appearance, dict):
        return None
    theme_value = appearance.get("theme")
    return theme_value if isinstance(theme_value, str) else None


def launch_ui(*, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
    """Create the Tk root window and delegate to the main application."""

    theme_value = _resolve_theme(configs)
    root = get_window(theme=theme_value)
    maybe_apply_style_defaults()
    launch_app(root=root, configs=configs, paths=paths)
