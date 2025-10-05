"""Helper utilities for building GUI tabs safely."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk


@dataclass(frozen=True)
class MissingTabBuilderError(RuntimeError):
    """Raised when a tab cannot locate its layout builder."""

    tab_name: str
    attr: str = "_build"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        RuntimeError.__init__(
            self,
            f"{self.tab_name} is missing a callable '{self.attr}()' layout builder.",
        )


def invoke_tab_builder(tab: object, *, attr: str = "_build") -> None:
    """Invoke the tab layout builder if available.

    If *attr* is missing or not callable the function raises
    :class:`MissingTabBuilderError` so callers can provide a fallback view
    instead of crashing the entire GUI launch.
    """

    builder = getattr(tab, attr, None)
    if not callable(builder):
        raise MissingTabBuilderError(tab.__class__.__name__, attr)
    builder()


def build_missing_tab_placeholder(
    parent: tk.Misc,
    *,
    tab_name: str,
    error: MissingTabBuilderError,
) -> ttk.Frame:
    """Render a gentle placeholder when a tab cannot construct itself."""

    frame = ttk.Frame(parent, style="DashboardBackground.TFrame")
    frame.columnconfigure(0, weight=1)

    heading = ttk.Label(
        frame,
        text=f"{tab_name} upgrade required",
        style="Surface.TLabel",
        justify=tk.LEFT,
        anchor=tk.W,
    )
    heading.grid(row=0, column=0, sticky=tk.W, padx=16, pady=(18, 6))

    body = ttk.Label(
        frame,
        text=(
            "This tab could not be initialised because its layout helper is "
            "missing. Please update to the latest ProjectX cockpit build "
            "and restart the application.\n\nDetails: "
            f"{error}"
        ),
        style="Surface.TLabel",
        wraplength=520,
        justify=tk.LEFT,
        anchor=tk.W,
    )
    body.grid(row=1, column=0, sticky=tk.W, padx=16, pady=(0, 18))

    return frame
