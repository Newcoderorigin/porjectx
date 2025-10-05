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


def invoke_tab_builder(tab: object, *, attr: str = "_build") -> ttk.Frame | None:
    """Invoke the tab layout builder if available.

    When *attr* is missing or not callable, a friendly placeholder is rendered
    so the surrounding notebook can continue initialising without raising a
    hard exception.
    """

    builder = getattr(tab, attr, None)
    if callable(builder):
        builder()
        return None

    error = MissingTabBuilderError(tab.__class__.__name__, attr)
    if isinstance(tab, tk.Misc):
        placeholder = build_missing_tab_placeholder(
            tab, tab_name=tab.__class__.__name__, error=error
        )
        placeholder.pack(fill=tk.BOTH, expand=True)
        return placeholder
    raise error


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
