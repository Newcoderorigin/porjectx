"""Tkinter widget visualising probability confidence."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class ConfidenceTheme:
    ring_width: int = 12
    stable_color: str = "#3fd08f"
    watch_color: str = "#f0c419"
    alert_color: str = "#ff6f61"
    background: str = "#1b1f24"
    text_color: str = "#e5ecf4"


class ConfidenceRing(tk.Frame):
    """Ring gauge displaying probability, coverage, and EV."""

    def __init__(
        self, master: tk.Misc | None = None, theme: ConfidenceTheme | None = None
    ) -> None:
        super().__init__(master, bg=(theme.background if theme else "#1b1f24"))
        self._theme = theme or ConfidenceTheme()
        self._canvas = tk.Canvas(
            self,
            width=140,
            height=140,
            highlightthickness=0,
            bg=self._theme.background,
        )
        self._canvas.grid(row=0, column=0, padx=8, pady=8)
        self._label = tk.Label(
            self,
            fg=self._theme.text_color,
            bg=self._theme.background,
            font=("Inter", 12, "bold"),
        )
        self._label.grid(row=1, column=0, pady=(0, 4))
        self._chip = tk.Label(
            self,
            fg=self._theme.text_color,
            bg=self._theme.background,
            font=("Inter", 10),
        )
        self._chip.grid(row=2, column=0, pady=(0, 6))
        self._metric = tk.Label(
            self,
            fg=self._theme.text_color,
            bg=self._theme.background,
            font=("Inter", 9),
        )
        self._metric.grid(row=3, column=0, pady=(0, 6))
        self._draw_ring(0.5)

    def _draw_ring(self, probability: float) -> None:
        radius = 120
        start_angle = 90
        extent = probability * 360
        arc_color = self._severity_color(probability)
        self._canvas.delete("all")
        self._canvas.create_oval(
            10,
            10,
            radius,
            radius,
            outline="#2c3238",
            width=self._theme.ring_width,
        )
        self._canvas.create_arc(
            10,
            10,
            radius,
            radius,
            start=start_angle,
            extent=-extent,
            style=tk.ARC,
            outline=arc_color,
            width=self._theme.ring_width,
            capstyle=tk.ROUND,
        )
        self._canvas.create_text(
            radius / 2,
            radius / 2,
            text=f"{probability*100:.1f}%",
            fill=self._theme.text_color,
            font=("Inter", 16, "bold"),
        )

    def _severity_color(self, probability: float) -> str:
        if probability >= 0.6:
            return self._theme.stable_color
        if probability >= 0.5:
            return self._theme.watch_color
        return self._theme.alert_color

    def update_from_payload(self, payload: Mapping[str, float]) -> None:
        probability = float(payload.get("p", payload.get("probability", 0.5)))
        coverage = float(payload.get("coverage", 0.0))
        ev = float(payload.get("ev", payload.get("expected_value", 0.0)))
        confidence = float(
            payload.get("conf", payload.get("confidence", abs(probability - 0.5) * 2))
        )
        self._draw_ring(probability)
        self._label.configure(text=f"Confidence {confidence*100:.0f}%")
        self._chip.configure(text=f"EV {ev:.3f}")
        self._metric.configure(text=f"Coverage {coverage*100:.1f}%")


__all__ = ["ConfidenceTheme", "ConfidenceRing"]
