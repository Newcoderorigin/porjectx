"""Yahoo-backed futures research tab for the Tkinter GUI."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import List, Mapping, MutableSequence
import urllib.error
import urllib.parse
import urllib.request

try:  # pragma: no cover - exercised via import behaviour
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:  # pragma: no cover - headless environments
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

from toptek.gui import DARK_PALETTE


def build_yahoo_csv_url(symbol: str, interval: str, start: datetime, end: datetime) -> str:
    """Construct the Yahoo Finance CSV download URL."""

    quoted_symbol = urllib.parse.quote_plus(symbol)
    params = {
        "period1": str(int(start.timestamp())),
        "period2": str(int(end.timestamp())),
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }
    query = urllib.parse.urlencode(params)
    return f"https://query1.finance.yahoo.com/v7/finance/download/{quoted_symbol}?{query}"


def fetch_futures_history(symbol: str, interval: str, days: int = 30) -> List[Mapping[str, str]]:
    """Fetch historical futures data as dictionaries."""

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    url = build_yahoo_csv_url(symbol, interval, start, end)
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"Failed to download futures data: {exc}") from exc
    reader = csv.DictReader(io.StringIO(raw))
    records: List[Mapping[str, str]] = []
    for row in reader:
        if not row.get("Date"):
            continue
        records.append(row)
    return records


_BaseFrame = ttk.Frame if ttk is not None else object


class FuturesResearchTab(_BaseFrame):
    """UI wrapper for futures research tooling."""

    def __init__(self, master: ttk.Notebook, config: Mapping[str, object]) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available for FuturesResearchTab")
        super().__init__(master, style="DashboardBackground.TFrame")  # type: ignore[misc]
        self._config = dict(config)
        self.symbol_var = tk.StringVar(value=str(self._config.get("default_symbol", "ES=F")))
        self.interval_var = tk.StringVar(value=str(self._config.get("default_interval", "1d")))
        self.status_var = tk.StringVar(value="Stale")
        self.error_var = tk.StringVar(value="")
        self._data: MutableSequence[Mapping[str, str]] = []

        self.sparkline = tk.Canvas(self, height=120, background=DARK_PALETTE["surface_alt"])
        self.table = ttk.Treeview(
            self,
            columns=("Date", "Open", "High", "Low", "Close", "Volume"),
            show="headings",
            height=5,
        )
        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, style="DashboardBackground.TFrame")
        controls.pack(fill=tk.X, padx=16, pady=(16, 8))

        ttk.Label(controls, text="Symbol").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.symbol_var, width=12).pack(
            side=tk.LEFT, padx=(4, 12)
        )

        ttk.Label(controls, text="Interval").pack(side=tk.LEFT)
        interval_box = ttk.Combobox(
            controls,
            textvariable=self.interval_var,
            values=("1d", "1wk", "1mo"),
            width=6,
            state="readonly",
        )
        interval_box.pack(side=tk.LEFT)

        ttk.Button(controls, text="Load", command=self.load_data).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        status_badge = ttk.Label(
            controls,
            textvariable=self.status_var,
            style="StatusInfo.TLabel",
            background=DARK_PALETTE["surface_alt"],
        )
        status_badge.pack(side=tk.RIGHT)

        ttk.Label(
            self,
            textvariable=self.error_var,
            foreground=DARK_PALETTE["danger"],
            style="StatusInfo.TLabel",
        ).pack(fill=tk.X, padx=16)

        spark_container = ttk.LabelFrame(self, text="Sparkline")
        spark_container.pack(fill=tk.X, padx=16, pady=(0, 12))
        self.sparkline.pack(in_=spark_container, fill=tk.X, padx=8, pady=8)

        table_container = ttk.LabelFrame(self, text="OHLC Preview")
        table_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        for column in self.table["columns"]:
            self.table.heading(column, text=column)
            self.table.column(column, width=90, anchor=tk.CENTER)
        self.table.pack(in_=table_container, fill=tk.BOTH, expand=True, padx=8, pady=8)

    def load_data(self) -> None:
        symbol = self.symbol_var.get().strip()
        interval = self.interval_var.get().strip() or "1d"
        if not symbol:
            self.error_var.set("Enter a symbol")
            self.status_var.set("Error")
            return
        self.error_var.set("")
        try:
            records = fetch_futures_history(symbol, interval)
        except Exception as exc:  # pragma: no cover - UI path
            self.error_var.set(str(exc))
            self.status_var.set("Error")
            return
        if not records:
            self.error_var.set("No data returned for symbol")
            self.status_var.set("Error")
            return
        self._data = list(records)
        self._render_sparkline()
        self._render_table()
        self.status_var.set("Fresh")

    def _render_sparkline(self) -> None:
        self.sparkline.delete("all")
        closes = [float(row["Close"]) for row in self._data if row.get("Close")]
        if not closes:
            return
        width = int(self.sparkline.winfo_width() or 400)
        height = int(self.sparkline.winfo_height() or 120)
        min_close = min(closes)
        max_close = max(closes)
        span = max_close - min_close
        if span == 0:
            span = max_close or 1.0
        points = closes[- min(50, len(closes)) :]
        step = width / max(1, len(points) - 1)
        coords: List[float] = []
        for index, close in enumerate(points):
            x = index * step
            if span:
                y = height - ((close - min_close) / span) * (height - 10) - 5
            else:
                y = height / 2
            coords.extend([x, y])
        if len(coords) >= 4:
            self.sparkline.create_line(
                coords,
                fill=DARK_PALETTE["accent"],
                width=2,
                smooth=True,
            )

    def _render_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        latest = list(self._data)[-5:]
        for row in reversed(latest):
            self.table.insert(
                "",
                tk.END,
                values=(
                    row.get("Date", ""),
                    row.get("Open", ""),
                    row.get("High", ""),
                    row.get("Low", ""),
                    row.get("Close", ""),
                    row.get("Volume", ""),
                ),
            )
