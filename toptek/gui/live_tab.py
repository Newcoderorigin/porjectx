"""Live trading tab wiring for Tkinter GUI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List
import tkinter as tk
from tkinter import ttk

from toptek.core import utils

from . import TEXT_WIDGET_DEFAULTS


SuccessCallback = Callable[[Dict[str, Any], Dict[str, Any]], None]
ErrorCallback = Callable[[Dict[str, Any], Exception], None]


class LiveTab(ttk.Frame):
    """Interactive controls for dispatching live trading requests."""

    DEFAULT_REQUEST: Dict[str, Any] = {
        "account_id": "",
        "symbol": "",
        "quantity": 1,
        "order_type": "MARKET",
        "time_in_force": "DAY",
        "route": "SIM",
        "limit_price": "",
        "stop_price": "",
    }

    def __init__(
        self,
        master: ttk.Notebook,
        configs: Dict[str, Dict[str, object]],
        paths: Any,
        *,
        client: Any | None = None,
        metrics_fetcher: Callable[[], Dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(master, style="DashboardBackground.TFrame")
        self.configs = configs
        self.paths = paths
        self.client = client
        self.metrics_fetcher = metrics_fetcher
        self._success_handlers: List[SuccessCallback] = []
        self._error_handlers: List[ErrorCallback] = []
        live_config = self.configs.setdefault("live", {})
        defaults = dict(self.DEFAULT_REQUEST)
        defaults.update(live_config.get("defaults", {}))
        self.request_defaults: Dict[str, Any] = defaults
        live_config.setdefault("defaults", dict(self.request_defaults))
        live_config.setdefault("last_request", None)
        live_config.setdefault("metrics", {})

        self.status_var = tk.StringVar(value="Ready to trade")
        self.metrics_visible = tk.BooleanVar(value=True)

        self.account_var = tk.StringVar(value=str(self.request_defaults["account_id"]))
        self.symbol_var = tk.StringVar(value=str(self.request_defaults["symbol"]))
        self.quantity_var = tk.StringVar(value=str(self.request_defaults["quantity"]))
        self.order_type_var = tk.StringVar(value=str(self.request_defaults["order_type"]))
        self.tif_var = tk.StringVar(value=str(self.request_defaults["time_in_force"]))
        self.route_var = tk.StringVar(value=str(self.request_defaults["route"]))
        self.limit_var = tk.StringVar(value=str(self.request_defaults["limit_price"]))
        self.stop_var = tk.StringVar(value=str(self.request_defaults["stop_price"]))

        self.metrics_state: Dict[str, Any] = {
            "orders_sent": 0,
            "errors": 0,
            "last_status": self.status_var.get(),
            "last_refresh": None,
        }

        self._build_controls()
        self._build_metrics()

    # ------------------------------------------------------------------ UI --
    def _build_controls(self) -> None:
        container = ttk.Frame(self, style="DashboardBackground.TFrame")
        container.pack(fill=tk.X, padx=16, pady=12)

        grid = ttk.Frame(container, style="DashboardBackground.TFrame")
        grid.pack(fill=tk.X)

        self._add_field(grid, "Account", self.account_var, row=0, column=0)
        self._add_field(grid, "Symbol", self.symbol_var, row=0, column=1)
        self._add_field(grid, "Quantity", self.quantity_var, row=0, column=2)
        self._add_field(grid, "Order type", self.order_type_var, row=1, column=0)
        self._add_field(grid, "Time-in-force", self.tif_var, row=1, column=1)
        self._add_field(grid, "Route", self.route_var, row=1, column=2)
        self._add_field(grid, "Limit", self.limit_var, row=2, column=0)
        self._add_field(grid, "Stop", self.stop_var, row=2, column=1)

        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_columnconfigure(2, weight=1)

        actions = ttk.Frame(container, style="DashboardBackground.TFrame")
        actions.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(actions, text="Send order", command=self.submit_order).pack(
            side=tk.LEFT
        )
        ttk.Checkbutton(
            actions,
            text="Show metrics",
            variable=self.metrics_visible,
            command=self._update_metrics_visibility,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(
            actions,
            textvariable=self.status_var,
            style="StatusInfo.TLabel",
        ).pack(side=tk.RIGHT)

    def _add_field(
        self,
        master: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        *,
        row: int,
        column: int,
    ) -> None:
        ttk.Label(master, text=label).grid(row=row, column=column, sticky=tk.W, padx=4)
        entry = ttk.Entry(master, textvariable=variable, width=18)
        entry.grid(row=row, column=column, padx=4, pady=(4, 8), sticky=tk.EW)

    def _build_metrics(self) -> None:
        self.metrics_frame = ttk.Frame(self, style="DashboardBackground.TFrame")
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        self.metrics_output = tk.Text(self.metrics_frame, height=10, wrap="word")
        self.metrics_output.pack(fill=tk.BOTH, expand=True)
        self._style_text_widget(self.metrics_output)
        self.refresh_metrics()

    # --------------------------------------------------------------- Actions --
    def register_callbacks(
        self,
        *,
        on_success: SuccessCallback | Iterable[SuccessCallback] | None = None,
        on_error: ErrorCallback | Iterable[ErrorCallback] | None = None,
    ) -> None:
        """Register callbacks invoked after request completion."""

        if on_success is not None:
            self._success_handlers.extend(self._normalise_callbacks(on_success))
        if on_error is not None:
            self._error_handlers.extend(self._normalise_callbacks(on_error))

    @staticmethod
    def _normalise_callbacks(
        callbacks: SuccessCallback | ErrorCallback | Iterable[Any],
    ) -> List[Any]:
        if callable(callbacks):
            return [callbacks]
        return [callback for callback in callbacks if callable(callback)]

    def compose_request(self) -> Dict[str, Any]:
        """Compose an order request from UI state and config defaults."""

        request = {
            "account_id": self._value_or_default(self.account_var),
            "symbol": self._value_or_default(self.symbol_var),
            "quantity": self._int_or_default(self.quantity_var),
            "order_type": self._value_or_default(self.order_type_var),
            "time_in_force": self._value_or_default(self.tif_var),
            "route": self._value_or_default(self.route_var),
            "limit_price": self._numeric_or_blank(self.limit_var),
            "stop_price": self._numeric_or_blank(self.stop_var),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.configs.setdefault("live", {})["last_request"] = request
        return request

    def submit_order(self) -> Dict[str, Any] | None:
        """Send the composed request to the live trading client."""

        payload = self.compose_request()
        if self.client is None:
            self._handle_error(payload, RuntimeError("Live client unavailable"))
            return None
        try:
            response = self.client.place_order(payload)
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_error(payload, exc)
            return None
        self._handle_success(payload, response)
        return response

    def refresh_metrics(self) -> Dict[str, Any]:
        """Refresh the metrics display using the configured fetcher."""

        if self.metrics_fetcher is not None:
            metrics = self.metrics_fetcher()
        else:
            metrics = dict(self.metrics_state)
        metrics.setdefault("orders_sent", self.metrics_state.get("orders_sent", 0))
        metrics.setdefault("errors", self.metrics_state.get("errors", 0))
        metrics["last_status"] = self.status_var.get()
        metrics["last_refresh"] = datetime.now(tz=timezone.utc).isoformat()
        self.metrics_state.update(metrics)
        self.metrics_output.delete("1.0", tk.END)
        self.metrics_output.insert("1.0", utils.json_dumps(metrics, indent=2))
        self.metrics_output.see("1.0")
        self.configs.setdefault("live", {})["metrics"] = dict(self.metrics_state)
        return metrics

    # ------------------------------------------------------------- Callbacks --
    def _handle_success(
        self, payload: Dict[str, Any], response: Dict[str, Any]
    ) -> None:
        self.metrics_state["orders_sent"] = self.metrics_state.get("orders_sent", 0) + 1
        status = response.get("status", "ACCEPTED")
        reference = response.get("id") or response.get("order_id") or "n/a"
        self.status_var.set(f"Order {status} · Ref {reference}")
        self.metrics_state["last_status"] = self.status_var.get()
        self.refresh_metrics()
        for callback in self._success_handlers:
            callback(payload, response)

    def _handle_error(self, payload: Dict[str, Any], error: Exception) -> None:
        self.metrics_state["errors"] = self.metrics_state.get("errors", 0) + 1
        self.status_var.set(f"Error: {error}")
        self.metrics_state["last_status"] = self.status_var.get()
        self.refresh_metrics()
        for callback in self._error_handlers:
            callback(payload, error)

    # ----------------------------------------------------------- UI helpers --
    def _value_or_default(self, variable: tk.StringVar) -> Any:
        value = variable.get().strip()
        if value:
            return value
        name = self._variable_name(variable)
        return self.request_defaults.get(name, "")

    def _int_or_default(self, variable: tk.StringVar) -> int:
        value = variable.get().strip()
        if value:
            try:
                return int(value)
            except ValueError:
                pass
        name = self._variable_name(variable)
        return int(self.request_defaults.get(name, 0) or 0)

    def _numeric_or_blank(self, variable: tk.StringVar) -> Any:
        value = variable.get().strip()
        if not value:
            name = self._variable_name(variable)
            return self.request_defaults.get(name, "")
        try:
            return float(value)
        except ValueError:
            return value

    def _variable_name(self, variable: tk.StringVar) -> str:
        mapping = {
            id(self.account_var): "account_id",
            id(self.symbol_var): "symbol",
            id(self.quantity_var): "quantity",
            id(self.order_type_var): "order_type",
            id(self.tif_var): "time_in_force",
            id(self.route_var): "route",
            id(self.limit_var): "limit_price",
            id(self.stop_var): "stop_price",
        }
        return mapping.get(id(variable), "")

    def _update_metrics_visibility(self) -> None:
        if self.metrics_visible.get():
            if not self.metrics_frame.winfo_manager():
                self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        else:
            if self.metrics_frame.winfo_manager():
                self.metrics_frame.pack_forget()

    def _style_text_widget(self, widget: tk.Text) -> None:
        widget.configure(**TEXT_WIDGET_DEFAULTS)


__all__ = ["LiveTab"]
