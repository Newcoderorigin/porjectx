"""Behavioural wiring tests for the Live tab widget."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

core_package = pytest.importorskip("toptek.core")
sys.modules.setdefault("core", core_package)

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

from toptek.core import utils  # noqa: E402

try:  # Prefer dedicated Live tab module when present
    from toptek.gui.live_tab import LiveTab  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - legacy builds without the Live tab
    from toptek.gui.widgets import LiveTab  # type: ignore  # noqa: F401


@pytest.fixture
def tk_root() -> Any:
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on CI environment
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def _paths(base: Path) -> utils.AppPaths:
    return utils.AppPaths(root=base, cache=base / "cache", models=base / "models")


def _build_tab(
    root: Any,
    tmp_path: Path,
    configs: Dict[str, Dict[str, object]],
    **kwargs: Any,
) -> LiveTab:
    notebook = ttk.Notebook(root)
    notebook.pack()
    return LiveTab(notebook, configs, _paths(tmp_path), **kwargs)


def test_live_tab_metrics_visibility_toggle(tk_root: Any, tmp_path: Path) -> None:
    configs: Dict[str, Dict[str, object]] = {"live": {"defaults": {"symbol": "ES"}}}
    tab = _build_tab(tk_root, tmp_path, configs)

    assert tab.metrics_frame.winfo_manager()

    tab.metrics_visible.set(False)
    tab._update_metrics_visibility()
    assert not tab.metrics_frame.winfo_manager()

    tab.metrics_visible.set(True)
    tab._update_metrics_visibility()
    assert tab.metrics_frame.winfo_manager()


def test_live_tab_compose_request_uses_config_defaults(
    tk_root: Any, tmp_path: Path
) -> None:
    configs: Dict[str, Dict[str, object]] = {
        "live": {
            "defaults": {
                "account_id": "ACC-1",
                "symbol": "MESU4",
                "quantity": 3,
                "order_type": "LIMIT",
                "time_in_force": "GTC",
                "route": "LIVE",
                "limit_price": 4321.0,
                "stop_price": "",
            }
        }
    }
    tab = _build_tab(tk_root, tmp_path, configs)

    tab.account_var.set("")
    tab.symbol_var.set("")
    tab.quantity_var.set("")
    tab.order_type_var.set("")
    tab.tif_var.set("")
    tab.route_var.set("")
    tab.limit_var.set("")
    tab.stop_var.set("")

    request = tab.compose_request()

    assert request["account_id"] == "ACC-1"
    assert request["symbol"] == "MESU4"
    assert request["quantity"] == 3
    assert request["order_type"] == "LIMIT"
    assert request["time_in_force"] == "GTC"
    assert request["route"] == "LIVE"
    assert request["limit_price"] == 4321.0
    assert request["stop_price"] == ""
    assert configs["live"]["last_request"] == request
    redacted = configs["live"].get("last_request_redacted")
    assert redacted
    assert redacted["symbol"] == "[REDACTED_TICKER]"
    assert redacted["account_id"] == "[REDACTED_TICKER]"
    assert tab.request_defaults["symbol"] == "MESU4"


def test_live_tab_submit_order_handles_success_and_error(
    tk_root: Any, tmp_path: Path
) -> None:
    success_events: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    error_events: List[Tuple[Dict[str, Any], Exception]] = []

    class RecordingClient:
        def __init__(self, response: Any) -> None:
            self.response = response
            self.calls: List[Dict[str, Any]] = []

        def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            self.calls.append(payload)
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    configs: Dict[str, Dict[str, object]] = {"live": {}}
    client = RecordingClient({"status": "ACCEPTED", "id": "123"})
    tab = _build_tab(tk_root, tmp_path, configs, client=client)
    tab.register_callbacks(
        on_success=lambda payload, response: success_events.append((payload, response)),
        on_error=lambda payload, exc: error_events.append((payload, exc)),
    )

    response = tab.submit_order()
    assert response == {"status": "ACCEPTED", "id": "123"}
    assert len(client.calls) == 1
    assert success_events and success_events[0][1]["id"] == "123"
    assert not error_events
    assert tab.metrics_state["orders_sent"] == 1
    assert tab.status_var.get().startswith("Order ACCEPTED")

    failing = RecordingClient(ValueError("route unavailable"))
    tab.client = failing
    tab.submit_order()
    assert error_events and isinstance(error_events[-1][1], ValueError)
    assert tab.metrics_state["errors"] == 1
    assert tab.status_var.get().startswith("Error:")
    metrics = configs["live"].get("metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("orders_sent") == 1
    assert metrics.get("errors") == 1


def test_live_tab_refresh_metrics_uses_fetcher(
    tk_root: Any, tmp_path: Path
) -> None:
    calls: List[int] = []

    def metrics_fetcher() -> Dict[str, Any]:
        calls.append(1)
        return {"latency_ms": 42, "fills": 5}

    configs: Dict[str, Dict[str, object]] = {"live": {}}
    tab = _build_tab(tk_root, tmp_path, configs, metrics_fetcher=metrics_fetcher)
    tab.metrics_state["orders_sent"] = 7
    metrics = tab.refresh_metrics()

    assert calls  # fetcher invoked
    assert metrics["latency_ms"] == 42
    assert metrics["fills"] == 5
    buffer = tab.metrics_output.get("1.0", "end-1c")
    assert '"latency_ms": 42' in buffer
    assert configs["live"]["metrics"]["fills"] == 5
