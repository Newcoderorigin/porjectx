import sys

import pytest

core_package = pytest.importorskip("toptek.core")
sys.modules.setdefault("core", core_package)

tk = pytest.importorskip("tkinter")

try:  # Prefer dedicated Live tab module when present
    from toptek.gui.live_tab import LiveTab  # type: ignore
except ModuleNotFoundError:
    try:
        from toptek.gui.widgets import LiveTab  # type: ignore
    except (ModuleNotFoundError, ImportError, AttributeError):
        LiveTab = None  # type: ignore

if LiveTab is None:  # pragma: no cover - legacy builds without the Live tab
    pytest.skip("Live tab implementation unavailable", allow_module_level=True)


def test_live_tab_placeholder() -> None:  # pragma: no cover - executed when LiveTab exists
    pytest.skip("Live tab behaviour tests require the implementation module")

