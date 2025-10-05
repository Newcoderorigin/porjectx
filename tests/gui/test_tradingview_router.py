from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Dict

sys.modules.setdefault(
    "dotenv", SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None)
)

stub_ai_server = ModuleType("toptek.ai_server")
stub_ai_server.__path__ = []  # type: ignore[attr-defined]
stub_fastapi_stub = ModuleType("toptek.ai_server._fastapi_stub")
stub_fastapi_stub.HTTPException = Exception  # type: ignore[attr-defined]
stub_fastapi_stub.Request = object  # type: ignore[attr-defined]
stub_fastapi_stub.WebSocket = object  # type: ignore[attr-defined]
sys.modules.setdefault("toptek.ai_server", stub_ai_server)
sys.modules.setdefault("toptek.ai_server._fastapi_stub", stub_fastapi_stub)

from toptek import main
from toptek.gui.tradingview import TradingViewRouter


def test_tradingview_router_respects_env_overrides() -> None:
    env: Dict[str, str] = {
        "TOPTEK_TV_ENABLED": "1",
        "TOPTEK_TV_SYMBOL": "NQ=F",
        "TOPTEK_TV_INTERVAL": "15m",
        "TOPTEK_TV_THEME": "light",
        "TOPTEK_TV_LOCALE": "en",
        "TOPTEK_TV_TAB_NOTEBOOK": "1",
        "TOPTEK_TV_TAB_RESEARCH": "0",
        "TOPTEK_TV_TAB_TRADE": "1",
    }
    configs, _ui = main.load_configs(env=env)
    router = TradingViewRouter(configs["app"], configs["ui"])

    assert router.enabled is True
    defaults = router.defaults()
    assert defaults.symbol == "NQ=F"
    assert defaults.interval == "15m"
    assert defaults.theme == "light"
    assert defaults.locale == "en"
    assert router.is_tab_enabled("notebook")
    assert not router.is_tab_enabled("research")
    assert router.is_tab_enabled("trade")

    launched: list[str] = []
    url = router.launch(opener=launched.append)
    assert launched and launched[0] == url
    assert "symbol=NQ%3DF" in url
    assert "interval=15" in url
