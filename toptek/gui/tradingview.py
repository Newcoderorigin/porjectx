"""Helpers for launching TradingView charts with project defaults."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from urllib.parse import quote_plus

__all__ = ["TradingViewDefaults", "TradingViewRouter"]


@dataclass(frozen=True)
class TradingViewDefaults:
    """Container describing default TradingView parameters."""

    symbol: str
    interval: str
    theme: str
    locale: str


class TradingViewRouter:
    """Derive TradingView launch URLs from config and environment overrides."""

    _BASE_URL = "https://www.tradingview.com/chart/"

    def __init__(
        self,
        app_config: Mapping[str, Any] | None = None,
        ui_config: Mapping[str, Any] | None = None,
    ) -> None:
        config_block = {}
        if app_config:
            tv_block = app_config.get("tv")
            if isinstance(tv_block, Mapping):
                config_block = dict(tv_block)
            else:
                config_block = dict(app_config)
        ui_block = {}
        if ui_config:
            tv_ui = ui_config.get("tradingview")
            if isinstance(tv_ui, Mapping):
                ui_block = dict(tv_ui)

        self.enabled = self._coerce_bool(config_block.get("enabled", False))
        self._symbol = self._coerce_str(
            config_block.get("symbol"),
            fallback=ui_block.get("symbol", "ES=F"),
        )
        self._interval = self._coerce_str(
            config_block.get("interval"),
            fallback=ui_block.get("interval", "5m"),
        )
        self._theme = self._normalise_theme(
            self._coerce_str(config_block.get("theme"), fallback=ui_block.get("theme", "dark"))
        )
        self._locale = self._coerce_str(
            config_block.get("locale"),
            fallback=ui_block.get("locale", "en"),
        )
        tabs = config_block.get("tabs", {})
        self._tabs: MutableMapping[str, bool] = {}
        if isinstance(tabs, Mapping):
            for key, value in tabs.items():
                self._tabs[str(key).lower()] = self._coerce_bool(value, default=True)
        self._favorites = self._sanitise_favorites(config_block.get("favorites", []))

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _coerce_str(value: Any, *, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback

    @staticmethod
    def _normalise_theme(value: str) -> str:
        lowered = value.lower()
        return "light" if lowered == "light" else "dark"

    @staticmethod
    def _sanitise_favorites(raw: Any) -> list[dict[str, str]]:
        favourites: list[dict[str, str]] = []
        if not isinstance(raw, Sequence):
            return favourites
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            symbol = str(entry.get("symbol", "")).strip()
            interval = str(entry.get("interval", "")).strip()
            if not symbol or not interval:
                continue
            label = str(entry.get("label", "")).strip() or f"{symbol} · {interval}"
            favourites.append({"symbol": symbol, "interval": interval, "label": label})
        return favourites

    def defaults(self) -> TradingViewDefaults:
        """Return the current default TradingView launch parameters."""

        return TradingViewDefaults(
            symbol=self._symbol,
            interval=self._interval,
            theme=self._theme,
            locale=self._locale,
        )

    def is_tab_enabled(self, name: str) -> bool:
        """Return ``True`` when the requested tab integration is enabled."""

        if not self.enabled:
            return False
        if not self._tabs:
            return True
        return self._tabs.get(name.lower(), False)

    @property
    def favorites(self) -> list[dict[str, str]]:
        """Structured watchlist entries configured in ``config/app.yml``."""

        return list(self._favorites)

    def build_url(
        self,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        theme: str | None = None,
        locale: str | None = None,
    ) -> str:
        """Compose a TradingView chart URL using the provided parameters."""

        defaults = self.defaults()
        resolved_symbol = symbol.strip() if symbol else defaults.symbol
        resolved_interval = interval.strip() if interval else defaults.interval
        resolved_theme = self._normalise_theme(theme.strip() if theme else defaults.theme)
        resolved_locale = (locale.strip() if locale else defaults.locale) or "en"
        interval_code = self._normalise_interval(resolved_interval)

        params = {
            "symbol": quote_plus(resolved_symbol),
            "interval": quote_plus(interval_code),
            "theme": quote_plus(resolved_theme),
            "locale": quote_plus(resolved_locale),
        }
        query = "&".join(f"{key}={value}" for key, value in params.items())
        return f"{self._BASE_URL}?{query}"

    def launch(
        self,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        theme: str | None = None,
        locale: str | None = None,
        opener: Callable[[str], None] | None = None,
    ) -> str:
        """Open TradingView in the default browser and return the URL used."""

        if not self.enabled:
            raise RuntimeError("TradingView integration disabled in config")
        url = self.build_url(symbol=symbol, interval=interval, theme=theme, locale=locale)
        action = opener or webbrowser.open_new
        action(url)
        return url

    @staticmethod
    def _normalise_interval(raw: str) -> str:
        value = raw.strip()
        lowered = value.lower()
        if lowered.isdigit():
            return lowered
        if lowered.endswith("m") and lowered[:-1].isdigit():
            return lowered[:-1]
        if lowered.endswith("h") and lowered[:-1].isdigit():
            try:
                minutes = int(lowered[:-1]) * 60
                return str(minutes)
            except ValueError:
                return value
        if lowered.endswith("d"):
            return "D"
        if lowered.endswith("w"):
            return "W"
        if lowered.endswith("mo") or lowered.endswith("mth"):
            return "M"
        return value
