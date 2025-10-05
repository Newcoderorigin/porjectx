"""UI configuration parsing utilities with environment overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from . import utils


def _coerce_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")
    return str(value)


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean or boolean-like string")


def _coerce_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be an integer") from exc
    if minimum is not None and coerced < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return coerced


@dataclass(frozen=True)
class ShellSettings:
    """Configuration for CLI shell defaults."""

    symbol: str = "ES=F"
    interval: str = "5m"
    research_bars: int = 240
    lookback_bars: int = 480
    calibrate: bool = True
    model: str = "logistic"
    simulation_bars: int = 720
    playbook: str = "momentum"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ShellSettings":
        calibrate_raw = data.get("calibrate", cls.calibrate)
        if isinstance(calibrate_raw, str):
            calibrate_value = _coerce_bool(calibrate_raw, "shell.calibrate")
        elif isinstance(calibrate_raw, bool):
            calibrate_value = calibrate_raw
        elif calibrate_raw is None:
            calibrate_value = cls.calibrate
        else:
            calibrate_value = bool(calibrate_raw)
        return cls(
            symbol=_coerce_str(data.get("symbol", cls.symbol), "shell.symbol"),
            interval=_coerce_str(data.get("interval", cls.interval), "shell.interval"),
            research_bars=_coerce_int(
                data.get("research_bars", cls.research_bars),
                "shell.research_bars",
                minimum=60,
            ),
            lookback_bars=_coerce_int(
                data.get("lookback_bars", cls.lookback_bars),
                "shell.lookback_bars",
                minimum=120,
            ),
            calibrate=calibrate_value,
            model=_coerce_str(data.get("model", cls.model), "shell.model"),
            simulation_bars=_coerce_int(
                data.get("simulation_bars", cls.simulation_bars),
                "shell.simulation_bars",
                minimum=120,
            ),
            playbook=_coerce_str(data.get("playbook", cls.playbook), "shell.playbook"),
        )

    def apply_environment(self, env: Mapping[str, str]) -> "ShellSettings":
        updates: Dict[str, Any] = {}
        if env.get("TOPTEK_UI_SYMBOL"):
            updates["symbol"] = env["TOPTEK_UI_SYMBOL"]
        if env.get("TOPTEK_UI_INTERVAL"):
            updates["interval"] = env["TOPTEK_UI_INTERVAL"]
        if env.get("TOPTEK_UI_RESEARCH_BARS"):
            updates["research_bars"] = _coerce_int(
                env["TOPTEK_UI_RESEARCH_BARS"],
                "env.TOPTEK_UI_RESEARCH_BARS",
                minimum=60,
            )
        if env.get("TOPTEK_UI_LOOKBACK_BARS"):
            updates["lookback_bars"] = _coerce_int(
                env["TOPTEK_UI_LOOKBACK_BARS"],
                "env.TOPTEK_UI_LOOKBACK_BARS",
                minimum=120,
            )
        if env.get("TOPTEK_UI_CALIBRATE"):
            updates["calibrate"] = _coerce_bool(
                env["TOPTEK_UI_CALIBRATE"], "env.TOPTEK_UI_CALIBRATE"
            )
        if env.get("TOPTEK_UI_MODEL"):
            updates["model"] = env["TOPTEK_UI_MODEL"]
        if env.get("TOPTEK_UI_SIMULATION_BARS"):
            updates["simulation_bars"] = _coerce_int(
                env["TOPTEK_UI_SIMULATION_BARS"],
                "env.TOPTEK_UI_SIMULATION_BARS",
                minimum=120,
            )
        if env.get("TOPTEK_UI_PLAYBOOK"):
            updates["playbook"] = env["TOPTEK_UI_PLAYBOOK"]
        return replace(self, **updates) if updates else self


@dataclass(frozen=True)
class ChartSettings:
    """Chart refresh and theming parameters."""

    fps: int = 12
    max_points: int = 180
    price_decimals: int = 2

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChartSettings":
        return cls(
            fps=_coerce_int(data.get("fps", cls.fps), "chart.fps", minimum=1),
            max_points=_coerce_int(
                data.get("max_points", cls.max_points), "chart.max_points", minimum=10
            ),
            price_decimals=_coerce_int(
                data.get("price_decimals", cls.price_decimals),
                "chart.price_decimals",
                minimum=0,
            ),
        )

    def apply_environment(self, env: Mapping[str, str]) -> "ChartSettings":
        updates: Dict[str, Any] = {}
        if env.get("TOPTEK_UI_FPS"):
            updates["fps"] = _coerce_int(
                env["TOPTEK_UI_FPS"], "env.TOPTEK_UI_FPS", minimum=1
            )
        if env.get("TOPTEK_UI_CHART_POINTS"):
            updates["max_points"] = _coerce_int(
                env["TOPTEK_UI_CHART_POINTS"], "env.TOPTEK_UI_CHART_POINTS", minimum=10
            )
        return replace(self, **updates) if updates else self


@dataclass(frozen=True)
class AppearanceSettings:
    """High-level UI theming choices."""

    theme: str = "dark"
    accent: str = "violet"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppearanceSettings":
        return cls(
            theme=_coerce_str(data.get("theme", cls.theme), "appearance.theme"),
            accent=_coerce_str(data.get("accent", cls.accent), "appearance.accent"),
        )

    def apply_environment(self, env: Mapping[str, str]) -> "AppearanceSettings":
        updates: Dict[str, Any] = {}
        if env.get("TOPTEK_UI_THEME"):
            updates["theme"] = env["TOPTEK_UI_THEME"]
        if env.get("TOPTEK_UI_ACCENT"):
            updates["accent"] = env["TOPTEK_UI_ACCENT"]
        return replace(self, **updates) if updates else self


@dataclass(frozen=True)
class TradingViewSettings:
    """Defaults for TradingView launch parameters."""

    symbol: str = "ES=F"
    interval: str = "5m"
    theme: str = "dark"
    locale: str = "en"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TradingViewSettings":
        return cls(
            symbol=_coerce_str(data.get("symbol", cls.symbol), "tradingview.symbol"),
            interval=_coerce_str(data.get("interval", cls.interval), "tradingview.interval"),
            theme=_coerce_str(data.get("theme", cls.theme), "tradingview.theme"),
            locale=_coerce_str(data.get("locale", cls.locale), "tradingview.locale"),
        )

    def apply_environment(self, env: Mapping[str, str]) -> "TradingViewSettings":
        updates: Dict[str, Any] = {}
        if env.get("TOPTEK_TV_SYMBOL"):
            updates["symbol"] = env["TOPTEK_TV_SYMBOL"]
        if env.get("TOPTEK_TV_INTERVAL"):
            updates["interval"] = env["TOPTEK_TV_INTERVAL"]
        if env.get("TOPTEK_TV_THEME"):
            updates["theme"] = env["TOPTEK_TV_THEME"]
        if env.get("TOPTEK_TV_LOCALE"):
            updates["locale"] = env["TOPTEK_TV_LOCALE"]
        return replace(self, **updates) if updates else self


@dataclass(frozen=True)
class LoginStatus:
    idle: str = "Awaiting verification"
    saved: str = "Saved. Run verification to confirm access."
    verified: str = "All keys present. Proceed to Research ▶"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoginStatus":
        return cls(
            idle=_coerce_str(data.get("idle", cls.idle), "status.login.idle"),
            saved=_coerce_str(data.get("saved", cls.saved), "status.login.saved"),
            verified=_coerce_str(
                data.get("verified", cls.verified), "status.login.verified"
            ),
        )


@dataclass(frozen=True)
class TrainingStatus:
    idle: str = "Awaiting training run"
    success: str = "Model artefact refreshed. Continue to Backtest ▶"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainingStatus":
        return cls(
            idle=_coerce_str(data.get("idle", cls.idle), "status.training.idle"),
            success=_coerce_str(
                data.get("success", cls.success), "status.training.success"
            ),
        )


@dataclass(frozen=True)
class BacktestStatus:
    idle: str = "No simulations yet"
    success: str = "Sim complete. If expectancy holds, draft a manual trade plan ▶"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BacktestStatus":
        return cls(
            idle=_coerce_str(data.get("idle", cls.idle), "status.backtest.idle"),
            success=_coerce_str(
                data.get("success", cls.success), "status.backtest.success"
            ),
        )


@dataclass(frozen=True)
class GuardStatus:
    pending: str = "Topstep Guard: pending review"
    intro: str = "Manual execution only. Awaiting guard refresh..."
    defensive_warning: str = (
        "DEFENSIVE_MODE active. Stand down and review your journal before trading."
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GuardStatus":
        return cls(
            pending=_coerce_str(
                data.get("pending", cls.pending), "status.guard.pending"
            ),
            intro=_coerce_str(data.get("intro", cls.intro), "status.guard.intro"),
            defensive_warning=_coerce_str(
                data.get("defensive_warning", cls.defensive_warning),
                "status.guard.defensive_warning",
            ),
        )


@dataclass(frozen=True)
class ReplayStatus:
    idle: str = "Load a dataset to begin playback."
    buffering: str = "Preparing replay dataset..."
    playing: str = "Streaming simulator feed."
    paused: str = "Replay paused."
    complete: str = "Reached end of recording."

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReplayStatus":
        return cls(
            idle=_coerce_str(data.get("idle", cls.idle), "status.replay.idle"),
            buffering=_coerce_str(
                data.get("buffering", cls.buffering), "status.replay.buffering"
            ),
            playing=_coerce_str(
                data.get("playing", cls.playing), "status.replay.playing"
            ),
            paused=_coerce_str(data.get("paused", cls.paused), "status.replay.paused"),
            complete=_coerce_str(
                data.get("complete", cls.complete), "status.replay.complete"
            ),
        )


@dataclass(frozen=True)
class StatusMessages:
    login: LoginStatus = field(default_factory=LoginStatus)
    training: TrainingStatus = field(default_factory=TrainingStatus)
    backtest: BacktestStatus = field(default_factory=BacktestStatus)
    guard: GuardStatus = field(default_factory=GuardStatus)
    replay: ReplayStatus = field(default_factory=ReplayStatus)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StatusMessages":
        return cls(
            login=LoginStatus.from_mapping(data.get("login", {})),
            training=TrainingStatus.from_mapping(data.get("training", {})),
            backtest=BacktestStatus.from_mapping(data.get("backtest", {})),
            guard=GuardStatus.from_mapping(data.get("guard", {})),
            replay=ReplayStatus.from_mapping(data.get("replay", {})),
        )


@dataclass(frozen=True)
class UIConfig:
    """Top-level structure for UI settings."""

    appearance: AppearanceSettings = field(default_factory=AppearanceSettings)
    shell: ShellSettings = field(default_factory=ShellSettings)
    chart: ChartSettings = field(default_factory=ChartSettings)
    tradingview: TradingViewSettings = field(default_factory=TradingViewSettings)
    status: StatusMessages = field(default_factory=StatusMessages)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UIConfig":
        return cls(
            appearance=AppearanceSettings.from_mapping(data.get("appearance", {})),
            shell=ShellSettings.from_mapping(data.get("shell", {})),
            chart=ChartSettings.from_mapping(data.get("chart", {})),
            tradingview=TradingViewSettings.from_mapping(data.get("tradingview", {})),
            status=StatusMessages.from_mapping(data.get("status", {})),
        )

    def apply_environment(self, env: Mapping[str, str]) -> "UIConfig":
        return replace(
            self,
            appearance=self.appearance.apply_environment(env),
            shell=self.shell.apply_environment(env),
            chart=self.chart.apply_environment(env),
            tradingview=self.tradingview.apply_environment(env),
        )

    def with_updates(
        self,
        *,
        appearance: Dict[str, Any] | None = None,
        shell: Dict[str, Any] | None = None,
        chart: Dict[str, Any] | None = None,
        tradingview: Dict[str, Any] | None = None,
    ) -> "UIConfig":
        """Return a copy of the config with provided section overrides."""

        updates: Dict[str, Any] = {}
        if appearance:
            updates["appearance"] = replace(self.appearance, **appearance)
        if shell:
            updates["shell"] = replace(self.shell, **shell)
        if chart:
            updates["chart"] = replace(self.chart, **chart)
        if tradingview:
            updates["tradingview"] = replace(self.tradingview, **tradingview)
        return replace(self, **updates) if updates else self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "appearance": asdict(self.appearance),
            "shell": asdict(self.shell),
            "chart": asdict(self.chart),
            "tradingview": asdict(self.tradingview),
            "status": asdict(self.status),
        }


def load_ui_config(path: Path, *, env: Mapping[str, str] | None = None) -> UIConfig:
    """Load :class:`UIConfig` from *path* applying environment overrides."""

    env_mapping = os.environ if env is None else env
    data = utils.load_yaml(path)
    config = UIConfig.from_mapping(data)
    return config.apply_environment(env_mapping)


__all__ = [
    "AppearanceSettings",
    "ShellSettings",
    "ChartSettings",
    "ReplayStatus",
    "StatusMessages",
    "UIConfig",
    "load_ui_config",
]
