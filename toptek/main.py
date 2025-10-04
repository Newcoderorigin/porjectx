"""Entry point for the Toptek application."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv

from core import backtest, data, model, risk, ui_config, utils
from toptek.features import build_features


ROOT = Path(__file__).parent


def load_configs() -> Tuple[Dict[str, Dict[str, object]], ui_config.UIConfig]:
    """Load configuration files along with UI defaults."""

    app_cfg = utils.load_yaml(ROOT / "config" / "app.yml")
    risk_cfg = utils.load_yaml(ROOT / "config" / "risk.yml")
    feature_cfg = utils.load_yaml(ROOT / "config" / "features.yml")
    ui_path = ROOT.parent / "configs" / "ui.yml"
    ui_cfg = ui_config.load_ui_config(ui_path)
    return (
        {
            "app": app_cfg,
            "risk": risk_cfg,
            "features": feature_cfg,
            "ui": ui_cfg.as_dict(),
        },
        ui_cfg,
    )


def run_cli(
    args: argparse.Namespace,
    configs: Dict[str, Dict[str, object]],
    paths: utils.AppPaths,
) -> None:
    """Dispatch CLI commands based on ``args``."""

    import numpy as np

    logger = utils.build_logger("toptek")
    logger.info(
        "CLI mode=%s symbol=%s timeframe=%s lookback=%s fps=%s",
        args.cli,
        args.symbol,
        args.timeframe,
        args.lookback,
        getattr(args, "fps", None),
    )
    risk_profile = risk.RiskProfile(
        max_position_size=configs["risk"].get("max_position_size", 1),
        max_daily_loss=configs["risk"].get("max_daily_loss", 1000),
        restricted_hours=configs["risk"].get("restricted_trading_hours", []),
        atr_multiplier_stop=configs["risk"].get("atr_multiplier_stop", 2.0),
        cooldown_losses=configs["risk"].get("cooldown_losses", 2),
        cooldown_minutes=configs["risk"].get("cooldown_minutes", 30),
    )

    lookback = int(args.lookback)
    df = data.sample_dataframe(lookback)
    try:
        bundle = build_features(df, cache_dir=paths.cache)
    except ValueError as exc:
        logger.error("Feature pipeline failed: %s", exc)
        return

    X = bundle.X
    y = bundle.y

    dropped = int(bundle.meta.get("dropped_rows", 0))
    if dropped:
        logger.warning("Feature pipeline dropped %s rows prior to training", dropped)

    logger.debug("Feature bundle meta: %s", bundle.meta)

    if args.cli == "train":
        if np.unique(y).size < 2:
            logger.error(
                "Training aborted: dataset lacks class diversity after cleaning"
            )
            return
        try:
            result = model.train_classifier(
                X, y, model_type=args.model, models_dir=paths.models
            )
        except ValueError as exc:
            logger.error("Training failed: %s", exc)
            return
        logger.info(
            "Training complete: metrics=%s threshold=%.2f",
            result.metrics,
            result.threshold,
        )
    elif args.cli == "backtest":
        returns = np.log(df["close"]).diff().fillna(0).to_numpy()
        signals = (returns > 0).astype(int)
        bt = backtest.run_backtest(returns, signals)
        logger.info(
            "Backtest: hit_rate=%.2f sharpe=%.2f maxDD=%.2f expectancy=%.4f",
            bt.hit_rate,
            bt.sharpe,
            bt.max_drawdown,
            bt.expectancy,
        )
    elif args.cli == "paper":
        feature_names = bundle.meta.get("feature_names", [])
        atr_value: float | None = None
        if "atr_14" in feature_names and bundle.X.size:
            atr_index = feature_names.index("atr_14")
            atr_series = bundle.X[:, atr_index]
            if atr_series.size:
                atr_value = float(atr_series[-1])
        if atr_value is None:
            logger.warning(
                "ATR14 feature unavailable from bundle; unable to size position"
            )
            return
        size = risk.position_size(
            account_balance=50000,
            risk_profile=risk_profile,
            atr=atr_value,
            tick_value=12.5,
        )
        logger.info("Suggested paper size: %s contracts", size)
    else:
        logger.error("Unknown CLI command: %s", args.cli)


def parse_args(settings: ui_config.UIConfig) -> argparse.Namespace:
    """Parse command-line arguments with defaults sourced from ``settings``."""

    parser = argparse.ArgumentParser(description="Toptek manual trading toolkit")
    parser.add_argument(
        "--cli",
        choices=["train", "backtest", "paper"],
        help="Run in CLI mode instead of GUI",
    )
    parser.add_argument(
        "--symbol",
        help=f"Futures symbol (default: {settings.shell.symbol})",
    )
    parser.add_argument(
        "--timeframe",
        help=f"Bar timeframe (default: {settings.shell.interval})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        help=f"Synthetic bar count for CLI flows (default: {settings.shell.lookback_bars})",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "gbm"],
        help=f"Model type for training (default: {settings.shell.model})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help=f"Live chart frames per second (default: {settings.chart.fps})",
    )
    parser.add_argument("--start", help="Start date for backtest")
    return parser.parse_args()


def _apply_cli_overrides(
    settings: ui_config.UIConfig,
    *,
    symbol: str | None,
    timeframe: str | None,
    lookback: int | None,
    model_name: str | None,
    fps: int | None,
) -> ui_config.UIConfig:
    shell_updates: Dict[str, object] = {}
    chart_updates: Dict[str, object] = {}
    if symbol is not None:
        shell_updates["symbol"] = symbol
    if timeframe is not None:
        shell_updates["interval"] = timeframe
    if lookback is not None:
        shell_updates["lookback_bars"] = lookback
    if model_name is not None:
        shell_updates["model"] = model_name
    if fps is not None:
        chart_updates["fps"] = fps
    if not shell_updates and not chart_updates:
        return settings
    return settings.with_updates(
        shell=shell_updates if shell_updates else None,
        chart=chart_updates if chart_updates else None,
    )


def main() -> None:
    """Program entry point."""

    load_dotenv(ROOT / ".env")
    configs, ui_settings = load_configs()
    app_config = configs.get("app", {})
    paths = utils.build_paths(ROOT, app_config)
    utils.ensure_directories(paths)

    log_level = str(app_config.get("log_level", "INFO"))
    utils.configure_logging(paths.logs, level=log_level)
    stack_versions = utils.assert_numeric_stack(reports_dir=paths.reports)
    utils.set_seeds(42)

    args = parse_args(ui_settings)
    raw_symbol = args.symbol
    raw_timeframe = args.timeframe
    raw_lookback = args.lookback
    raw_model = args.model
    raw_fps = args.fps
    args.symbol = raw_symbol or ui_settings.shell.symbol
    args.timeframe = raw_timeframe or ui_settings.shell.interval
    args.lookback = raw_lookback or ui_settings.shell.lookback_bars
    args.model = raw_model or ui_settings.shell.model
    args.fps = raw_fps or ui_settings.chart.fps
    ui_settings = _apply_cli_overrides(
        ui_settings,
        symbol=raw_symbol,
        timeframe=raw_timeframe,
        lookback=raw_lookback,
        model_name=raw_model,
        fps=raw_fps,
    )
    configs["ui"] = ui_settings.as_dict()

    logger = utils.build_logger("toptek", level=log_level)
    logger.info(
        "Numeric stack verified: numpy=%s scipy=%s sklearn=%s",
        stack_versions["numpy"],
        stack_versions["scipy"],
        stack_versions["sklearn"],
    )
    if args.cli:
        run_cli(args, configs, paths)
        return

    from gui.app import launch_app  # imported lazily to avoid Tkinter cost

    launch_app(configs=configs, paths=paths)


if __name__ == "__main__":
    main()
