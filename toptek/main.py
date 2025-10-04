"""Entry point for the Toptek application."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from dotenv import load_dotenv

from core import backtest, data, features, model, risk, utils


ROOT = Path(__file__).parent


def load_configs() -> Dict[str, Dict[str, object]]:
    """Load configuration files into a dictionary."""

    app_cfg = utils.load_yaml(ROOT / "config" / "app.yml")
    risk_cfg = utils.load_yaml(ROOT / "config" / "risk.yml")
    feature_cfg = utils.load_yaml(ROOT / "config" / "features.yml")
    return {"app": app_cfg, "risk": risk_cfg, "features": feature_cfg}


def run_cli(args: argparse.Namespace, configs: Dict[str, Dict[str, object]], paths: utils.AppPaths) -> None:
    """Dispatch CLI commands based on ``args``."""

    logger = utils.build_logger("toptek")
    risk_profile = risk.RiskProfile(
        max_position_size=configs["risk"].get("max_position_size", 1),
        max_daily_loss=configs["risk"].get("max_daily_loss", 1000),
        restricted_hours=configs["risk"].get("restricted_trading_hours", []),
        atr_multiplier_stop=configs["risk"].get("atr_multiplier_stop", 2.0),
        cooldown_losses=configs["risk"].get("cooldown_losses", 2),
        cooldown_minutes=configs["risk"].get("cooldown_minutes", 30),
    )

    df = data.sample_dataframe()
    feat_map = features.compute_features(df)
    X = np.column_stack(list(feat_map.values()))
    y = (np.diff(df["close"], prepend=df["close"].iloc[0]) > 0).astype(int)

    if args.cli == "train":
        try:
            result = model.train_classifier(X, y, model_type=args.model, models_dir=paths.models)
        except ValueError as exc:
            logger.error("Training failed: %s", exc)
            return
        preprocess = result.preprocessing or {}
        if preprocess:
            logger.info(
                "Preprocessing summary: imputed=%s dropped_rows=%s dropped_columns=%s",
                preprocess.get("imputed_cells", 0),
                preprocess.get("dropped_rows", 0),
                preprocess.get("dropped_columns", 0),
            )
        if result.retained_columns is not None and result.original_feature_count is not None:
            logger.info(
                "Retained %s of %s feature columns after cleaning",
                len(result.retained_columns),
                result.original_feature_count,
            )
        logger.info("Training complete: metrics=%s threshold=%.2f", result.metrics, result.threshold)
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
        atr = feat_map["atr_14"][-1]
        size = risk.position_size(account_balance=50000, risk_profile=risk_profile, atr=atr, tick_value=12.5)
        logger.info("Suggested paper size: %s contracts", size)
    else:
        logger.error("Unknown CLI command: %s", args.cli)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Toptek manual trading toolkit")
    parser.add_argument("--cli", choices=["train", "backtest", "paper"], help="Run in CLI mode instead of GUI")
    parser.add_argument("--symbol", default="ESZ5", help="Futures symbol")
    parser.add_argument("--timeframe", default="5m", help="Bar timeframe")
    parser.add_argument("--lookback", default="90d", help="Lookback period for CLI commands")
    parser.add_argument("--model", default="logistic", choices=["logistic", "gbm"], help="Model type for training")
    parser.add_argument("--start", help="Start date for backtest")
    return parser.parse_args()


def main() -> None:
    """Program entry point."""

    load_dotenv(ROOT / ".env")
    configs = load_configs()
    paths = utils.build_paths(ROOT, configs["app"])
    utils.ensure_directories(paths)

    args = parse_args()
    if args.cli:
        run_cli(args, configs, paths)
        return

    from gui.app import launch_app  # imported lazily to avoid Tkinter cost

    launch_app(configs=configs, paths=paths)


if __name__ == "__main__":
    main()
