"""UI hooks for logging trades and predictions into the data layer."""

from __future__ import annotations

import json
from typing import Any, Dict

from toptek.data import io


def log_trade(**kwargs: Any) -> None:
    """Persist a trade event to the SQLite store."""

    conn = io.connect()
    try:
        io.run_migrations(conn)
        payload = {
            "trade_id": kwargs.get("trade_id"),
            "session_id": kwargs.get("session_id"),
            "symbol": kwargs.get("symbol"),
            "side": kwargs.get("side"),
            "qty": kwargs.get("qty"),
            "entry_ts": kwargs.get("entry_ts"),
            "exit_ts": kwargs.get("exit_ts"),
            "entry_px": kwargs.get("entry_px"),
            "exit_px": kwargs.get("exit_px"),
            "pnl": kwargs.get("pnl"),
            "label_hit": kwargs.get("label_hit"),
            "label_return": kwargs.get("label_return"),
            "meta": json.dumps(kwargs.get("meta", {})),
        }
        conn.execute(
            (
                "INSERT OR REPLACE INTO trades(trade_id, session_id, symbol, side, qty, entry_ts, exit_ts, entry_px, exit_px, pnl, "
                "label_hit, label_return, meta) VALUES (:trade_id, :session_id, :symbol, :side, :qty, :entry_ts, :exit_ts, :entry_px, :exit_px, :pnl, :label_hit, :label_return, :meta)"
            ),
            payload,
        )
        conn.commit()
    finally:
        conn.close()


def record_prediction(**kwargs: Any) -> None:
    """Log model predictions for later analysis."""

    conn = io.connect()
    try:
        io.run_migrations(conn)
        payload: Dict[str, Any] = {
            "pred_id": kwargs.get("pred_id"),
            "ts": kwargs.get("ts"),
            "symbol": kwargs.get("symbol"),
            "prob_up": kwargs.get("prob_up"),
            "prob_down": kwargs.get("prob_down"),
            "model_ver": kwargs.get("model_ver"),
            "features_hash": kwargs.get("features_hash"),
            "decision_threshold": kwargs.get("decision_threshold"),
            "chosen": kwargs.get("chosen"),
            "realized_hit": kwargs.get("realized_hit"),
            "realized_return": kwargs.get("realized_return"),
            "realized_ts": kwargs.get("realized_ts"),
            "meta": json.dumps(kwargs.get("meta", {})),
        }
        conn.execute(
            (
                "INSERT OR REPLACE INTO model_predictions(pred_id, ts, symbol, prob_up, prob_down, model_ver, features_hash, decision_threshold, chosen, realized_hit, realized_return, realized_ts, meta) "
                "VALUES (:pred_id, :ts, :symbol, :prob_up, :prob_down, :model_ver, :features_hash, :decision_threshold, :chosen, :realized_hit, :realized_return, :realized_ts, :meta)"
            ),
            payload,
        )
        conn.commit()
    finally:
        conn.close()


__all__ = ["log_trade", "record_prediction"]
