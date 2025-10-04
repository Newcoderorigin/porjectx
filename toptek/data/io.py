"""SQLite IO helpers for the TopTek project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class IOPaths:
    """Container describing canonical filesystem locations."""

    root: Path
    var: Path
    data: Path
    db: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.var = Path(self.var)
        self.data = Path(self.data)
        self.db = Path(self.db)
        self.var.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "IOPaths":
        project_root = Path(__file__).resolve().parents[2]
        var_dir = project_root / "var"
        data_dir = project_root / "data"
        db_path = var_dir / "toptek.db"
        return cls(root=project_root, var=var_dir, data=data_dir, db=db_path)


_DEFAULT_PATHS = IOPaths.default()
DATA_DIR = _DEFAULT_PATHS.data
VAR_DIR = _DEFAULT_PATHS.var


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create a SQLite connection using the configured locations."""

    target = Path(db_path) if db_path is not None else _DEFAULT_PATHS.db
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def run_migrations(conn: sqlite3.Connection) -> None:
    """Create tables required for logging trades and predictions."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            session_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT,
            qty REAL,
            entry_ts TEXT,
            exit_ts TEXT,
            entry_px REAL,
            exit_px REAL,
            pnl REAL,
            label_hit INTEGER,
            label_return REAL,
            meta TEXT
        );

        CREATE TABLE IF NOT EXISTS model_predictions (
            pred_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prob_up REAL NOT NULL,
            prob_down REAL NOT NULL,
            model_ver TEXT,
            features_hash TEXT,
            decision_threshold REAL,
            chosen INTEGER,
            meta TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts
            ON trades(symbol, entry_ts);
        CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts
            ON model_predictions(symbol, ts);
        """
    )
    conn.commit()


def _generate_bars(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2021-01-01", periods=rows, freq="H")
    shocks = rng.normal(0.0, 0.004, rows)
    log_prices = np.cumsum(shocks)
    close = 100.0 * np.exp(log_prices)
    close[0] = 100.0
    open_px = np.empty_like(close)
    open_px[0] = close[0]
    open_px[1:] = close[:-1]
    spreads = np.abs(rng.normal(0.001, 0.0005, rows))
    high = np.maximum(open_px, close) * (1 + spreads)
    low = np.minimum(open_px, close) * (1 - spreads)
    volume = rng.integers(500, 1500, rows)
    bars = pd.DataFrame(
        {
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    bars.index.name = "ts"
    return bars


def _generate_trades_and_predictions(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = bars["close"].to_numpy(dtype=float)
    opens = bars["open"].to_numpy(dtype=float)
    timestamps = bars.index.to_pydatetime()
    rng = np.random.default_rng(7)

    trades_records = []
    predictions_records = []

    for i, ts in enumerate(timestamps):
        entry_px = float(opens[i])
        if i < len(closes) - 1:
            exit_px = float(closes[i + 1])
            exit_ts = timestamps[i + 1]
        else:
            exit_px = float(closes[i])
            exit_ts = ts

        ret = 0.0 if entry_px == 0 else (exit_px - entry_px) / entry_px
        label_hit = int(exit_px > entry_px)
        pnl = (exit_px - entry_px) * 1.0

        trade_payload = {
            "trade_id": f"demo_trade_{i}",
            "session_id": "demo_session",
            "symbol": "DEMO",
            "side": "BUY" if label_hit else "SELL",
            "qty": 1.0,
            "entry_ts": ts.isoformat(),
            "exit_ts": exit_ts.isoformat(),
            "entry_px": entry_px,
            "exit_px": exit_px,
            "pnl": pnl,
            "label_hit": label_hit,
            "label_return": ret,
            "meta": "{}",
        }
        trades_records.append(trade_payload)

        prob_up = float(np.clip(0.5 + ret * 5 + rng.normal(0, 0.02), 0.01, 0.99))
        prob_down = float(np.clip(1.0 - prob_up + rng.normal(0, 0.01), 0.0, 0.99))
        normaliser = prob_up + prob_down
        if normaliser <= 0:
            prob_up = 0.5
            prob_down = 0.5
        else:
            prob_up /= normaliser
            prob_down /= normaliser
        chosen = int(prob_up >= 0.5)

        predictions_records.append(
            {
                "pred_id": f"demo_pred_{i}",
                "ts": ts.isoformat(),
                "symbol": "DEMO",
                "prob_up": prob_up,
                "prob_down": prob_down,
                "model_ver": "demo-v1",
                "features_hash": f"hash_{i}",
                "decision_threshold": 0.5,
                "chosen": chosen,
                "meta": "{}",
            }
        )

    trades_df = pd.DataFrame(trades_records)
    preds_df = pd.DataFrame(predictions_records)
    return trades_df, preds_df


def load_demo_data(
    conn: sqlite3.Connection,
    *,
    rows: int = 512,
) -> Mapping[str, int]:
    """Populate the SQLite store and filesystem with deterministic demo data."""

    if rows <= 0:
        raise ValueError("rows must be positive")

    run_migrations(conn)
    bars = _generate_bars(rows)
    trades_df, preds_df = _generate_trades_and_predictions(bars)

    conn.execute("DELETE FROM trades")
    conn.execute("DELETE FROM model_predictions")
    conn.executemany(
        (
            "INSERT OR REPLACE INTO trades(" \
            "trade_id, session_id, symbol, side, qty, entry_ts, exit_ts, entry_px, exit_px, pnl, label_hit, label_return, meta) "
            "VALUES (:trade_id, :session_id, :symbol, :side, :qty, :entry_ts, :exit_ts, :entry_px, :exit_px, :pnl, :label_hit, :label_return, :meta)"
        ),
        trades_df.to_dict(orient="records"),
    )
    conn.executemany(
        (
            "INSERT OR REPLACE INTO model_predictions(" \
            "pred_id, ts, symbol, prob_up, prob_down, model_ver, features_hash, decision_threshold, chosen, meta) "
            "VALUES (:pred_id, :ts, :symbol, :prob_up, :prob_down, :model_ver, :features_hash, :decision_threshold, :chosen, :meta)"
        ),
        preds_df.to_dict(orient="records"),
    )
    conn.commit()

    bars_path = DATA_DIR / "demo_bars.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(bars_path)

    return {
        "trades": int(trades_df.shape[0]),
        "predictions": int(preds_df.shape[0]),
    }


def export_to_parquet(
    conn: sqlite3.Connection,
    *,
    dest: str | Path,
    tables: Iterable[str] | None = None,
) -> Dict[str, Path]:
    """Export requested tables to Parquet files under ``dest``."""

    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    selected_tables = list(tables) if tables is not None else [
        "trades",
        "model_predictions",
    ]

    exports: Dict[str, Path] = {}
    for table in selected_tables:
        frame = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        output_path = destination / f"{table}.parquet"
        frame.to_parquet(output_path, index=False)
        exports[table] = output_path
    return exports


__all__ = [
    "IOPaths",
    "DATA_DIR",
    "VAR_DIR",
    "connect",
    "run_migrations",
    "load_demo_data",
    "export_to_parquet",
]
