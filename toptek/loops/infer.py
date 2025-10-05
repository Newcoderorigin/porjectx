"""Real-time inference loop streaming predictions into the data layer."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml
import pickle

try:  # pragma: no cover - optional dependency
    import joblib
except ModuleNotFoundError:  # pragma: no cover - fallback to stdlib pickle
    joblib = None  # type: ignore[assignment]

from toptek.core import features as core_features
from toptek.data import SQLiteBarFeed, connect, run_migrations


@dataclass(frozen=True)
class InferConfig:
    """Configuration for the streaming inference service."""

    models_dir: Path
    symbols: Sequence[str]
    decision_threshold: float = 0.5
    max_history: int = 2048


def load_config(path: Path) -> InferConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    base = path.parent
    models_dir = base / raw["models_dir"] if not Path(raw["models_dir"]).is_absolute() else Path(raw["models_dir"])
    symbols_raw = raw.get("symbols") or []
    if isinstance(symbols_raw, str):
        symbols = (symbols_raw,)
    else:
        symbols = tuple(str(symbol).upper() for symbol in symbols_raw if symbol)
    if not symbols:
        raise ValueError("At least one symbol must be configured for inference")
    return InferConfig(
        models_dir=models_dir.resolve(),
        symbols=symbols,
        decision_threshold=float(raw.get("decision_threshold", 0.5)),
        max_history=int(raw.get("max_history", 2048)),
    )


class PredictionService:
    """Stream bars, emit calibrated probabilities, and settle outcomes."""

    def __init__(self, config: InferConfig, *, conn=None) -> None:
        self.config = config
        self._owns_connection = conn is None
        self.conn = conn or connect()
        run_migrations(self.conn)
        self.feed = SQLiteBarFeed(self.conn)
        self.calibrator, self.card = self._load_artifacts(config.models_dir)
        self.feature_names: List[str] = list(self.card.get("feature_names", []))
        if not self.feature_names:
            raise ValueError("Model card missing feature names")
        self.model_version = str(
            self.card.get("versions", {}).get("model", "unknown")
        )
        self.features_hash = str(self.card.get("features_hash", ""))
        self._bars: Dict[str, pd.DataFrame] = {}
        self._known_predictions: Dict[str, set[str]] = {
            symbol: self._load_existing_predictions(symbol) for symbol in config.symbols
        }

    def _load_artifacts(self, models_dir: Path):
        calibrator_path = models_dir / "calibrator.pkl"
        card_path = models_dir / "model_card.json"
        if not calibrator_path.exists():
            raise FileNotFoundError(f"Missing calibrator: {calibrator_path}")
        if not card_path.exists():
            raise FileNotFoundError(f"Missing model card: {card_path}")
        calibrator = self._load_pickle(calibrator_path)
        with card_path.open("r", encoding="utf-8") as handle:
            card = json.load(handle)
        return calibrator, card

    @staticmethod
    def _load_pickle(path: Path):
        if joblib is not None:
            return joblib.load(path)
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _load_existing_predictions(self, symbol: str) -> set[str]:
        cursor = self.conn.execute(
            "SELECT ts FROM model_predictions WHERE symbol = ? ORDER BY ts", (symbol,)
        )
        rows = cursor.fetchall()
        return {
            pd.Timestamp(row[0]).isoformat()
            for row in rows
            if row[0] is not None
        }

    def close(self) -> None:
        if self._owns_connection:
            self.conn.close()

    def run_once(self) -> Dict[str, List[Dict[str, object]]]:
        """Process the latest bars for all configured symbols."""

        results: Dict[str, List[Dict[str, object]]] = {}
        changed = False
        for symbol in self.config.symbols:
            ingested = self._ingest(symbol)
            new_predictions = self._predict(symbol) if ingested else []
            settled = self._settle(symbol)
            if new_predictions:
                results[symbol] = new_predictions
            if ingested or new_predictions or settled:
                changed = True
        if changed:
            self.conn.commit()
        return results

    def _ingest(self, symbol: str) -> bool:
        history = self._bars.get(symbol)
        since = history.index.max() if history is not None and not history.empty else None
        frame = self.feed.fetch(symbol, since=since)
        if frame.empty:
            return False
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        if history is None or history.empty:
            updated = frame
        else:
            updated = pd.concat([history, frame])
            updated = updated[~updated.index.duplicated(keep="last")]
        if len(updated) > self.config.max_history:
            updated = updated.tail(self.config.max_history)
        self._bars[symbol] = updated
        return True

    def _predict(self, symbol: str) -> List[Dict[str, object]]:
        bars = self._bars.get(symbol)
        if bars is None or bars.empty:
            return []
        features_df = self._build_feature_frame(bars)
        if features_df.empty:
            return []
        known = self._known_predictions.setdefault(symbol, set())
        index_iso = features_df.index.map(pd.Timestamp.isoformat)
        candidate_mask = ~pd.Index(index_iso).isin(known)
        candidates = features_df.loc[candidate_mask]
        if candidates.empty:
            return []
        matrix = candidates[self.feature_names].to_numpy(dtype=np.float64)
        probabilities = self.calibrator.predict_proba(matrix)[:, 1]
        inserted: List[Dict[str, object]] = []
        candidate_isos = pd.Index(index_iso)[candidate_mask]
        for ts, ts_iso, prob_up in zip(candidates.index, candidate_isos, probabilities):
            record = self._persist_prediction(symbol, ts, float(prob_up))
            if record is None:
                continue
            known.add(ts_iso)
            inserted.append(record)
        return inserted

    def _persist_prediction(
        self, symbol: str, ts: pd.Timestamp, prob_up: float
    ) -> Dict[str, object] | None:
        prob_down = float(max(0.0, min(1.0, 1.0 - prob_up)))
        threshold = self.config.decision_threshold
        chosen = int(prob_up >= threshold)
        pred_id = f"{symbol}-{ts.isoformat()}"
        payload = {
            "pred_id": pred_id,
            "ts": ts.isoformat(),
            "symbol": symbol,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "model_ver": self.model_version,
            "features_hash": self.features_hash,
            "decision_threshold": threshold,
            "chosen": chosen,
            "meta": json.dumps({"source": "infer"}),
        }
        cursor = self.conn.execute(
            (
                "INSERT OR IGNORE INTO model_predictions(" \
                "pred_id, ts, symbol, prob_up, prob_down, model_ver, features_hash, decision_threshold, chosen, meta) "
                "VALUES (:pred_id, :ts, :symbol, :prob_up, :prob_down, :model_ver, :features_hash, :decision_threshold, :chosen, :meta)"
            ),
            payload,
        )
        if cursor.rowcount and cursor.rowcount > 0:
            return payload
        return None

    def _build_feature_frame(self, bars: pd.DataFrame) -> pd.DataFrame:
        feature_map = core_features.compute_features(bars)
        features_df = pd.DataFrame(feature_map, index=bars.index)
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df = features_df[self.feature_names]
        features_df.dropna(inplace=True)
        return features_df

    def _settle(self, symbol: str) -> bool:
        bars = self._bars.get(symbol)
        if bars is None or len(bars) < 2:
            return False
        cursor = self.conn.execute(
            "SELECT pred_id, ts FROM model_predictions WHERE symbol = ? AND realized_ts IS NULL ORDER BY ts",
            (symbol,),
        )
        rows = cursor.fetchall()
        if not rows:
            return False
        timestamps = bars.index
        closes = bars["close"].to_numpy(dtype=float)
        updated = False
        for pred_id, ts_raw in rows:
            ts = pd.Timestamp(ts_raw)
            try:
                idx = int(timestamps.get_loc(ts))
            except KeyError:
                continue
            if idx >= len(timestamps) - 1:
                continue
            current_close = closes[idx]
            next_close = closes[idx + 1]
            realized_return = 0.0 if current_close == 0 else (next_close - current_close) / current_close
            realized_hit = int(next_close > current_close)
            realized_ts = timestamps[idx + 1]
            self.conn.execute(
                (
                    "UPDATE model_predictions "
                    "SET realized_hit = ?, realized_return = ?, realized_ts = ? "
                    "WHERE pred_id = ?"
                ),
                (
                    realized_hit,
                    realized_return,
                    realized_ts.isoformat(),
                    pred_id,
                ),
            )
            updated = True
        return updated

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toptek streaming inference service")
    parser.add_argument("--config", required=True, help="Path to inference YAML config")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - exercised by integration test entry point
    args = _parse_args()
    config = load_config(Path(args.config).resolve())
    service = PredictionService(config)
    try:
        while True:
            produced = service.run_once()
            if not produced:
                break
    finally:
        service.close()


__all__ = ["InferConfig", "PredictionService", "load_config"]
