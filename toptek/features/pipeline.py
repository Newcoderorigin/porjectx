"""Unified feature pipeline shared across research, training, and execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from toptek.core import features as core_features


@dataclass(frozen=True)
class FeatureBundle:
    """Container for feature arrays, labels, and metadata."""

    X: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any]


def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    return digest


def _load_cache(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    arrays = np.load(cache_path, allow_pickle=False)
    meta_path = cache_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return arrays["X"], arrays["y"], meta


def _store_cache(cache_path: Path, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, mask=np.asarray(meta.get("mask", []), dtype=np.int8))
    with cache_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def build_features(
    df: pd.DataFrame,
    *,
    engine: str = "pandas",
    cache_dir: str | Path = ".cache",
) -> FeatureBundle:
    """Construct features and labels using a shared pipeline."""

    if engine != "pandas":
        raise ValueError(f"Unsupported engine: {engine}")
    if df.empty:
        raise ValueError("Input dataframe must not be empty")

    cache_root = Path(cache_dir)
    cache_key = _hash_dataframe(df)
    cache_path = cache_root / f"features_{cache_key}.npz"
    cached = _load_cache(cache_path)
    if cached:
        X_cached, y_cached, meta_cached = cached
        return FeatureBundle(X_cached, y_cached, meta_cached)

    feature_map = core_features.compute_features(df)
    feature_df = pd.DataFrame(feature_map, index=df.index)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Labels represent next-bar hit rate: 1 if next close is above current close.
    target = (df["close"].shift(-1) > df["close"]).astype(float)
    target.iloc[-1] = np.nan

    valid_mask = feature_df.notna().all(axis=1) & target.notna()
    cleaned_features = feature_df.loc[valid_mask]
    cleaned_target = target.loc[valid_mask].astype(np.int8)

    if cleaned_features.empty:
        raise ValueError("No valid feature rows after cleaning")

    X = cleaned_features.to_numpy(dtype=np.float64)
    y = cleaned_target.to_numpy(dtype=np.int8)

    meta: Dict[str, Any] = {
        "feature_names": cleaned_features.columns.tolist(),
        "valid_index": cleaned_features.index.astype(str).tolist(),
        "mask": valid_mask.to_numpy(dtype=np.int8).tolist(),
        "cache_key": cache_key,
        "dropped_rows": int((~valid_mask).sum()),
    }

    _store_cache(cache_path, X, y, meta)

    return FeatureBundle(X=X, y=y, meta=meta)


__all__ = ["build_features", "FeatureBundle"]
