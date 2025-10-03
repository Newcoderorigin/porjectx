"""Feature engineering utilities built on ``ta`` and ``numpy``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD, PSARIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel
from ta.volume import EaseOfMovementIndicator, MFIIndicator, OnBalanceVolumeIndicator


@dataclass
class FeatureResult:
    """Represents computed feature arrays."""

    name: str
    values: np.ndarray


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute a broad set of technical indicators.

    Args:
        data: DataFrame with columns ``open``, ``high``, ``low``, ``close``, ``volume``.

    Returns:
        ``pandas.DataFrame`` of indicator features aligned to ``data`` index with
        early NaN rows removed.
    """

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"].replace(0, np.nan)
    features: Dict[str, pd.Series] = {}

    features["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    features["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    features["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    features["ema_26"] = EMAIndicator(close, window=26).ema_indicator()
    features["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    features["ema_200"] = EMAIndicator(close, window=200).ema_indicator()

    macd = MACD(close)
    features["macd"] = macd.macd()
    features["macd_signal"] = macd.macd_signal()
    features["macd_hist"] = macd.macd_diff()

    features["rsi_14"] = RSIIndicator(close, window=14).rsi()
    features["roc_10"] = ROCIndicator(close, window=10).roc()
    features["roc_20"] = ROCIndicator(close, window=20).roc()
    features["willr_14"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    features["stoch_k"] = StochasticOscillator(high, low, close).stoch()
    features["stoch_d"] = StochasticOscillator(high, low, close).stoch_signal()

    atr = AverageTrueRange(high, low, close, window=14)
    features["atr_14"] = atr.average_true_range()

    bb = BollingerBands(close, window=20, window_dev=2)
    features["bb_high"] = bb.bollinger_hband()
    features["bb_low"] = bb.bollinger_lband()
    features["bb_percent"] = bb.bollinger_pband()
    features["bb_width"] = bb.bollinger_wband()

    donchian = DonchianChannel(high, low, close, window=20)
    features["donchian_high"] = donchian.donchian_channel_hband()
    features["donchian_low"] = donchian.donchian_channel_lband()
    features["donchian_width"] = features["donchian_high"] - features["donchian_low"]

    adx = ADXIndicator(high, low, close, window=14)
    features["adx_14"] = adx.adx()
    features["di_plus"] = adx.adx_pos()
    features["di_minus"] = adx.adx_neg()

    features["obv"] = OnBalanceVolumeIndicator(close, volume.fillna(0)).on_balance_volume()
    features["mfi_14"] = MFIIndicator(high, low, close, volume.fillna(0), window=14).money_flow_index()
    features["eom_14"] = EaseOfMovementIndicator(high, low, volume.fillna(1), window=14).ease_of_movement()

    features["cci_20"] = CCIIndicator(high, low, close, window=20).cci()
    psar = PSARIndicator(high, low, close)
    features["psar"] = psar.psar()

    log_returns = np.log(close).diff().fillna(0)
    features["return_1"] = log_returns
    features["return_5"] = log_returns.rolling(window=5).sum()
    features["return_20"] = log_returns.rolling(window=20).sum()

    features["volatility_close"] = log_returns.rolling(window=20).std()
    high_low = np.log(high / low)
    features["volatility_parkinson"] = high_low.rolling(window=20).std()

    volume_zscore = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    features["volume_zscore"] = volume_zscore

    frame = pd.DataFrame(features, index=data.index)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna().astype(float)
    return frame


__all__ = ["compute_features", "FeatureResult"]
