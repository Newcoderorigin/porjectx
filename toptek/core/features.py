"""Feature engineering utilities built on ``ta`` and ``numpy``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from ta.momentum import (
    RSIIndicator,
    ROCIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
from ta.trend import (
    ADXIndicator,
    CCIIndicator,
    EMAIndicator,
    MACD,
    PSARIndicator,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel
from ta.volume import EaseOfMovementIndicator, MFIIndicator, OnBalanceVolumeIndicator


@dataclass
class FeatureResult:
    """Represents computed feature arrays."""

    name: str
    values: np.ndarray


def compute_features(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute a broad set of technical indicators.

    Args:
        data: DataFrame with columns ``open``, ``high``, ``low``, ``close``, ``volume``.

    Returns:
        Mapping from feature name to numpy array.
    """

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"].replace(0, np.nan)
    features: Dict[str, np.ndarray] = {}

    features["sma_10"] = SMAIndicator(close, window=10).sma_indicator().to_numpy()
    features["sma_20"] = SMAIndicator(close, window=20).sma_indicator().to_numpy()
    features["ema_12"] = EMAIndicator(close, window=12).ema_indicator().to_numpy()
    features["ema_26"] = EMAIndicator(close, window=26).ema_indicator().to_numpy()
    features["ema_50"] = EMAIndicator(close, window=50).ema_indicator().to_numpy()
    features["ema_200"] = EMAIndicator(close, window=200).ema_indicator().to_numpy()

    macd = MACD(close)
    features["macd"] = macd.macd().to_numpy()
    features["macd_signal"] = macd.macd_signal().to_numpy()
    features["macd_hist"] = macd.macd_diff().to_numpy()

    features["rsi_14"] = RSIIndicator(close, window=14).rsi().to_numpy()
    features["roc_10"] = ROCIndicator(close, window=10).roc().to_numpy()
    features["roc_20"] = ROCIndicator(close, window=20).roc().to_numpy()
    features["willr_14"] = (
        WilliamsRIndicator(high, low, close, lbp=14).williams_r().to_numpy()
    )
    features["stoch_k"] = StochasticOscillator(high, low, close).stoch().to_numpy()
    features["stoch_d"] = (
        StochasticOscillator(high, low, close).stoch_signal().to_numpy()
    )

    atr = AverageTrueRange(high, low, close, window=14)
    features["atr_14"] = atr.average_true_range().to_numpy()

    bb = BollingerBands(close, window=20, window_dev=2)
    features["bb_high"] = bb.bollinger_hband().to_numpy()
    features["bb_low"] = bb.bollinger_lband().to_numpy()
    features["bb_percent"] = bb.bollinger_pband().to_numpy()
    features["bb_width"] = bb.bollinger_wband().to_numpy()

    donchian = DonchianChannel(high, low, close, window=20)
    features["donchian_high"] = donchian.donchian_channel_hband().to_numpy()
    features["donchian_low"] = donchian.donchian_channel_lband().to_numpy()
    features["donchian_width"] = features["donchian_high"] - features["donchian_low"]

    adx = ADXIndicator(high, low, close, window=14)
    features["adx_14"] = adx.adx().to_numpy()
    features["di_plus"] = adx.adx_pos().to_numpy()
    features["di_minus"] = adx.adx_neg().to_numpy()

    features["obv"] = (
        OnBalanceVolumeIndicator(close, volume.fillna(0)).on_balance_volume().to_numpy()
    )
    features["mfi_14"] = (
        MFIIndicator(high, low, close, volume.fillna(0), window=14)
        .money_flow_index()
        .to_numpy()
    )
    features["eom_14"] = (
        EaseOfMovementIndicator(high, low, volume.fillna(1), window=14)
        .ease_of_movement()
        .to_numpy()
    )

    features["cci_20"] = CCIIndicator(high, low, close, window=20).cci().to_numpy()
    # Normalise PSAR inputs to avoid pandas treating integer keys as labels.
    original_index = high.index
    high_psar = high.reset_index(drop=True)
    low_psar = low.reset_index(drop=True)
    close_psar = close.reset_index(drop=True)
    psar = PSARIndicator(high_psar, low_psar, close_psar).psar()
    features["psar"] = pd.Series(psar.to_numpy(), index=original_index).to_numpy()

    log_returns = np.log(close).diff().fillna(0).to_numpy()
    features["return_1"] = log_returns
    features["return_5"] = pd.Series(log_returns).rolling(window=5).sum().to_numpy()
    features["return_20"] = pd.Series(log_returns).rolling(window=20).sum().to_numpy()

    features["volatility_close"] = (
        pd.Series(log_returns).rolling(window=20).std().to_numpy()
    )
    high_low = np.log(high / low)
    features["volatility_parkinson"] = high_low.rolling(window=20).std().to_numpy()

    features["volume_zscore"] = (volume - volume.rolling(20).mean()) / volume.rolling(
        20
    ).std()
    features["volume_zscore"] = features["volume_zscore"].to_numpy()

    return features


__all__ = ["compute_features", "FeatureResult"]
