"""Feature engineering: compute multi-timeframe momentum, volatility,
liquidity, regime filters and normalization routines.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return (df - df.rolling(window).mean()) / (df.rolling(window).std() + 1e-9)


def compute_features(ohlcv: pd.DataFrame, ohlcv_extra: dict = None) -> dict:
    """Compute a feature dictionary from prices.

    `ohlcv` is expected to be a DataFrame of close prices (columns = tickers).
    If `ohlcv_extra` provided, it may contain `high`, `low`, `volume` DataFrames.
    """
    close = ohlcv
    returns = close.pct_change()

    mom_5 = close.pct_change(5)
    mom_20 = close.pct_change(20)
    mom_60 = close.pct_change(60)

    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()

    high = None
    low = None
    volume = None
    if ohlcv_extra:
        high = ohlcv_extra.get("high")
        low = ohlcv_extra.get("low")
        volume = ohlcv_extra.get("volume")

    if high is not None and low is not None and close is not None:
        atr_14 = atr(high, low, close, 14)
    else:
        atr_14 = vol_20

    avg_volume_20 = None
    if volume is not None:
        avg_volume_20 = volume.rolling(20).mean()

    ma_200 = close.rolling(200).mean()
    trend_regime = (close > ma_200).astype(int)  # 1 = uptrend, 0 = down

    # Z-score normalize momentum and volatility features cross-sectionally
    def cs_zscore(df):
        return df.rank(axis=1, pct=True)

    mom_rank = cs_zscore(mom_20)

    features = {
        "close": close,
        "returns": returns,
        "mom_5": mom_5,
        "mom_20": mom_20,
        "mom_60": mom_60,
        "mom_rank": mom_rank,
        "vol_20": vol_20,
        "vol_60": vol_60,
        "atr_14": atr_14,
        "avg_vol_20": avg_volume_20,
        "ma_200": ma_200,
        "trend_regime": trend_regime,
        "z_mom_20": zscore(mom_20, 20),
    }

    return features


__all__ = ["compute_features", "atr", "zscore"]
