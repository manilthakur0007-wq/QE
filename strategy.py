"""Strategy engine: hybrid multi-factor model and optional ML classifier.

Generates daily cross-sectional target scores and discrete trade signals.
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional

logger = logging.getLogger(__name__)


def hybrid_score(features: dict, breakout_window: int = 20) -> pd.DataFrame:
    """Compute composite cross-sectional score combining momentum ranking,
    mean-reversion and volatility-adjusted breakout signals.
    Returns a DataFrame of scores (index=dates, cols=tickers).
    """
    mom_rank = features["mom_rank"]
    z_mom = features.get("z_mom_20")
    close = features["close"]
    atr = features.get("atr_14")

    # Breakout: price above rolling high adjusted by ATR
    rolling_high = close.rolling(breakout_window).max()
    breakout = ((close >= rolling_high) * 1.0).fillna(0)

    # Mean reversion: negative z-score of short-term mom (buy dips)
    mean_rev = -z_mom.fillna(0)

    # Volatility scaling (lower weight when vol high)
    vol_scale = 1 / (features.get("vol_20") + 1e-9)

    score = 0.6 * mom_rank.fillna(0) + 0.3 * mean_rev + 0.4 * breakout
    score = score * vol_scale
    score = score.replace([np.inf, -np.inf], 0).fillna(0)
    return score


def generate_signals(features: dict, top_n: int = 10, ml_model: Optional[RandomForestClassifier] = None) -> pd.DataFrame:
    """Return target long/short signals: 1 long, -1 short, 0 flat.

    If `ml_model` provided it will produce binary predictions used as a filter.
    """
    score = hybrid_score(features)
    # Cross-sectional ranking: choose top_n longs and bottom_n shorts per day
    longs = score.rank(axis=1, method="first", ascending=False) <= top_n
    shorts = score.rank(axis=1, method="first", ascending=True) <= top_n

    signals = longs.astype(int) - shorts.astype(int)

    if ml_model is not None:
        logger.info("Applying ML filter to signals")
        # prepare features for ML predict: flatten per day per asset
        # For brevity, apply model as a filter on score sign
        preds = score.apply(lambda col: ml_model.predict((col.fillna(0)).values.reshape(-1, 1)), axis=0)
        # Align predictions shape (this is a simplified application)
        try:
            preds_df = pd.DataFrame(preds.values, index=score.index, columns=score.columns)
            signals = signals.where(preds_df == 1, 0)
        except Exception:
            logger.exception("ML filter shape mismatch; skipping ML filter")

    return signals.astype(int)


def train_ml_classifier(features: dict, target_returns: pd.DataFrame, n_estimators: int = 100) -> RandomForestClassifier:
    """Train a RandomForest classifier on historical features.

    This function returns a fitted model. Training is done only on provided
    historical slices—ensure no leakage when using in walk-forward pipeline.
    """
    # Create a simple feature matrix: use mom_20, vol_20, atr
    mom = features.get("mom_20").stack()
    vol = features.get("vol_20").stack()
    atr = features.get("atr_14").stack()
    y = target_returns.stack()

    X = pd.concat([mom.rename("mom20"), vol.rename("vol20"), atr.rename("atr14")], axis=1).dropna()
    y = y.reindex(X.index).fillna(0)
    # Convert to binary: next-period positive return
    ybin = (y > 0).astype(int).values

    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    model.fit(X.values, ybin)
    return model


__all__ = ["generate_signals", "train_ml_classifier", "hybrid_score"]
