"""Risk management utilities: position sizing, stops, exposure limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def volatility_position_sizing(signals: pd.DataFrame, atr: pd.DataFrame, capital: float, max_leverage: float = 2.0, max_pos_pct: float = 0.1) -> pd.DataFrame:
    """Simple volatility-based sizing: allocate notional proportional to 1/ATR,
    capped by `max_pos_pct` and overall `max_leverage`.
    Returns weights as fraction of portfolio (signed).
    """
    # risk per unit price ~ ATR; smaller ATR => larger size
    inv_risk = 1 / (atr + 1e-9)
    raw = signals * inv_risk

    # normalize abs exposure to 1 then scale to max leverage and cap per position
    abs_sum = raw.abs().sum(axis=1).replace(0, np.nan)
    norm = raw.div(abs_sum, axis=0).fillna(0)

    # apply gross leverage cap
    gross = norm.abs().sum(axis=1)
    lever = (np.minimum(max_leverage, 1.0 / (gross + 1e-9))).replace([np.inf, -np.inf], 1.0)
    weights = norm.mul(lever, axis=0)

    # enforce per-position max percent
    weights = weights.clip(lower=-max_pos_pct, upper=max_pos_pct)
    return weights


def apply_stop_loss(entry_price: float, atr: float, multiplier: float = 2.0) -> float:
    """Return stop price given entry and ATR multiplier.
    For long positions stop = entry - multiplier * ATR; for short, reverse.
    """
    return entry_price - multiplier * atr


def portfolio_drawdown_cutoff(equity_series: pd.Series, max_drawdown: float = 0.15) -> bool:
    """Return True if drawdown exceeds max_drawdown (stop trading / liquidate).
    equity_series is index by date of portfolio equity.
    """
    peak = equity_series.cummax()
    drawdown = (peak - equity_series) / peak
    return drawdown.iloc[-1] >= max_drawdown


__all__ = ["volatility_position_sizing", "apply_stop_loss", "portfolio_drawdown_cutoff"]
