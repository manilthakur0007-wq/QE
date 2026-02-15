"""Event-driven-like backtest engine with realistic fills, commission and slippage."""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, weights: pd.DataFrame, capital: float = 10000.0, commission: float = 0.0005, slippage: float = 0.0005) -> Tuple[pd.Series, pd.DataFrame]:
    """Run a simple daily backtest.

    prices: close prices DataFrame (date x symbol)
    signals: discrete signals (-1, 0, 1)
    weights: target weights (fractional of portfolio)

    Returns (daily_returns, ledger) where ledger is detailed PnL tracking.
    """
    dates = prices.index
    cash = capital
    holdings = pd.Series(0.0, index=prices.columns)
    avg_price = pd.Series(np.nan, index=prices.columns)

    equity_curve = []
    ledger_rows = []

    prev_weights = pd.Series(0.0, index=prices.columns)

    for dt in dates:
        px = prices.loc[dt]
        # determine desired weights for this date
        if dt in weights.index:
            tgt_w = weights.loc[dt].fillna(0)
        else:
            tgt_w = prev_weights * 0

        # compute portfolio value
        mv = (holdings * px).sum()
        total_equity = cash + mv

        # target dollar allocations
        tgt_dollars = tgt_w * total_equity

        # compute notional trades in dollars and simulate fills at today's close
        trade_dollars = tgt_dollars - (holdings * px)
        # apply slippage and commission as cost proportional to trade dollars
        costs = trade_dollars.abs() * (commission + slippage)

        # update cash and holdings
        for sym in prices.columns:
            td = trade_dollars.get(sym, 0.0)
            p = px.get(sym, np.nan)
            if np.isnan(p) or p == 0:
                continue
            qty = td / p
            holdings[sym] += qty
            avg_price[sym] = p if np.isfinite(p) else avg_price[sym]
            cash -= td
            cash -= costs.get(sym, 0.0)
            ledger_rows.append({
                "date": dt,
                "symbol": sym,
                "price": p,
                "trade_dollars": td,
                "qty": qty,
                "cost": costs.get(sym, 0.0),
                "cash": cash,
            })

        mv = (holdings * px).sum()
        total_equity = cash + mv
        equity_curve.append(total_equity)
        prev_weights = tgt_w

    equity = pd.Series(equity_curve, index=dates)
    returns = equity.pct_change().fillna(0)
    ledger = pd.DataFrame(ledger_rows)
    return returns, ledger


__all__ = ["run_backtest"]
