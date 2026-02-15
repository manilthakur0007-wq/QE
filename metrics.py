"""Performance metrics used to evaluate strategy results."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    cumulative = (1 + returns).cumprod()
    total_ret = cumulative.iloc[-1]
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return total_ret ** (1 / years) - 1


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    exret = returns - rf / periods_per_year
    return np.sqrt(periods_per_year) * exret.mean() / (exret.std() + 1e-9)


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    exret = returns - rf / periods_per_year
    downside = exret[exret < 0]
    dstd = downside.std()
    if dstd == 0 or np.isnan(dstd):
        return np.nan
    return np.sqrt(periods_per_year) * exret.mean() / dstd


def win_rate(returns: pd.Series) -> float:
    wins = (returns > 0).sum()
    total = returns.count()
    if total == 0:
        return 0.0
    return wins / total


def expectancy(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    if len(wins) == 0 or len(losses) == 0:
        return np.nan
    return (wins.mean() * len(wins) + losses.mean() * len(losses)) / len(returns)


def calculate_metrics(returns: pd.Series) -> dict:
    return {
        "Sharpe": float(sharpe_ratio(returns)),
        "Sortino": float(sortino_ratio(returns)),
        "Max Drawdown": float(max_drawdown(returns)),
        "CAGR": float(annualize_return(returns)),
        "Win Rate": float(win_rate(returns)),
        "Expectancy": float(expectancy(returns)),
    }


__all__ = ["calculate_metrics"]
