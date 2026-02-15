"""Demo runner that produces a resume-ready backtest artifact using real data."""
from __future__ import annotations

import os
import json
import matplotlib.pyplot as plt

from data import fetch_historical
from features import compute_features
from strategy import generate_signals
from risk import volatility_position_sizing
from backtest import run_backtest
from metrics import calculate_metrics


def run_demo():
    tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","BAC","XOM","CVX","WMT","KO","PEP","NKE","MCD","DIS","NFLX"]
    start = "2018-01-01"
    end = "2023-12-31"

    prices = fetch_historical(tickers, start, end)
    if prices.empty:
        print("No price data; aborting demo")
        return

    features = compute_features(prices)
    # Aggressive small-N top selection for demo clarity
    signals = generate_signals(features, top_n=3)

    weights = volatility_position_sizing(signals, features.get("atr_14"), 10000.0, max_leverage=1.0, max_pos_pct=0.2)

    returns, ledger = run_backtest(prices, signals, weights, capital=10000.0)

    metrics = calculate_metrics(returns)

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    equity = (1 + returns).cumprod()
    plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values)
    plt.title("Demo Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("artifacts", "equity_curve.png"))
    print("Demo complete — artifacts saved in ./artifacts/")


if __name__ == "__main__":
    run_demo()
