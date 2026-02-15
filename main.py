"""Main entry point to run backtests or paper trading using the modular engine."""
from __future__ import annotations

import argparse
import logging
import yaml
import os

from logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from data import fetch_historical
from features import compute_features
from strategy import generate_signals, train_ml_classifier
from risk import volatility_position_sizing, portfolio_drawdown_cutoff
from backtest import run_backtest
from metrics import calculate_metrics
from paper_trader import PaperTrader


def select_universe(prices, min_avg_volume: int = 100_000, max_universe: int = 200, volume_df=None):
    # If volume_df provided, apply liquidity filter, else use available tickers
    if volume_df is None:
        tickers = list(prices.columns)[:max_universe]
        return tickers
    avg_vol = volume_df.rolling(20).mean().iloc[-1].dropna()
    eligible = avg_vol[avg_vol >= min_avg_volume].sort_values(ascending=False)
    return list(eligible.index[:max_universe])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "paper"], default="backtest")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        logger.error("Config file not found: %s", cfg_path)
        return
    cfg = yaml.safe_load(open(cfg_path))

    # sample tickers; in production you'd feed a dynamic universe source
    sample_universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "IBM", "INTC", "TSLA"]

    provider = cfg.get("data", {}).get("provider", "yfinance")
    start = "2018-01-01"
    end = "2023-12-31"

    prices = fetch_historical(sample_universe, start, end, provider=provider)
    if prices.empty:
        logger.error("No price data available; aborting")
        return

    # compute features
    features = compute_features(prices)

    # optional ML: train on historical returns with walk-forward caution (simplified)
    ml_model = None
    if cfg.get("strategy", {}).get("use_ml", False):
        logger.info("Training ML classifier on historical data (simplified)")
        # target: next-day returns
        target = features["close"].pct_change().shift(-1)
        try:
            ml_model = train_ml_classifier(features, target)
        except Exception:
            logger.exception("ML training failed; continuing without ML")

    signals = generate_signals(features, top_n=cfg.get("strategy", {}).get("top_n", 10), ml_model=ml_model)

    # sizing
    weights = volatility_position_sizing(signals, features.get("atr_14"), cfg.get("risk", {}).get("capital", 10000.0), max_leverage=cfg.get("risk", {}).get("max_leverage", 2.0), max_pos_pct=cfg.get("risk", {}).get("max_position_pct", 0.1))

    if args.mode == "backtest":
        returns, ledger = run_backtest(prices, signals, weights, capital=cfg.get("risk", {}).get("capital", 10000.0), commission=cfg.get("execution", {}).get("commission", 0.0005), slippage=cfg.get("execution", {}).get("slippage", 0.0005))
        metrics = calculate_metrics(returns)
        logger.info("Performance Metrics:")
        for k, v in metrics.items():
            logger.info("%s: %s", k, v)
        print("Sample ledger rows:")
        print(ledger.head())

    else:
        trader = PaperTrader()
        # For demo, execute initial orders for top signals on the last date
        last_signals = signals.iloc[-1]
        last_weights = weights.iloc[-1]
        account = trader.get_account()
        logger.info("Account: %s", account)
        total_equity = cfg.get("risk", {}).get("capital", 10000.0)
        for sym, w in last_weights.items():
            if w == 0:
                continue
            dollars = float(w) * total_equity
            price = float(prices[sym].iloc[-1])
            qty = abs(dollars) / price if price > 0 else 0
            side = "buy" if dollars > 0 else "sell"
            res = trader.submit_order(sym, qty, side=side)
            logger.info("Order result: %s", res)


if __name__ == "__main__":
    main()
