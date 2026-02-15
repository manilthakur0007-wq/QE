"""Data layer: historical ingestion and async streaming adapters.

Supports multiple providers via a simple adapter pattern. Uses environment
variables and `config.yaml` to choose providers. Provides data cleaning,
missing value handling and corporate action adjustment (when provider
supports it).

This module intentionally provides a clear interface that other modules
call: `fetch_historical(universe, start, end)` and `stream_prices(universe)`.
"""
from __future__ import annotations

import os
import asyncio
import logging
from typing import List, AsyncIterator

import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE = os.getenv("ALPACA_API_BASE", "https://paper-api.alpaca.markets")

try:
    import yfinance as yf  # type: ignore[import]
except Exception:
    yf = None

try:
    from alpaca_trade_api.rest import REST as AlpacaREST  # type: ignore[import]
except Exception:
    AlpacaREST = None


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Forward-fill then backfill for short gaps, drop rows with all-nans
    df = df.ffill(limit=3).bfill(limit=3)
    df = df.dropna(how="all")
    return df


def fetch_historical(tickers: List[str], start: str, end: str, provider: str = "yfinance") -> pd.DataFrame:
    """Fetch historical OHLCV close prices for the `tickers` between dates.

    Returns a DataFrame indexed by date with columns as tickers (close price).
    """
    provider = (provider or "yfinance").lower()
    if provider == "alpaca" and AlpacaREST is not None and ALPACA_KEY and ALPACA_SECRET:
        api = AlpacaREST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE)
        frames = []
        for t in tickers:
            try:
                barset = api.get_bars(t, "1Day", start=start, end=end, adjustment="all")
                df = barset.df
                if df is None or df.empty:
                    logger.warning("No data for %s from Alpaca", t)
                    continue
                # Safely remove timezone if present
                try:
                    if getattr(df.index, "tz", None) is not None:
                        df = df.tz_convert(None)
                except Exception:
                    try:
                        df.index = df.index.tz_localize(None)
                    except Exception:
                        pass
                df = df.set_index(pd.DatetimeIndex(df.index.date))
                # prefer lowercase close if present, fallback to first column
                close_col = "close" if "close" in df.columns else df.columns[0]
                frames.append(df[close_col].rename(t))
            except Exception as e:
                logger.exception("Alpaca fetch error for %s: %s", t, e)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, axis=1)
        return _clean_ohlcv(data)

    # fallback: yfinance
    if yf is None:
        # Provide a deterministic synthetic fallback so the engine can run
        # for demos/CI even when `yfinance` is not installed. This is NOT
        # real market data — users should install `yfinance` or configure
        # Alpaca for real historical prices.
        logger.warning("yfinance not available — generating synthetic price data for demo")
        dates = pd.bdate_range(start=start, end=end)
        rng = np.random.default_rng(seed=42)
        data = {}
        for t in tickers:
            # random walk starting at 100
            steps = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
            prices = 100 * np.exp(np.cumsum(steps))
            data[t] = prices
        closes = pd.DataFrame(data, index=dates)
        closes.index.name = "date"
        closes = _clean_ohlcv(closes)
        return closes

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    # yfinance returns a DataFrame; when multiple tickers it uses a MultiIndex columns
    closes = None
    if isinstance(data, pd.DataFrame):
        cols = data.columns
        if isinstance(cols, pd.MultiIndex):
            # MultiIndex: first level OHLCV, second level tickers
            if "Close" in cols.levels[0]:
                closes = data["Close"]
            elif "close" in cols.levels[0]:
                closes = data["close"]
            else:
                # fallback: try to find any numeric column per ticker
                try:
                    closes = data.xs("Close", level=0, axis=1)
                except Exception:
                    # take the first level for each ticker
                    closes = data.iloc[:, data.columns.get_level_values(0) == data.columns.get_level_values(0)[0]]
        else:
            # single-level columns, could already be close prices or OHLCV
            if "Close" in cols:
                closes = data["Close"]
            elif "close" in cols:
                closes = data["close"]
            else:
                closes = data
    else:
        closes = pd.DataFrame(data)
    # ensure DataFrame with single level columns
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    closes = closes.dropna(how="all")
    closes = _clean_ohlcv(closes)
    return closes


async def stream_prices(universe: List[str]) -> AsyncIterator[dict]:
    """Async generator that yields price ticks for symbols in `universe`.

    If Alpaca websocket is configured this will connect; otherwise it yields
    simulated end-of-day ticks by sleeping and reading latest historical.
    """
    # If Alpaca websocket available, connect (best-effort). Otherwise simulate.
    if AlpacaREST is not None and ALPACA_KEY and ALPACA_SECRET:
        # Simple polling-based async stream using REST as a fallback.
        api = AlpacaREST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE)
        logger.info("Starting simple polling stream via Alpaca REST")
        while True:
            try:
                for sym in universe:
                    bars = api.get_bars(sym, "1Min", limit=1)
                    if bars and bars.df is not None and not bars.df.empty:
                        row = bars.df.iloc[-1]
                        yield {"symbol": sym, "ts": row.name, "price": float(row.close)}
                await asyncio.sleep(1)
            except Exception:
                logger.exception("Streaming poll error, retrying in 2s")
                await asyncio.sleep(2)
    else:
        # Simulated streaming: yield EOD closes from yfinance historical in small batches
        logger.info("Alpaca not configured; streaming simulated prices from yfinance")
        # use last 60 trading days for a simple demo if available
        df = fetch_historical(universe, pd.Timestamp.today() - pd.Timedelta(days=90), pd.Timestamp.today())
        if df.empty:
            logger.warning("No historical prices available for simulated streaming")
            return
        for ts, row in df.iterrows():
            for sym in universe:
                price = None
                try:
                    price = float(row[sym])
                except Exception:
                    price = np.nan
                yield {"symbol": sym, "ts": pd.Timestamp(ts), "price": price}
            await asyncio.sleep(0)

