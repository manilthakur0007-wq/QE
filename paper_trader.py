"""Paper trading connector using Alpaca REST (paper account).

This module will only place live orders if `PAPER_TRADING` env var is true
and Alpaca keys are configured. Otherwise it can simulate fills locally.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE = os.getenv("ALPACA_API_BASE", "https://paper-api.alpaca.markets")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes")

try:
    from alpaca_trade_api.rest import REST
except Exception:
    REST = None


class PaperTrader:
    def __init__(self):
        self.enabled = PAPER_TRADING and REST is not None and ALPACA_KEY and ALPACA_SECRET
        if self.enabled:
            self.client = REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE)
            logger.info("Alpaca paper trader enabled")
        else:
            self.client = None
            logger.warning("Paper trading disabled or Alpaca SDK not available. Running simulated mode.")

    def submit_order(self, symbol: str, qty: float, side: str = "buy", order_type: str = "market", time_in_force: str = "day") -> Dict[str, Any]:
        """Submit order to Alpaca paper account or simulate a fill.

        qty can be fractional if supported by account, otherwise will be floored.
        Returns order dict with timestamp and executed price when available.
        """
        if not self.enabled:
            logger.info("Simulating order %s %s @ qty=%s", side, symbol, qty)
            return {"symbol": symbol, "qty": qty, "side": side, "status": "simulated"}

        try:
            order = self.client.submit_order(symbol=symbol, qty=qty, side=side, type=order_type, time_in_force=time_in_force)
            logger.info("Submitted paper order %s %s qty=%s id=%s", side, symbol, qty, getattr(order, 'id', None))
            return order._raw
        except Exception:
            logger.exception("Failed to submit paper order")
            return {"symbol": symbol, "qty": qty, "side": side, "status": "failed"}

    def get_account(self):
        if not self.enabled:
            return {"status": "simulated", "cash": None}
        return self.client.get_account()._raw


__all__ = ["PaperTrader"]
