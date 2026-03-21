"""
spread_filter.py — Strict Spread Limiter
Blocks trades unconditionally if spread spikes beyond tolerance (Gold <= 20 pips, BTC <= 0.05%).
"""

import logging

logger = logging.getLogger("apexalgo.filters.spread")

class SpreadFilter:
    def __init__(self, gold_max_pips: int = 20, btc_max_pct: float = 0.05):
        self.gold_max_pips = gold_max_pips
        self.btc_max_pct = btc_max_pct

    def check_spread(self, symbol: str, bid: float, ask: float) -> bool:
        """
        Returns True if the spread is SAFE to trade. Returns False if it is too wide.
        """
        if bid <= 0 or ask <= 0:
            return False

        spread = ask - bid
        is_xau = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        is_btc = "BTC" in symbol.upper() or "BITCOIN" in symbol.upper()

        if is_xau:
            # Gold MT5 tick resolution is typically 1 pip = 0.01
            # E.g., if ask=2500.50, bid=2500.25 -> spread = 0.25 (25 pips)
            spread_pips = spread * 100
            if spread_pips > self.gold_max_pips:
                logger.warning(f"[SPREAD] {symbol} Spread spiked! Current: {spread_pips:.1f} pips. Limit: {self.gold_max_pips}. Blocked.")
                return False
            else:
                logger.debug(f"[SPREAD] {symbol} Spread safe: {spread_pips:.1f} pips.")
                return True

        if is_btc:
            # BTC spread logic is based on percentage of asset price
            # E.g., if price = $100,000, 0.05% = $50 maximum spread
            spread_pct = (spread / ask) * 100
            if spread_pct > self.btc_max_pct:
                logger.warning(f"[SPREAD] {symbol} Spread spiked! Current: {spread_pct:.3f}%. Limit: {self.btc_max_pct}%. Blocked.")
                return False
            else:
                logger.debug(f"[SPREAD] {symbol} Spread safe: {spread_pct:.3f}%.")
                return True

        # Default naive safeguard for unknown assets
        if (spread / ask) * 100 > 0.1:
            return False
            
        return True

global_spread_filter = SpreadFilter()
