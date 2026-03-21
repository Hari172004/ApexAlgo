"""
correlation.py — Dynamic Inter-market Correlation Engine
==========================================================
Verifies that the bot doesn't trade XAUUSD or BTCUSD against 
the macroeconomic sentiment of the US Dollar Index (DXY).

If DXY is spiking up -> Block XAUUSD BUYS.
If DXY is dumping -> Block XAUUSD SELLS.
"""

import pandas as pd
from history_store import HistoryStore
import logging

logger = logging.getLogger("apexalgo.correlation")

class CorrelationEngine:
    def __init__(self, history_store: HistoryStore):
        self.store = history_store
    
    def check_correlation_guard(self, symbol: str, direction: str) -> dict:
        """
        Calculates the 4-Hour trend of the DXY string.
        Returns {"safe": True, "reason": ""} if safe to trade.
        Returns {"safe": False, "reason": "DXY trending opposite..."} if blocked.
        """
        # We only really care about correlation blocks for XAUUSD and BTCUSD
        if symbol not in ["XAUUSD", "BTCUSD"]:
            return {"safe": True, "reason": f"No fixed correlation guard for {symbol}"}

        try:
            # We use DX-Y.NYB which is the Yahoo Finance ticker for the US Dollar Index
            dxy_candles = self.store.get_candles_json("DX-Y.NYB", "H4", limit=50)
            
            # If we don't have DXY data, we assume it's safe (fail-open) to avoid blocking the bot entirely
            if not dxy_candles or len(dxy_candles) < 20:
                # Let's try to fetch it if it's missing
                self.store.fetch_and_cache("DX-Y.NYB", "H4")
                dxy_candles = self.store.get_candles_json("DX-Y.NYB", "H4", limit=50)
                if not dxy_candles or len(dxy_candles) < 20:
                    return {"safe": True, "reason": "DXY Data unavailable, ignoring correlation guard."}

            df = pd.DataFrame(dxy_candles)
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Simple Trend Determination
            # Is the current price above the EMA20 and are recent candles strongly up?
            current_close = df['close'].iloc[-1]
            ema20 = df['ema20'].iloc[-1]
            
            dxy_is_bullish = current_close > ema20 and df['close'].iloc[-1] > df['close'].iloc[-5]
            dxy_is_bearish = current_close < ema20 and df['close'].iloc[-1] < df['close'].iloc[-5]

            # Enforce the logic: DXY and XAUUSD are inversely correlated
            if symbol == "XAUUSD":
                if direction == "BUY" and dxy_is_bullish:
                    return {"safe": False, "reason": "DXY is strongly Bullish. Blocking XAUUSD BUY."}
                elif direction == "SELL" and dxy_is_bearish:
                    return {"safe": False, "reason": "DXY is strongly Bearish. Blocking XAUUSD SELL."}
            
            # BTCUSD correlation is weaker, but generally inverse as well
            if symbol == "BTCUSD":
                if direction == "BUY" and dxy_is_bullish:
                    return {"safe": False, "reason": "DXY is strongly Bullish. Blocking BTCUSD BUY."}
                elif direction == "SELL" and dxy_is_bearish:
                    return {"safe": False, "reason": "DXY is strongly Bearish. Blocking BTCUSD SELL."}

            return {"safe": True, "reason": f"DXY momentum supports {symbol} {direction}."}

        except Exception as e:
            logger.error(f"[Correlation] Error checking DXY: {e}")
            return {"safe": True, "reason": f"Fallback: Error parsing DXY - {e}"}
