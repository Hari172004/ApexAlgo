"""
btc_market_structure.py — BTC SMC Detection
===========================================
Detects BOS, CHOCH, Order Blocks, and FVGs for Bitcoin.
"""

import pandas as pd # type: ignore
import numpy as np

class BTCMarketStructure:
    """Analyzes BTC price action for Smart Money Concepts."""

    @staticmethod
    def detect_structure(df: pd.DataFrame) -> dict:
        """
        Detects latest BOS, CHOCH, OB, and FVG.
        Returns context dict.
        """
        if len(df) < 50:
            return {"bos": None, "choch": None, "zones": []}

        df = df.tail(100).copy()
        
        # Simple Pivot detection
        df["pivot_high"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
        df["pivot_low"]  = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))

        # Placeholder for complex BOS/CHOCH logic (simplified for now)
        last_high = df[df["pivot_high"]]["high"].tail(2).values
        last_low  = df[df["pivot_low"]]["low"].tail(2).values
        
        bos = None
        if len(last_high) >= 2 and last_high[-1] > last_high[-2]:
            bos = "BULLISH"
        elif len(last_low) >= 2 and last_low[-1] < last_low[-2]:
            bos = "BEARISH"

        # FVG Detection
        fvgs = []
        for i in range(2, len(df)):
            # Bullish FVG
            if df["low"].iloc[i] > df["high"].iloc[i-2]:
                fvgs.append({
                    "type": "BULL_FVG",
                    "top": df["low"].iloc[i],
                    "bottom": df["high"].iloc[i-2],
                    "size": df["low"].iloc[i] - df["high"].iloc[i-2]
                })
            # Bearish FVG
            elif df["high"].iloc[i] < df["low"].iloc[i-2]:
                fvgs.append({
                    "type": "BEAR_FVG",
                    "top": df["low"].iloc[i-2],
                    "bottom": df["high"].iloc[i],
                    "size": df["low"].iloc[i-2] - df["high"].iloc[i]
                })

        # Order Blocks (Last candle before impulsive move)
        obs = []
        # (Simplified: find largest candles and pick the previous one)
        df["body_size"] = (df["close"] - df["open"]).abs()
        mean_body = df["body_size"].mean()
        
        for i in range(1, len(df)):
            if df["body_size"].iloc[i] > 2 * mean_body:
                # Potential OB is the previous candle
                ob_type = "BULL_OB" if df["close"].iloc[i] > df["open"].iloc[i] else "BEAR_OB"
                obs.append({
                    "type": ob_type,
                    "high": df["high"].iloc[i-1],
                    "low": df["low"].iloc[i-1],
                    "age": len(df) - i
                })

        return {
            "bos": bos,
            "fvgs": fvgs[-5:], # latest 5
            "obs": obs[-5:]    # latest 5
        }
