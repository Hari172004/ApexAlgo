"""
volatility_filter.py — ATR Volatility Zones
Blocks trades if market is dead (below 0.8x ATR) or dangerously chaotic (above 2.0x ATR).
"""

import logging
import pandas as pd

logger = logging.getLogger("apexalgo.filters.volatility")

class VolatilityFilter:
    def __init__(self, period: int = 20, min_atr_multiplier: float = 0.8, max_atr_multiplier: float = 2.0):
        self.period = period
        self.min_multiplier = min_atr_multiplier
        self.max_multiplier = max_atr_multiplier

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Standard Wilder's ATR calculation.
        Requires high, low, and close columns.
        """
        if len(df) <= period:
            return pd.Series([0]*len(df), index=df.index)

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Using simple moving average of True Range for baseline
        atr = tr.rolling(window=period).mean()
        return atr

    def evaluate_volatility(self, dataframe: pd.DataFrame) -> dict:
        """
        Compares the current short-term volatility (True Range of current candle)
        against the longer-term ATR.
        
        Returns {"status": "GREEN"|"YELLOW"|"RED", "safe": bool, "atr_ratio": float}
        """
        if len(dataframe) < self.period + 1:
            return {"status": "YELLOW", "safe": False, "atr_ratio": 0.0}

        atr_series = self._calculate_atr(dataframe, self.period)
        
        # We compare the current True Range against the moving ATR
        tr1 = dataframe['high'].iloc[-1] - dataframe['low'].iloc[-1]
        tr2 = abs(dataframe['high'].iloc[-1] - dataframe['close'].iloc[-2])
        tr3 = abs(dataframe['low'].iloc[-1]  - dataframe['close'].iloc[-2])
        current_tr = max(tr1, tr2, tr3)
        
        avg_atr = atr_series.iloc[-2] # Uses the fully formed previous average 
        
        if avg_atr <= 0:
            return {"status": "RED", "safe": False, "atr_ratio": 0.0}

        ratio = current_tr / avg_atr

        # Too wild -> RED
        if ratio > self.max_multiplier:
            logger.warning(f"[VOLATILITY] Exteme volatility! ATR Ratio is {ratio:.2f}x (Limit: {self.max_multiplier}x). Trade Blocked.")
            return {"status": "RED", "safe": False, "atr_ratio": ratio}
            
        # Too slow -> YELLOW
        elif ratio < self.min_multiplier:
            logger.warning(f"[VOLATILITY] Dead market. ATR Ratio is {ratio:.2f}x (Min: {self.min_multiplier}x). Trade Blocked.")
            return {"status": "YELLOW", "safe": False, "atr_ratio": ratio}
            
        # Optimal -> GREEN
        else:
            logger.info(f"[VOLATILITY] Optimal Zone. ATR Ratio is {ratio:.2f}x.")
            return {"status": "GREEN", "safe": True, "atr_ratio": ratio}

global_volatility_filter = VolatilityFilter()
