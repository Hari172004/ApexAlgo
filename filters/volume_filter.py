"""
volume_filter.py — Volume Confirmation Filter
Requires the current entry candle volume to be 1.5x the rolling 20-candle average.
Prevents entry during dead/flat markets.
"""

import logging
import pandas as pd

logger = logging.getLogger("apexalgo.filters.volume")

class VolumeSpikeFilter:
    def __init__(self, multiplier: float = 1.5, lookback: int = 20):
        self.multiplier = multiplier
        self.lookback = lookback

    def check_volume_spike(self, dataframe: pd.DataFrame) -> bool:
        """
        Receives OHLCV dataframe, typically M5 or M15.
        Returns True if the current (or last closed) candle's volume is >= 1.5x of the 20 EMA/SMA volume.
        """
        if len(dataframe) < self.lookback + 1:
            logger.debug("[VOLUME] Not enough data to calculate volume average.")
            return False

        # Calculate the simple moving average of volume for the intended lookback
        # Shift(1) to not include the current live candle in its own average calculation
        dataframe['vol_sma'] = dataframe['volume'].shift(1).rolling(window=self.lookback).mean()

        current_volume = dataframe['volume'].iloc[-1]
        avg_volume = dataframe['vol_sma'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume <= 0:
            return False

        ratio = current_volume / avg_volume

        if ratio >= self.multiplier:
            logger.info(f"[VOLUME] SPIKE CONFIRMED! Current Vol: {current_volume:.2f} | Avg: {avg_volume:.2f} | Ratio: {ratio:.2f}x")
            return True
        else:
            logger.debug(f"[VOLUME] No spike. Ratio {ratio:.2f}x < {self.multiplier}x")
            return False

global_volume_filter = VolumeSpikeFilter()
