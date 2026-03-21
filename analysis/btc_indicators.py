"""
btc_indicators.py — BTC Technical Indicators
============================================
Custom implementations and wrappers for BTC analysis.
Uses the 'ta' library.
"""

import pandas as pd # type: ignore
import ta # type: ignore
import numpy as np

class BTCIndicators:
    """Provides technical analysis indicators specifically tuned for BTC."""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 1. EMA Suite ──────────────────────────────────────
        df["ema_9"]   = ta.trend.ema_indicator(close, window=9)  # type: ignore
        df["ema_21"]  = ta.trend.ema_indicator(close, window=21)  # type: ignore
        df["ema_50"]  = ta.trend.ema_indicator(close, window=50)  # type: ignore
        df["ema_200"] = ta.trend.ema_indicator(close, window=200)  # type: ignore

        # ── 2. Momentum ───────────────────────────────────────
        df["rsi"] = ta.momentum.rsi(close, window=14)  # type: ignore
        
        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)  # type: ignore
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        # MACD
        macd_ind = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)  # type: ignore
        df["macd"]      = macd_ind.macd()
        df["macd_sig"]  = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()

        # ── 3. Volatility ─────────────────────────────────────
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)  # type: ignore
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()
        
        df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)  # type: ignore

        # ── 4. Intraday / Volume ──────────────────────────────
        # VWAP
        df["vwap"] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=14)  # type: ignore

        # Supertrend
        df["supertrend_ub"] = df["bb_mid"] + (df["atr"] * 3)
        df["supertrend_lb"] = df["bb_mid"] - (df["atr"] * 3)

        return df

    @staticmethod
    def get_trend_direction(df: pd.DataFrame) -> str:
        """Returns 'BULL', 'BEAR', or 'SIDEWAYS'."""
        if len(df) < 200: return "SIDEWAYS"
        last = df.iloc[-1]
        if last["ema_50"] > last["ema_200"] and last["close"] > last["ema_200"]:
            return "BULL"
        elif last["ema_50"] < last["ema_200"] and last["close"] < last["ema_200"]:
            return "BEAR"
        return "SIDEWAYS"
