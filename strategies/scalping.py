"""
scalping.py — Scalping Strategy (1m / 5m)
==========================================
Signals generated from: RSI, EMA crossover, Bollinger Bands, MACD.
Uses the `ta` library (stable PyPI package, Python 3.9 compatible).
Returns: 'BUY', 'SELL', or 'HOLD'
"""

import logging
import os
import json
import pandas as pd # type: ignore
import ta # type: ignore
from strategies.smc import SMCEngine # type: ignore

logger = logging.getLogger("apexalgo.scalping")

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized_params.json")


class ScalpingStrategy:
    """
    Fast 1m/5m scalping strategy.
    Combines RSI + EMA crossover + Bollinger Bands + MACD.
    All four must loosely agree (3/4 consensus) before a signal is issued.
    """

    def __init__(self,
                 ema_fast: int  = 9,
                 ema_slow: int  = 21,
                 rsi_period: int = 14,
                 rsi_ob: float   = 70.0,
                 rsi_os: float   = 30.0,
                 bb_period: int  = 20,
                 bb_std: float   = 2.0):
        # Attempt to load optimized params
        opt = {}
        if os.path.exists(PARAMS_PATH):
            try:
                with open(PARAMS_PATH, "r") as f:
                    opt = json.load(f).get("scalping", {})
            except Exception as e:
                logger.warning(f"Error reading optimized_params.json: {e}")

        self.ema_fast   = opt.get("ema_fast", ema_fast)
        self.ema_slow   = opt.get("ema_slow", ema_slow)
        self.rsi_period = opt.get("rsi_period", rsi_period)
        self.rsi_ob     = opt.get("rsi_ob", rsi_ob)
        self.rsi_os     = opt.get("rsi_os", rsi_os)
        self.bb_period  = opt.get("bb_period", bb_period)
        self.bb_std     = opt.get("bb_std", bb_std)
        logger.debug(f"[Scalp] Loaded params: ema={self.ema_fast}/{self.ema_slow} rsi={self.rsi_os}/{self.rsi_ob}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicator columns to a copy of df and return it."""
        df = df.copy()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # EMA
        df["ema_fast"] = ta.trend.ema_indicator(close, window=self.ema_fast)  # type: ignore
        df["ema_slow"] = ta.trend.ema_indicator(close, window=self.ema_slow)  # type: ignore

        # RSI
        df["rsi"] = ta.momentum.rsi(close, window=self.rsi_period)  # type: ignore

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=self.bb_period, window_dev=self.bb_std)  # type: ignore
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()

        # MACD
        macd_ind       = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)  # type: ignore
        df["macd"]     = macd_ind.macd()
        df["macd_sig"] = macd_ind.macd_signal()
        df["macd_hist"]= macd_ind.macd_diff()

        # ATR for position sizing
        df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)  # type: ignore

        return df

    def generate_signal(self, df: pd.DataFrame, smc_context: dict = None) -> dict:  # type: ignore
        """
        Analyse latest candle data and return a signal dict:
            { 'signal': 'BUY'|'SELL'|'HOLD', 'strength': 0–1,
              'reason': str, 'atr': float, 'rsi': float }
        """
        if df is None or len(df) < 50:
            return self._hold("Not enough data")

        df = self.calculate_indicators(df)
        df.dropna(inplace=True)
        if len(df) < 5:
            return self._hold("Indicators need more bars")

        row = df.iloc[-1]

        ema_fast_val  = row["ema_fast"]
        ema_slow_val  = row["ema_slow"]
        rsi_val       = row["rsi"]
        bb_upper_val  = row["bb_upper"]
        bb_lower_val  = row["bb_lower"]
        macd_hist_val = row["macd_hist"]
        close_val     = row["close"]
        atr_val       = row["atr"]

        buy_score  = 0
        sell_score = 0
        reasons    = []

        # ── 1. EMA crossover ─────────────────────────────────
        if ema_fast_val > ema_slow_val:
            buy_score += 1
            reasons.append("EMA bullish")
        else:
            sell_score += 1
            reasons.append("EMA bearish")

        # ── 2. RSI ───────────────────────────────────────────
        if rsi_val < self.rsi_os:
            buy_score += 1
            reasons.append(f"RSI oversold ({rsi_val:.1f})")
        elif rsi_val > self.rsi_ob:
            sell_score += 1
            reasons.append(f"RSI overbought ({rsi_val:.1f})")

        # ── 3. Bollinger Band touch ───────────────────────────
        if close_val <= bb_lower_val * 1.001:
            buy_score += 1
            reasons.append("Price at BB lower")
        elif close_val >= bb_upper_val * 0.999:
            sell_score += 1
            reasons.append("Price at BB upper")

        # ── 4. MACD histogram direction ───────────────────────
        if macd_hist_val > 0:
            buy_score += 1
            reasons.append("MACD bullish")
        elif macd_hist_val < 0:
            sell_score += 1
            reasons.append("MACD bearish")

        # ── 5. Smart Money Concepts (SMC) ─────────────────────
        smc_ctx = smc_context if smc_context else SMCEngine.get_smc_context(df, close_val)
        if smc_ctx["in_bull_zone"]:
            buy_score += 1
            reasons.append("In Bullish FVG/OB")
        elif smc_ctx["in_bear_zone"]:
            sell_score += 1
            reasons.append("In Bearish FVG/OB")

        # ── 6. Candlestick Patterns (Engulfing) ───────────────
        if len(df) >= 2:
            prev_open  = df["open"].iloc[-2]
            prev_close = df["close"].iloc[-2]
            curr_open  = df["open"].iloc[-1]
            
            if prev_close < prev_open and close_val > curr_open: # Prev Bearish, Curr Bullish
                if close_val > prev_open and curr_open < prev_close:
                    buy_score += 1
                    reasons.append("Bullish Engulfing")
                    
            elif prev_close > prev_open and close_val < curr_open: # Prev Bullish, Curr Bearish
                if close_val < prev_open and curr_open > prev_close:
                    sell_score += 1
                    reasons.append("Bearish Engulfing")

        # ── Decision Logic ────────────────────────────────────
        # Consensus threshold: at least 3/6 score for a signal
        final_signal = "HOLD"
        strength = 0.0
        
        if buy_score >= 3 and buy_score > sell_score:
            final_signal = "BUY"
            strength = buy_score / 6.0
        elif sell_score >= 3 and sell_score > buy_score:
            final_signal = "SELL"
            strength = sell_score / 6.0

        return {
            "signal":   final_signal,
            "strength": float(f"{strength:.2f}"),
            "reason":   ", ".join(reasons) if reasons else "Mixed",
            "atr":      float(f"{atr_val:.5f}"),
            "rsi":      float(f"{rsi_val:.1f}")
        }

    def _hold(self, reason: str) -> dict:
        return {"signal": "HOLD", "strength": 0.0, "reason": reason, "atr": 0, "rsi": 50}
