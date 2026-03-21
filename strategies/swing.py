"""
swing.py — Swing Strategy (H1 / H4)
======================================
Signals generated from: Multi-timeframe trend, ADX strength, RSI, S/R zones.
Returns: 'BUY', 'SELL', or 'HOLD'
"""

import logging
import os
import pandas as pd # type: ignore
import ta # type: ignore
from strategies.smc import SMCEngine # type: ignore

logger = logging.getLogger("apexalgo.swing")


class SwingStrategy:
    """
    Higher timeframe strategy.
    H4 trend direction + H1 entry signals (RSI / Moving Average / Support Resistance).
    """

    def __init__(self, sr_lookback: int = 200):
        self.sr_lookback = sr_lookback
        logger.debug(f"[Swing] Loaded params: ema_trend=50 ema_fast=20")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        df["ema_fast"]  = ta.trend.ema_indicator(close, window=20)  # type: ignore
        df["ema_trend"] = ta.trend.ema_indicator(close, window=50)  # type: ignore
        df["atr"]       = ta.volatility.average_true_range(high, low, close, window=14)  # type: ignore
        df["adx"]       = ta.trend.adx(high, low, close, window=14)  # type: ignore
        df["rsi"]       = ta.momentum.rsi(close, window=14)  # type: ignore
        return df

    def detect_support_resistance(self, df: pd.DataFrame) -> tuple:
        lookback = min(self.sr_lookback, len(df))
        recent = df.tail(lookback)
        return float(recent["low"].min()), float(recent["high"].max())

    def detect_trend(self, df: pd.DataFrame) -> str:
        if "ema_trend" not in df.columns:
            return "SIDEWAYS"
        vals = df["ema_trend"].dropna()
        if len(vals) < 5:
            return "SIDEWAYS"
        slope = (float(vals.iloc[-1]) - float(vals.iloc[-5])) / float(vals.iloc[-5]) * 100
        if slope > 0.05:
            return "UPTREND"
        elif slope < -0.05:
            return "DOWNTREND"
        return "SIDEWAYS"

    def generate_signal(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame = None,   # type: ignore
                        smc_context: dict = None) -> dict:  # type: ignore
        """
        Primary signal from H1, confirmed by H4 trend direction.
        Returns: { 'signal', 'strength', 'reason', 'atr', 'support', 'resistance' }
        """
        if df_h1 is None or len(df_h1) < 100:
            return self._hold("Not enough H1 data")

        df_h1 = self.calculate_indicators(df_h1)
        df_h1.dropna(inplace=True)
        if len(df_h1) < 5:
            return self._hold("Not enough data after dropna")

        row = df_h1.iloc[-1]

        atr_val       = float(row["atr"])
        adx_val       = float(row["adx"])
        rsi_val       = float(row["rsi"])
        ema_fast_val  = float(row["ema_fast"])
        ema_trend_val = float(row["ema_trend"])
        close_val     = float(row["close"])

        # Multi-timeframe trend confirmation from H4
        h4_trend = self.detect_trend(df_h4) if df_h4 is not None else "SIDEWAYS"

        support, resistance = self.detect_support_resistance(df_h1)

        buy_score  = 0
        sell_score = 0
        reasons    = []

        # ── 1. Trend Filter ──────────────────────────────────
        if h4_trend == "UPTREND":
            buy_score += 1
            reasons.append("H4 Trend UP")
        elif h4_trend == "DOWNTREND":
            sell_score += 1
            reasons.append("H4 Trend DOWN")

        # ── 2. EMA Crossover (H1) ───────────────────────────
        if ema_fast_val > ema_trend_val:
            buy_score += 1
            reasons.append("H1 EMA Bullish")
        elif ema_fast_val < ema_trend_val:
            sell_score += 1
            reasons.append("H1 EMA Bearish")

        # ── 3. ADX Strength ──────────────────────────────────
        if adx_val > 25:
            if ema_fast_val > ema_trend_val:
                buy_score += 1
            else:
                sell_score += 1
            reasons.append(f"Strong Trend (ADX={adx_val:.1f})")

        # ── 4. RSI Overbought/Oversold ──────────────────────
        if rsi_val < 30:
            buy_score += 1
            reasons.append("RSI Oversold")
        elif rsi_val > 70:
            sell_score += 1
            reasons.append("RSI Overbought")

        # ── 5. S/R proximity ──────────────────────────────────
        sr_range = resistance - support
        if sr_range > 0:
            prox_support    = (close_val - support) / sr_range
            prox_resistance = (resistance - close_val) / sr_range
            if prox_support < 0.15:
                buy_score += 1
                reasons.append("Near support")
            elif prox_resistance < 0.15:
                sell_score += 1
                reasons.append("Near resistance")

        # ── 6. Smart Money Concepts ──────────────────────────
        smc_ctx = smc_context if smc_context else SMCEngine.get_smc_context(df_h1, close_val)
        if smc_ctx["in_bull_zone"]:
            buy_score += 1
            reasons.append("In Bullish OB/FVG")
        elif smc_ctx["in_bear_zone"]:
            sell_score += 1
            reasons.append("In Bearish OB/FVG")

        # ── 7. Candlestick Patterns (Engulfing) ──────────────
        if len(df_h1) >= 2:
            prev_open  = df_h1["open"].iloc[-2]
            prev_close = df_h1["close"].iloc[-2]
            curr_open  = df_h1["open"].iloc[-1]
            
            if prev_close < prev_open and close_val > curr_open:
                if close_val > prev_open and curr_open < prev_close:
                    buy_score += 1
                    reasons.append("Bullish Engulfing")
                    
            elif prev_close > prev_open and close_val < curr_open:
                if close_val < prev_open and curr_open > prev_close:
                    sell_score += 1
                    reasons.append("Bearish Engulfing")

        # ── Decision ─────────────────────────────────────────
        strength = max(buy_score, sell_score) / 7.0

        if buy_score >= 4 and buy_score > sell_score:
            signal = "BUY"
        elif sell_score >= 4 and sell_score > buy_score:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal":     signal,
            "strength":   float(f"{strength:.2f}"),
            "reason":     ", ".join(reasons) if reasons else "Mixed",
            "atr":        float(f"{atr_val:.5f}"),
            "support":    float(f"{support:.2f}"),
            "resistance": float(f"{resistance:.2f}")
        }

    def _hold(self, reason: str) -> dict:
        return {"signal": "HOLD", "strength": 0.0, "reason": reason, "atr": 0, "support": 0, "resistance": 0}
