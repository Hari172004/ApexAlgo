"""
btc_scalp.py — BTC Scalping Strategy (1m / 5m)
=============================================
High-frequency triggers based on:
1. EMA 9/21 cross
2. RSI 40-60 level
3. Volume spikes
4. SMC (FVG/OB) alignment
"""

import logging
import pandas as pd # type: ignore
from analysis.btc_indicators import BTCIndicators # type: ignore
from analysis.btc_market_structure import BTCMarketStructure # type: ignore

logger = logging.getLogger("apexalgo.btc_scalp")

class BTCScalpStrategy:
    """Fast scalping for BTC."""

    def __init__(self, risk_reward: float = 2.0):
        self.risk_reward = risk_reward

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """
        Analyzes 1m/5m data for entry.
        Returns: { 'signal', 'strength', 'reason', 'atr', 'sl', 'tp' }
        """
        if len(df) < 50:
            return {"signal": "HOLD", "reason": "No data"}

        df = BTCIndicators.add_all_indicators(df)
        smc = BTCMarketStructure.detect_structure(df)
        
        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        close = row["close"]
        ema9  = row["ema_9"]
        ema21 = row["ema_21"]
        rsi   = row["rsi"]
        atr   = row["atr"]
        vol   = row["volume"]
        avg_vol = df["volume"].tail(20).mean()

        buy_score = 0
        sell_score = 0
        reasons = []

        # ── 1. EMA Cross ──────────────────────────────────────
        if prev_row["ema_9"] <= prev_row["ema_21"] and ema9 > ema21:
            buy_score += 2
            reasons.append("EMA9/21 Bullish Cross")
        elif prev_row["ema_9"] >= prev_row["ema_21"] and ema9 < ema21:
            sell_score += 2
            reasons.append("EMA9/21 Bearish Cross")

        # ── 2. RSI Filter (40-60 for momentum) ────────────────
        if 40 < rsi < 60:
            buy_score += 1
            sell_score += 1
            reasons.append("RSI in momentum zone")

        # ── 3. Volume Spike ───────────────────────────────────
        if vol > 1.5 * avg_vol:
            if ema9 > ema21:
                buy_score += 1
            else:
                sell_score += 1
            reasons.append("Volume Spike Detected")

        # ── 4. SMC Confirmation ──────────────────────────────
        for ob in smc["obs"]:
            if ob["type"] == "BULL_OB" and ob["low"] <= close <= ob["high"]:
                buy_score += 2
                reasons.append("Reacting to Bullish OB")
            elif ob["type"] == "BEAR_OB" and ob["low"] <= close <= ob["high"]:
                sell_score += 2
                reasons.append("Reacting to Bearish OB")

        # ── Decision ──────────────────────────────────────────
        signal = "HOLD"
        strength = 0.0
        
        if buy_score >= 4 and buy_score > sell_score:
            signal = "BUY"
            strength = buy_score / 6.0
        elif sell_score >= 4 and sell_score > buy_score:
            signal = "SELL"
            strength = sell_score / 6.0

        # SL and TP calculation
        sl = 0.0
        tp = 0.0
        if signal == "BUY":
            sl = close - (1.0 * atr)
            tp = close + (self.risk_reward * atr)
        elif signal == "SELL":
            sl = close + (1.0 * atr)
            tp = close - (self.risk_reward * atr)

        return {
            "signal": signal,
            "strength": float(f"{strength:.2f}"),
            "reason": ", ".join(reasons),
            "atr": atr,
            "sl": float(f"{sl:.2f}"),
            "tp": float(f"{tp:.2f}")
        }
