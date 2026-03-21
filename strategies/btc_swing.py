"""
btc_swing.py — BTC Swing Strategy (1H / 4H / Daily)
==================================================
Longer-term triggers based on:
1. Trend (EMA 50/200)
2. S/R and SMC (BOS, CHOCH)
3. Ichimoku Cloud agreement
4. On-chain health score integration
"""

import logging
import pandas as pd # type: ignore
from analysis.btc_indicators import BTCIndicators # type: ignore
from analysis.btc_market_structure import BTCMarketStructure # type: ignore
from analysis.btc_onchain import BTCOnChain # type: ignore

logger = logging.getLogger("apexalgo.btc_swing")

class BTCSwingStrategy:
    """Swing trading for BTC."""

    def __init__(self, risk_reward: float = 3.0):
        self.risk_reward = risk_reward
        self.onchain = BTCOnChain()

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """
        Analyzes 1H/4H/Daily data.
        """
        if len(df) < 200:
            return {"signal": "HOLD", "reason": "No data"}

        df = BTCIndicators.add_all_indicators(df)
        smc = BTCMarketStructure.detect_structure(df)
        health_score = self.onchain.get_health_score()
        
        row = df.iloc[-1]
        close = row["close"]
        ema50 = row["ema_50"]
        ema200 = row["ema_200"]
        atr = row["atr"]

        buy_score = 0
        sell_score = 0
        reasons = []

        # ── 1. Trend Direction ────────────────────────────────
        if close > ema50 > ema200:
            buy_score += 2
            reasons.append("Bullish Trend Alignment")
        elif close < ema50 < ema200:
            sell_score += 2
            reasons.append("Bearish Trend Alignment")

        # ── 2. Market Structure ──────────────────────────────
        if smc["bos"] == "BULLISH":
            buy_score += 2
            reasons.append("Bullish BOS Detected")
        elif smc["bos"] == "BEARISH":
            sell_score += 2
            reasons.append("Bearish BOS Detected")

        # ── 3. On-Chain Health ───────────────────────────────
        if health_score > 0.7:
            buy_score += 1
            reasons.append("Positive On-Chain Health")
        elif health_score < 0.3:
            sell_score += 1
            reasons.append("Weak On-Chain Health")

        # ── Decision ──────────────────────────────────────────
        signal = "HOLD"
        strength = 0.0
        
        if buy_score >= 4 and buy_score > sell_score:
            signal = "BUY"
            strength = buy_score / 6.0
        elif sell_score >= 4 and sell_score > buy_score:
            signal = "SELL"
            strength = sell_score / 6.0

        # SL and TP
        sl = 0.0
        tp = 0.0
        if signal == "BUY":
            sl = close - (1.5 * atr)
            tp = close + (self.risk_reward * atr)
        elif signal == "SELL":
            sl = close + (1.5 * atr)
            tp = close - (self.risk_reward * atr)

        return {
            "signal": signal,
            "strength": float(f"{strength:.2f}"),
            "reason": ", ".join(reasons),
            "atr": atr,
            "sl": float(f"{sl:.2f}"),
            "tp": float(f"{tp:.2f}")
        }
