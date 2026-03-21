"""
gold_swing.py -- Gold swing trading strategy (H1/H4/Daily)
Rules: EMA50/200 alignment, Ichimoku cloud confirmation, DXY & Yield macro filter,
fundamental score threshold, Fibonacci TP targets, partial close at TP1.
"""

import logging
import pandas as pd
from analysis.gold_indicators import calculate_gold_indicators
from analysis.gold_market_structure import detect_gold_smc, near_ob
from analysis.gold_fundamentals import get_gold_fundamental_score
from analysis.gold_sentiment import get_gold_news_sentiment

logger = logging.getLogger("apexalgo.gold_swing")

FUNDAMENTAL_THRESHOLD = 20     # composite score must exceed this for swing
MAX_HOLD_DAYS         = 10     # auto-close if no movement


class GoldSwingStrategy:
    def __init__(self):
        self.name = "GoldSwing"

    # ── Public API ────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """
        Apply full swing trading rules:
        1. EMA 50/200 alignment (trend direction)
        2. Ichimoku cloud confirmation
        3. DXY must be weakening for longs (macro alignment)
        4. Fundamental score > threshold
        5. Fibonacci TP levels calculated
        """
        empty = {"signal": "HOLD", "strength": 0.0, "reason": "No setup", "atr": 0.0,
                 "sl_distance": 0.0, "tp1": 0.0, "tp2": 0.0}

        if df.empty or len(df) < 200:
            return empty

        df   = calculate_gold_indicators(df)
        smc  = detect_gold_smc(df)
        last = df.iloc[-1]

        close   = float(last["close"])
        atr     = float(last.get("atr", close * 0.005))

        # 1. EMA 50/200 — primary trend filter
        ema50  = float(last.get("ema_50", 0))
        ema200 = float(last.get("ema_200", 0))
        rsi    = float(last.get("rsi", 50))

        if ema50 == 0 or ema200 == 0:
            return {**empty, "reason": "Insufficient EMA data"}

        long_trend  = ema50 > ema200 and close > ema200
        short_trend = ema50 < ema200 and close < ema200

        # 2. Ichimoku cloud confirmation
        span_a = float(last.get("span_a", 0))
        span_b = float(last.get("span_b", 0))
        ichi_bull = close > max(span_a, span_b) if (span_a > 0 and span_b > 0) else True
        ichi_bear = close < min(span_a, span_b) if (span_a > 0 and span_b > 0) else True

        # 3. Supertrend alignment
        st_dir = int(last.get("supertrend_dir", 0))
        st_bull = st_dir == 1
        st_bear = st_dir == -1

        # 4. Fundamental score (DXY + Yield + VIX + ETF)
        fundamentals = get_gold_fundamental_score()
        fund_score   = fundamentals.get("score", 0)
        fund_bias    = fundamentals.get("bias", "NEUTRAL")

        # 5. Sentiment overlay
        sentiment     = get_gold_news_sentiment()
        sent_label    = sentiment.get("label", "NEUTRAL")

        # SMC context
        bull_obs = smc.get("bull_obs", [])
        bear_obs = smc.get("bear_obs", [])
        in_bull_ob = near_ob(close, bull_obs, threshold_pips=5.0)
        in_bear_ob = near_ob(close, bear_obs, threshold_pips=5.0)

        # Fibonacci levels for TP
        fib_272 = float(last.get("fib_ext_1272", close + atr * 5))
        fib_618 = float(last.get("fib_ext_1618", close + atr * 8))
        fib_272_low = float(last.get("fib_ext_1272", close - atr * 5))  # short TPs
        fib_618_low = float(last.get("fib_ext_1618", close - atr * 8))

        # ── Long Signal ───────────────────────────────────────────────────

        signal   = "HOLD"
        strength = 0.0
        reasons  = []

        if long_trend and ichi_bull and fund_score > FUNDAMENTAL_THRESHOLD:
            signal   = "BUY"
            strength = 0.65
            reasons.append(f"EMA50 > EMA200 + Ichimoku Bull + Fund Score {fund_score:.0f}")
            if st_bull:
                strength += 0.10; reasons.append("Supertrend Bull")
            if in_bull_ob:
                strength += 0.10; reasons.append("Near Bull Order Block")
            if sent_label == "BULLISH":
                strength += 0.05; reasons.append("News Sentiment Bullish")
            if rsi < 65:
                reasons.append(f"RSI {rsi:.0f} not overbought")

            return {
                "signal":      signal,
                "strength":    round(min(strength, 1.0), 3),
                "reason":      ", ".join(reasons),
                "atr":         atr,
                "sl_distance": 2.5 * atr,
                "tp1":         fib_272,
                "tp2":         fib_618,
                "tp1_partial": 0.5,  # close 50% at TP1
            }

        # ── Short Signal ──────────────────────────────────────────────────

        if short_trend and ichi_bear and fund_score < -FUNDAMENTAL_THRESHOLD:
            signal   = "SELL"
            strength = 0.65
            reasons.append(f"EMA50 < EMA200 + Ichimoku Bear + Fund Score {fund_score:.0f}")
            if st_bear:
                strength += 0.10; reasons.append("Supertrend Bear")
            if in_bear_ob:
                strength += 0.10; reasons.append("Near Bear Order Block")
            if sent_label == "BEARISH":
                strength += 0.05; reasons.append("News Sentiment Bearish")
            if rsi > 35:
                reasons.append(f"RSI {rsi:.0f} not oversold")

            return {
                "signal":      signal,
                "strength":    round(min(strength, 1.0), 3),
                "reason":      ", ".join(reasons),
                "atr":         atr,
                "sl_distance": 2.5 * atr,
                "tp1":         fib_272_low,
                "tp2":         fib_618_low,
                "tp1_partial": 0.5,
            }

        return {**empty, "reason": f"Trend/Macro not aligned — Fund:{fund_score:.0f} Bias:{fund_bias}"}
