"""
gold_scalp.py -- Full Gold scalping strategy (1m/5m)
Rules: Kill Zone only, EMA cross + RSI + volume + BB squeeze + Order Block alignment.
Max 5 trades/session, no trades 30 mins before news, min 10 pip SL.
"""

import logging
import pandas as pd
from analysis.gold_indicators import calculate_gold_indicators
from analysis.gold_market_structure import detect_gold_smc, near_ob, near_fvg
from analysis.gold_sessions import is_gold_scalp_time, get_current_gold_session

logger = logging.getLogger("apexalgo.gold_scalp")

MAX_SCALPS_PER_SESSION = 5


class GoldScalpStrategy:
    def __init__(self):
        self.name = "GoldScalp"
        self._session_trades: dict = {}   # session_key → trade count

    # ── Public API ────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """
        Apply all scalping rules and return a signal dict.
        Returns: {signal, strength, reason, atr, sl_distance, tp_distance}
        """
        empty = {"signal": "HOLD", "strength": 0.0, "reason": "No setup", "atr": 0.0,
                 "sl_distance": 0.0, "tp_distance": 0.0}

        if df.empty or len(df) < 50:
            return empty

        # 1. Kill Zone gating (no Asian session scalps, no LBMA)
        if not is_gold_scalp_time():
            sess = get_current_gold_session()
            reason = "LBMA fix window" if sess["is_lbma_fix"] else "Outside Kill Zone"
            return {**empty, "reason": reason}

        # 2. Session trade limit
        session_key = self._session_key()
        trades_this_session = self._session_trades.get(session_key, 0)
        if trades_this_session >= MAX_SCALPS_PER_SESSION:
            return {**empty, "reason": f"Max {MAX_SCALPS_PER_SESSION} scalps hit this session"}

        # 3. Compute indicators
        df = calculate_gold_indicators(df)
        smc = detect_gold_smc(df)

        last  = df.iloc[-1]
        prev  = df.iloc[-2]

        atr   = float(last.get("atr", 0))
        close = float(last["close"])

        # 4. Bollinger Band squeeze check (no scalp if BB flat)
        bb_squeeze = bool(last.get("bb_squeeze", False))
        if bb_squeeze:
            return {**empty, "reason": "BB squeeze flat — waiting for breakout"}

        # 5. ATR spike guard (no scalp if volatility extreme)
        atr_spike = bool(last.get("atr_spike", False))
        if atr_spike:
            return {**empty, "reason": "ATR spike detected — paused"}

        # 6. EMA crossover
        ema9_now  = float(last.get("ema_9", 0))
        ema21_now = float(last.get("ema_21", 0))
        ema9_prev = float(prev.get("ema_9", 0))
        ema21_prev= float(prev.get("ema_21", 0))
        ema_cross_up   = ema9_prev <= ema21_prev and ema9_now > ema21_now
        ema_cross_down = ema9_prev >= ema21_prev and ema9_now < ema21_now

        # 7. RSI filter (40-60 = neutral/momentum range for entry)
        rsi = float(last.get("rsi", 50))
        rsi_ok = 38 <= rsi <= 62

        # 8. MACD confirmation
        macd_hist = float(last.get("macd_hist", 0))
        macd_bull = macd_hist > 0
        macd_bear = macd_hist < 0

        # 9. Volume spike
        vol       = float(last.get("volume", 0))
        vol_avg   = float(df["volume"].tail(20).mean())
        vol_spike = vol > vol_avg * 1.3

        # 10. SMC alignment
        bull_obs = smc.get("bull_obs", [])
        bear_obs = smc.get("bear_obs", [])
        in_bull_ob = near_ob(close, bull_obs)
        in_bear_ob = near_ob(close, bear_obs)
        in_fvg     = near_fvg(close, smc.get("fvgs", []))

        # ── Build signal ──────────────────────────────────────────────────

        signal   = "HOLD"
        strength = 0.0
        reasons  = []

        if ema_cross_up and rsi_ok and macd_bull:
            signal   = "BUY"
            strength = 0.60
            reasons.append("EMA9 > EMA21 + RSI OK + MACD Bull")
            if vol_spike:
                strength += 0.10; reasons.append("Volume spike")
            if in_bull_ob or in_fvg:
                strength += 0.15; reasons.append("Inside Bull OB/FVG")

        elif ema_cross_down and rsi_ok and macd_bear:
            signal   = "SELL"
            strength = 0.60
            reasons.append("EMA9 < EMA21 + RSI OK + MACD Bear")
            if vol_spike:
                strength += 0.10; reasons.append("Volume spike")
            if in_bear_ob or in_fvg:
                strength += 0.15; reasons.append("Inside Bear OB/FVG")

        if signal != "HOLD":
            # Minimum SL check — 10 pips (0.10 on XAUUSD)
            sl_dist = max(1.0 * atr, 1.0)   # 1x ATR or minimum $1
            tp_dist = sl_dist * 2.0           # 1:2 RR minimum

            return {
                "signal":      signal,
                "strength":    round(min(strength, 1.0), 3),
                "reason":      ", ".join(reasons),
                "atr":         atr,
                "sl_distance": sl_dist,
                "tp_distance": tp_dist,
                "rsi":         rsi,
                "kill_zone":   get_current_gold_session()["active_kz"],
            }

        return {**empty, "reason": "No EMA cross + RSI + MACD alignment"}

    def record_trade(self):
        """Call this when a scalp trade is placed to track session limit."""
        key = self._session_key()
        self._session_trades[key] = self._session_trades.get(key, 0) + 1

    def _session_key(self) -> str:
        from datetime import date
        sess = get_current_gold_session()
        return f"{date.today()}_{sess['active_kz']}"
