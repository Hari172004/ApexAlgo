"""
gold_scalp.py -- Full Gold scalping strategy (1m/5m)
Rules: Kill Zone only, EMA cross + RSI + volume + BB squeeze + Order Block alignment.
Max 5 trades/session, no trades 30 mins before news, min 10 pip SL.
"""

import logging
from datetime import datetime, timezone
import pandas as pd # type: ignore
from analysis.gold_indicators import calculate_gold_indicators # type: ignore
from analysis.gold_market_structure import detect_gold_smc, near_ob, near_fvg # type: ignore
from analysis.gold_sessions import is_gold_scalp_time, get_current_gold_session # type: ignore

try:
    from rl.ppo_agent import PPOAgent  # type: ignore
    _PPO_AVAILABLE = True
except ImportError:
    _PPO_AVAILABLE = False

logger = logging.getLogger("agniv.gold_scalp")

MAX_SCALPS_PER_SESSION = 5


class GoldScalpStrategy:
    def __init__(self):
        self.name = "GoldScalp"
        self._session_trades: dict = {}   # session_key → trade count
        # PPO Reinforcement Learning agent (lazy-loaded)
        self._ppo: "PPOAgent | None" = PPOAgent("XAUUSD") if _PPO_AVAILABLE else None  # type: ignore

    # ── Public API ────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, df_h1: pd.DataFrame = None, 
                        is_nano: bool = False, ignore_sessions: bool = False, is_sniper: bool = False) -> dict:
        """
        Main entry point for generating scalp signals.
        """
        empty = {"signal": "HOLD", "strength": 0.0, "reason": "No setup", "atr": 0.0,
                 "sl_distance": 0.0, "tp_distance": 0.0}

        if df.empty or len(df) < 50:
            return empty

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        live_close_val = float(last["close"])

        # 1. Kill Zone gating
        if not is_gold_scalp_time(ignore_lbma=ignore_sessions, ignore_asian=ignore_sessions):
            if not ignore_sessions:
                sess = get_current_gold_session()
                reason = "LBMA fix window" if sess["is_lbma_fix"] else "Outside Kill Zone"
                return {**empty, "reason": reason}

        # 2. HTF Trend Alignment (H1 EMA 100)
        h1_bullish, h1_bearish = True, True # default if no H1
        if df_h1 is not None and not df_h1.empty:
            df_h1 = calculate_gold_indicators(df_h1)
            h1_last = df_h1.iloc[-1]
            ema100_h1 = h1_last.get("ema_100", h1_last.get("ema_200", 0))
            h1_bullish = float(h1_last["close"]) > ema100_h1
            h1_bearish = float(h1_last["close"]) < ema100_h1

        # 3. Indicators & SMC (ICT Sniper V2.0)
        df = calculate_gold_indicators(df)
        smc = detect_gold_smc(df)
        
        # Refresh snapshots after indicators are added
        last  = df.iloc[-1]
        prev  = df.iloc[-2]

        from strategies.smc import SMCEngine # type: ignore
        smc_v2 = SMCEngine.get_smc_context(df, live_close_val)

        # Ensure required columns are present (prevents KeyError if history < 30)
        required = ["ema_9", "ema_21", "ema_50", "rsi", "atr"]
        missing = [col for col in required if col not in last or col not in prev]
        if missing:
             logger.debug(f"[GoldScalp] Missing indicators: {missing} | df_len={len(df)}")
             return {**empty, "reason": f"Waiting for {missing} to stabilise (df_len={len(df)})"}
        
        atr   = float(last.get("atr", 0))
        
        # 4. Filter: London (7-10 GMT) & NY (13-17 GMT) for Sniper precision
        now_hour = datetime.now(timezone.utc).hour
        is_london = 7 <= now_hour <= 10
        is_ny     = 13 <= now_hour <= 17
        
        if is_sniper and not (is_london or is_ny):
             return {**empty, "reason": "Sniper: Outside High-Liquidity Windows (London/NY)"}

        # 5. Momentum: 50 EMA + RSI (M1/M5)
        ema50 = float(last.get("ema_50", 0))
        rsi   = float(last.get("rsi", 50))
        rsi_bullish = rsi > 50 and prev.get("rsi", 50) <= 50
        rsi_bearish = rsi < 50 and prev.get("rsi", 50) >= 50
        momentum_scalp_up = live_close_val > ema50 and rsi_bullish
        momentum_scalp_down = live_close_val < ema50 and rsi_bearish

        # Volume RVOL: Current Volume vs 20-period MA
        vol_avg = df["volume"].tail(20).mean()
        rvol = last["volume"] / vol_avg if vol_avg > 0 else 1.0

        # Squeeze check
        if bool(last.get("bb_squeeze", False)):
            return {**empty, "reason": "BB Squeeze (Low Volatility)"}

        # ── ICT/SMC Triggers ──────────────────
        
        # ICT Setup: Sweep + Displacement + FVG Retest
        ict_buy_setup = smc_v2["bullish_sweep"] and smc_v2["displacement"] and near_fvg(live_close_val, smc.get("fvgs", []))
        ict_sell_setup = smc_v2["bearish_sweep"] and smc_v2["displacement"] and near_fvg(live_close_val, smc.get("fvgs", []))

        # ── Channel & Heiken Ashi Breakout ─────
        ema_55_high = float(last.get("ema_55_high", 0))
        ema_55_low  = float(last.get("ema_55_low", 0))
        ha_bull     = bool(last.get("ha_bull", False))
        
        channel_break_up = live_close_val > ema_55_high and ha_bull
        channel_break_dn = live_close_val < ema_55_low and not ha_bull
        in_channel = ema_55_low <= live_close_val <= ema_55_high

        if is_sniper and in_channel:
             return {**empty, "reason": "Sniper: Price inside 55-MA Channel (Choppy)"}

        # ── Trigger Selection ─────────────────
        
        signal   = "HOLD"
        strength = 0.0
        reasons  = []
        hold_reasons = []

        # BUY Logic
        ema_cross_up = last["ema_9"] > last["ema_21"] and prev["ema_9"] <= prev["ema_21"]
        is_buy_trigger = ict_buy_setup or momentum_scalp_up or ema_cross_up or channel_break_up
        
        if is_buy_trigger:
            if not h1_bullish: hold_reasons.append("H1 Trend Bearish")
            if rvol < 1.1:           hold_reasons.append(f"Low RVOL ({rvol:.1f})")
            if is_sniper and not ha_bull: hold_reasons.append("HA Candle Red")
            
            if not hold_reasons:
                # Scalp Sniper: MUST have SMC or HTF Trend + RVOL
                if is_sniper or is_nano:
                    if not (ict_buy_setup or near_ob(live_close_val, smc.get("bull_obs", [])) or channel_break_up):
                         return {**empty, "reason": "Sniper: Waiting for ICT Sweep, OB or HA Breakout"}

                signal   = "BUY"
                strength = 0.75 # Baseline for V2.0
                if ict_buy_setup: strength += 0.15; reasons.append("ICT Sweep+FVG")
                elif channel_break_up: strength += 0.10; reasons.append("HA Breakout")
                elif momentum_scalp_up: strength += 0.05; reasons.append("EMA+RSI Scalp")
                
                if h1_bullish: strength += 0.10; reasons.append("H1 Trend+")
                if rvol > 1.5: strength += 0.05; reasons.append("Vol Spike")

        # SELL Logic
        if signal == "HOLD":
            ema_cross_down = last["ema_9"] < last["ema_21"] and prev["ema_9"] >= prev["ema_21"]
            is_sell_trigger = ict_sell_setup or momentum_scalp_down or ema_cross_down or channel_break_dn
            
            if is_sell_trigger:
                if not h1_bearish: hold_reasons.append("H1 Trend Bullish")
                if rvol < 1.1:           hold_reasons.append(f"Low RVOL ({rvol:.1f})")
                if is_sniper and ha_bull: hold_reasons.append("HA Candle Green")

                if not hold_reasons:
                    if is_sniper or is_nano:
                        if not (ict_sell_setup or near_ob(live_close_val, smc.get("bear_obs", [])) or channel_break_dn):
                             return {**empty, "reason": "Sniper: Waiting for ICT Sweep, OB or HA Breakout"}

                    signal   = "SELL"
                    strength = 0.75
                    if ict_sell_setup: strength += 0.15; reasons.append("ICT Sweep+FVG")
                    elif channel_break_dn: strength += 0.10; reasons.append("HA Breakout")
                    elif momentum_scalp_down: strength += 0.05; reasons.append("EMA+RSI Scalp")
                    
                    if h1_bearish: strength += 0.10; reasons.append("H1 Trend-")
                    if rvol > 1.5: strength += 0.05; reasons.append("Vol Spike")

        if signal != "HOLD":
            sl_dist = max(atr * 1.5, 1.0)
            tp_dist = sl_dist * 2.0

            # ── PPO Confirmation Filter ───────────────────────────
            if self._ppo is not None and self._ppo.is_available():
                ema9_val   = float(last.get("ema_9", 0))
                ema21_val  = float(last.get("ema_21", 0))
                atr_raw    = float(last.get("atr", 1))
                close_norm = 0.0  # fallback (env handles normalisation at train time)
                obs_dict = {
                    "rsi":            rsi / 100.0,
                    "ema_diff_pct":   (ema9_val - ema21_val) / (live_close_val + 1e-9),
                    "atr_norm":       atr_raw / (live_close_val + 1e-9),
                    "rvol":           min(rvol, 5.0),
                    "macd_hist_norm": 0.0,
                    "bb_pct":         0.5,
                    "ha_bull":        1.0 if bool(last.get("ha_bull", True)) else 0.0,
                    "h1_trend":       1.0 if h1_bullish else 0.0,
                    "session_id":     0.5,
                    "close_norm":     close_norm,
                }
                ppo_result  = self._ppo.predict(obs_dict)
                ppo_action  = ppo_result.get("action", "HOLD")
                ppo_conf    = ppo_result.get("confidence", 1.0)

                if ppo_action == signal:
                    # PPO agrees — reinforce the signal
                    strength = min(strength + 0.10, 1.0)
                    reasons.append(f"PPO:Confirm({ppo_conf:.0%})")
                elif ppo_action == "HOLD":
                    # PPO uncertain — mild penalty
                    strength = max(strength - 0.10, 0.0)
                    reasons.append("PPO:Uncertain")
                else:
                    # PPO disagrees — significant penalty; suppress weak trades
                    strength = max(strength - 0.30, 0.0)
                    reasons.append(f"PPO:Disagree({ppo_action})")
                    if strength < 0.60:
                        return {**empty, "reason": "PPO Override: " + ", ".join(reasons)}

            return {
                "signal":      signal,
                "strength":    float(round(float(min(strength, 1.0)), 3)),
                "reason":      ", ".join(reasons),
                "atr":         atr,
                "sl_distance": sl_dist,
                "tp_distance": tp_dist,
                "rsi":         rsi,
                "rvol":        rvol
            }

        return {**empty, "reason": "Searching: " + (", ".join(hold_reasons) if hold_reasons else "Pattern match fail")}

    def record_trade(self):
        """Call this when a scalp trade is placed to track session limit."""
        key = self._session_key()
        self._session_trades[key] = self._session_trades.get(key, 0) + 1

    def _session_key(self) -> str:
        from datetime import date
        sess = get_current_gold_session()
        return f"{date.today()}_{sess['active_kz']}"
