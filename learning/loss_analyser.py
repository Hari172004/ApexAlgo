"""
loss_analyser.py — Post-Trade Loss Diagnostics
Analyzes every closed losing trade to determine the specific point of failure
(e.g., News spike, weak signal score, spread hunt, wrong trend direction).
Stores diagnostics to Supabase for the Auto Improver to consume.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("apexalgo.learning.analyser")

class LossAnalyser:
    def __init__(self):
        # In production this connects to Supabase via supabase-py
        pass

    def analyze_loss(self, trade_data: Dict[str, Any], market_context: Dict[str, Any]) -> str:
        """
        Calculates the primary reason a trade hit its Stop Loss.
        trade_data: includes ticket, symbol, open_time, close_time, hit_sl, initial_score
        market_context: historical data at the time of closure (spread, news, atr)
        """
        logger.info(f"[LOSS_ANALYSER] Executing post-mortem on Ticket #{trade_data.get('ticket', 'UNKNOWN')}")
        
        # 1. Spread Outlier (Stop Hunted)
        # If the bid/ask spread at close was > 2x the normal average
        close_spread = market_context.get("close_spread", 0)
        avg_spread = market_context.get("avg_spread", 1)
        if close_spread > (avg_spread * 2):
            reason = "SPREAD_SPIKE"
            self._store_loss_record(trade_data, reason)
            return reason
            
        # 2. News/Fundamental Event
        # If a strongly bearish/bullish news headline broke within 15 minutes of the SL hit
        news_event = market_context.get("news_event_near_close", False)
        if news_event:
            reason = "NEWS_VOLATILITY"
            self._store_loss_record(trade_data, reason)
            return reason

        # 3. Weak Signal Execution
        # If the trade was taken with a score exactly near the threshold (e.g. 70 or 71)
        score = trade_data.get("signal_score", 100)
        if score <= 75:
            reason = "WEAK_SIGNAL_SCORE"
            self._store_loss_record(trade_data, reason)
            return reason

        # 4. Wrong Trend Direction (Higher Timeframe shifted)
        # If the D1/H4 flipped direction right after we entered
        trend_shift = market_context.get("htf_trend_shifted", False)
        if trend_shift:
            reason = "WRONG_TREND_DIRECTION"
            self._store_loss_record(trade_data, reason)
            return reason

        # 5. Outside / Edge of Kill Zone
        # If the trade survived but closed outside the primary volume hours
        outside_kz = market_context.get("closed_outside_kz", False)
        if outside_kz:
            reason = "OUTSIDE_KILLZONE"
            self._store_loss_record(trade_data, reason)
            return reason

        # Default fallback
        reason = "NORMAL_MARKET_MOVEMENT"
        self._store_loss_record(trade_data, reason)
        return reason

    def _store_loss_record(self, trade_data: dict, reason: str):
        """Mocks saving the post-mortem reason into the Supabase database."""
        ticket = trade_data.get('ticket', 'UNKNOWN')
        logger.warning(f"[LOSS_ANALYSER] Stored Loss Reason for #{ticket}: {reason}")
        # supabase.table("loss_records").insert({"ticket": ticket, "reason": reason}).execute()

global_loss_analyser = LossAnalyser()
