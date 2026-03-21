"""
signal_scorer.py — 100 Point Master Signal Scoring Engine
Generates a weighted score out of 100 for every incoming trade signal.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("apexalgo.filters.scorer")

class SignalScorer:
    """
    Score breakdown (Total: 100):
    RSI in ideal zone (40–60):      +15 points
    EMA alignment correct:          +15 points
    MACD confirms direction:        +10 points
    Volume spike present:           +15 points
    Kill Zone active:               +15 points
    News sentiment agrees:          +10 points
    Multi timeframe confluence:     +10 points
    Order Block or FVG present:     +10 points
    """
    def __init__(self):
        self.min_trade_threshold = 70
        self.min_log_threshold = 60

    def evaluate_signal(self, metrics: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """
        Receives raw indicators and flags, calculates final score out of 100.
        Returns a dict containing the score and decision.
        """
        score = 0
        breakdown = {}

        is_buy = direction.upper() == "BUY"

        # 1. RSI (Ideal pullback zone 40-60)
        rsi = metrics.get('rsi', 50)
        if 40 <= rsi <= 60:
            score += 15
            breakdown['rsi'] = 15
        else:
            breakdown['rsi'] = 0

        # 2. EMA Alignment (e.g., 9 > 21 > 50 for BUY)
        ema_aligned = metrics.get('ema_aligned', False)
        if ema_aligned:
            score += 15
            breakdown['ema_aligned'] = 15
        else:
            breakdown['ema_aligned'] = 0

        # 3. MACD Confirmation
        macd_confirms = metrics.get('macd_confirms', False)
        if macd_confirms:
            score += 10
            breakdown['macd_confirms'] = 10
        else:
            breakdown['macd_confirms'] = 0

        # 4. Volume Spike
        volume_spike = metrics.get('volume_spike', False)
        if volume_spike:
            score += 15
            breakdown['volume_spike'] = 15
        else:
            breakdown['volume_spike'] = 0

        # 5. Kill Zone Active
        kz_active = metrics.get('kill_zone_active', False)
        if kz_active:
            score += 15
            breakdown['kill_zone_active'] = 15
        else:
            breakdown['kill_zone_active'] = 0

        # 6. News Sentiment (FinBERT agrees)
        # e.g., Bearish news for SELL = True
        news_agrees = metrics.get('news_agrees', False)
        if news_agrees:
            score += 10
            breakdown['news_agrees'] = 10
        else:
            breakdown['news_agrees'] = 0

        # 7. MTF Confluence (4 out of 5 minimum)
        mtf_score = metrics.get('mtf_confluence', 0)  # out of 5
        if mtf_score >= 4:
            score += 10
            breakdown['mtf_confluence'] = 10
        else:
            breakdown['mtf_confluence'] = 0

        # 8. Market Structure (OB / FVG)
        structure_present = metrics.get('structure_present', False)
        if structure_present:
            score += 10
            breakdown['structure_present'] = 10
        else:
            breakdown['structure_present'] = 0

        # Decision
        can_trade = score >= self.min_trade_threshold
        
        status = "IGNORE"
        if can_trade:
            status = "EXECUTE"
        elif score >= self.min_log_threshold:
            status = "LOG_ONLY"

        if status == "EXECUTE":
            logger.info(f"✅ [SCORER] Signal {direction} passed with Score {score}/100")
        elif status == "LOG_ONLY":
            logger.warning(f"⚠️ [SCORER] Signal {direction} logged (Score {score}/100). Minimum 70 required.")
        else:
            logger.debug(f"❌ [SCORER] Signal {direction} ignored (Score {score}/100)")

        return {
            "score": score,
            "breakdown": breakdown,
            "status": status,
            "direction": direction
        }

global_signal_scorer = SignalScorer()
