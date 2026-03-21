"""
auto_improver.py — Autonomous Optimization Loop
Runs weekly to aggregate recent loss records. Discovers the weakest filter 
and recalculates minimum thresholds (e.g., increasing win rate requirement 
if too many 'WEAK_SIGNAL' losses occurred).
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger("apexalgo.learning.improver")

class AutoImprover:
    def __init__(self):
        pass

    def fetch_weekly_loss_data(self) -> List[Dict[str, str]]:
        """
        Queries DB for last 7 days of losses.
        Returns a mock list of generalized reasons for local testing.
        """
        # Mock database query
        return [
            {"reason": "SPREAD_SPIKE"},
            {"reason": "WEAK_SIGNAL_SCORE"},
            {"reason": "WEAK_SIGNAL_SCORE"},
            {"reason": "WEAK_SIGNAL_SCORE"},
            {"reason": "NEWS_VOLATILITY"},
            {"reason": "WRONG_TREND_DIRECTION"}
        ]

    def execute_weekly_review(self) -> Dict[str, Any]:
        """
        Aggregates reasons, applies percentage thresholds, and triggers
        configuration tightening.
        """
        logger.info("[AUTO_IMPROVER] 🧠 Initiating Weekly Feedback Loop Review...")
        
        loss_data = self.fetch_weekly_loss_data()
        if not loss_data:
            logger.info("[AUTO_IMPROVER] No losses recorded this week. Optimal performance.")
            return {}

        total_losses = len(loss_data)
        metrics = {}
        for record in loss_data:
            r = record["reason"]
            metrics[r] = metrics.get(r, 0) + 1

        adjustments = {}
        report_lines = []
        
        for reason, count in metrics.items():
            percentage = (count / total_losses) * 100
            report_lines.append(f"- {reason}: {percentage:.1f}% ({count} trades)")
            
            # 1. If > 30% of losses are due to weak signals, we raise the Signal Scorer threshold
            if reason == "WEAK_SIGNAL_SCORE" and percentage > 30.0:
                logger.warning("[AUTO_IMPROVER] Signal scores are too generous. Tightening config...")
                adjustments['MIN_SIGNAL_SCORE'] = 75  # Increased from 70
                
            # 2. If > 25% due to spread hunt, tighten spread tolerance
            elif reason == "SPREAD_SPIKE" and percentage > 25.0:
                logger.warning("[AUTO_IMPROVER] Spread spikes hurting performance. Tightening config...")
                adjustments['MAX_GOLD_SPREAD'] = 18   # Decreased from 20 pips
                
            # 3. If > 20% due to trend shifts, force MTF confluence to 5/5
            elif reason == "WRONG_TREND_DIRECTION" and percentage > 20.0:
                logger.warning("[AUTO_IMPROVER] Higher timeframes shifting. Modifying MTF requirement...")
                adjustments['REQ_MTF_CONFLUENCE'] = 5 # Increased from 4
                
        # Generate weekly Telegram Report
        report = "*ApexAlgo Auto-Improver Report*\n"
        report += "\n".join(report_lines)
        if adjustments:
            report += "\n\n*Applied Adjustments:*\n"
            for k, v in adjustments.items():
                report += f"⚙️ `{k}` -> `{v}`\n"
        else:
            report += "\n\n*No algorithmic adjustments required.*"
            
        logger.info(f"[AUTO_IMPROVER] Review Complete.\n{report}")
        
        # Normally would save 'adjustments' back to a .env override or Supabase config table.
        return adjustments

global_auto_improver = AutoImprover()
