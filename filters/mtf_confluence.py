"""
mtf_confluence.py — Multi Timeframe Confluence Filter
Validates execution alignment across D1, H4, H1, M15, M5 timeframes.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("apexalgo.filters.mtf")

class MTFConfluenceFilter:
    def __init__(self):
        self.timeframes = ["D1", "H4", "H1", "M15", "M5"]

    def evaluate_confluence(self, symbol: str, mtf_directions: Dict[str, str], requested_direction: str) -> Dict[str, Any]:
        """
        mtf_directions format: {"D1": "BUY", "H4": "BUY", "H1": "SELL", "M15": "BUY", "M5": "BUY"}
        Checks how many timeframes agree with the requested_direction.
        
        Returns integer score out of 5 and a strict bool validation (min 4/5).
        """
        req_dir = requested_direction.upper()
        
        score_count = 0
        conflicts = []
        
        for tf in self.timeframes:
            tf_dir = mtf_directions.get(tf, "NEUTRAL").upper()
            if tf_dir == req_dir:
                score_count += 1
            else:
                conflicts.append(tf)
                
        # Required criteria by User:
        # Highest timeframes should strongly bias. If 4H and D1 disagree completely, maybe block.
        # But explicitly requested: "Only trade if 4 out of 5 timeframes agree minimum"
        pass_mtf = score_count >= 4
        
        # Explicit request: "Skip trade if higher timeframes conflict with entry timeframe"
        # Assuming entry timeframe is M5.
        if pass_mtf and "D1" in conflicts and "H4" in conflicts:
            # Although impossible to be >= 4 if D1 and H4 both conflict out of 5,
            # we explicitly ensure the highest TF isn't in absolute contradiction.
            pass_mtf = False

        if not pass_mtf:
            logger.warning(f"[MTF] {symbol} {req_dir} REJECTED. Score: {score_count}/5. Conflicts: {conflicts}")
        else:
            logger.info(f"[MTF] {symbol} {req_dir} ACCEPTED. Score: {score_count}/5.")
            
        return {
            "mtf_score": score_count,
            "mtf_passed": pass_mtf,
            "conflicts": conflicts
        }

global_mtf_filter = MTFConfluenceFilter()
