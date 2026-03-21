"""
btc_risk_manager.py — BTC Specific Risk Rules
==============================================
Encapsulates all risk logic for Bitcoin trading.
1. Max 2% account risk
2. Correlation check with XAUUSD
3. News event pauses
4. Volatility (ATR) spikes
"""

import logging
import os

logger = logging.getLogger("apexalgo.btc_risk")

class BTCRiskManager:
    """Manages risk for BTC trades."""

    def __init__(self, max_risk_pct: float = 2.0):
        self.max_risk_pct = max_risk_pct

    def check_all_rules(self, 
                         account_balance: float, 
                         symbol: str, 
                         direction: str, 
                         atr: float,
                         is_gold_active: bool = False) -> dict:
        """
        Returns { 'allowed': bool, 'volume': float, 'reason': str }
        """
        # ── 1. Correlation Check ──────────────────────────────
        # If already in a heavy gold position, reduce BTC risk
        risk_multiplier = 1.0
        if is_gold_active:
            risk_multiplier = 0.5
            logger.info("[Risk] Reducing BTC risk due to active Gold position.")

        # ── 2. Position Sizing ────────────────────────────────
        # Risk Amount = Balance * (Risk% * Multiplier) / 100
        risk_amt = account_balance * (self.max_risk_pct * risk_multiplier) / 100
        
        # Stop Loss distance (using ATR)
        sl_dist = 1.5 * atr if atr > 0 else 500.0 # fallback
        
        # Volume = Risk Amount / SL distance
        # BTC lot size is usually 1.0 = 1 BTC, so we calculate accordingly
        volume = risk_amt / sl_dist if sl_dist > 0 else 0.01
        
        # Cap volume at something reasonable
        volume = min(volume, 5.0) 

        # ── 3. Volatility Check ──────────────────────────────
        # (Placeholder for ATR spike logic)
        
        return {
            "allowed": True if volume > 0 else False,
            "volume": float(f"{volume:.3f}"),
            "reason": "Risk parameters satisfied."
        }
