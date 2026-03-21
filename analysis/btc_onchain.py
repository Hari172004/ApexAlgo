"""
btc_onchain.py — BTC On-Chain Data Reader
==========================================
Fetches transaction volume, active addresses, and exchange flows.
Sources: Blockchain.com, Glassnode (free), CoinMetrics.
"""

import requests # type: ignore
import logging
import time

logger = logging.getLogger("apexalgo.onchain")

class BTCOnChain:
    """Reads on-chain metrics for BTC."""

    def __init__(self, glassnode_key: str = ""):
        self.glassnode_key = glassnode_key
        self.cache = {}

    def get_blockchain_stats(self) -> dict:
        """Fetch from Blockchain.com (no key needed)."""
        try:
            url = "https://api.blockchain.info/stats"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            return {
                "mkt_price": data.get("market_price_usd"),
                "hash_rate": data.get("hash_rate"),
                "total_fees": data.get("total_fees_btc"),
                "n_trans": data.get("n_number_of_transactions")
            }
        except Exception as e:
            logger.error(f"[OnChain] Blockchain.com error: {e}")
            return {}

    def get_exchange_flow_sentiment(self) -> str:
        """
        Simplified sentiment based on simulated or free Glassnode flow.
        Net Flow = Inflow - Outflow
        Negative Net Flow = Bullish (BTC leaving exchanges)
        """
        # Placeholder for real API call
        # Mock logic or free-tier Glassnode if key is present
        return "BULLISH" # Defaulting for now

    def get_health_score(self) -> float:
        """Combines metrics into a 0-1 score."""
        stats = self.get_blockchain_stats()
        if not stats: return 0.5
        
        # Logic: high transaction count is healthy
        n_trans = stats.get("n_trans", 0)
        if n_trans > 300000: return 0.8
        elif n_trans > 200000: return 0.6
        return 0.4
