"""
xm_connector.py -- XM 360 Global API integration for XAUUSD live feed
"""

import logging
import requests
from datetime import datetime

logger = logging.getLogger("apexalgo.xm")

class XMConnector:
    """
    Connects to XM's real-time gold price feed as a redundant data source.
    Falls back to MT5 if unavailable.
    """

    BASE_URL = "https://api.xm.com/v1"  # Placeholder URL - update with real XM API endpoint

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.connected = False

    def get_gold_price(self) -> dict:
        """
        Fetches live XAUUSD bid/ask from XM API.
        Returns empty dict if unavailable.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(f"{self.BASE_URL}/quote/XAUUSD", headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "bid": data.get("bid", 0.0),
                    "ask": data.get("ask", 0.0),
                    "time": datetime.utcnow()
                }
        except Exception as e:
            logger.debug(f"[XM] Price fetch failed: {e}")
        return {}

    def is_available(self) -> bool:
        """Check if XM feed is reachable."""
        try:
            resp = requests.get(f"{self.BASE_URL}/ping", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
