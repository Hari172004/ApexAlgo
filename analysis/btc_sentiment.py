"""
btc_sentiment.py — Multi-source BTC Sentiment
=============================================
Aggregates news, social, and aggregator sentiment for BTC.
Sources: CryptoPanic, NewsAPI, Cointelegraph RSS, CoinDesk RSS, Decrypt RSS.
"""

import requests # type: ignore
import logging
import xml.etree.ElementTree as ET
from typing import List

logger = logging.getLogger("apexalgo.sentiment")

class BTCSentiment:
    """Aggregates BTC sentiment across various platforms."""

    RSS_FEEDS = [
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed"
    ]

    def __init__(self, cryptopanic_key: str = ""):
        self.cryptopanic_key = cryptopanic_key

    def fetch_rss_headlines(self) -> List[str]:
        headlines = []
        for url in self.RSS_FEEDS:
            try:
                resp = requests.get(url, timeout=10, headers={'User-Agent': 'ApexAlgoBot/1.0'})
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item"):
                    title = item.find("title")
                    if title is not None and title.text is not None:
                        headlines.append(title.text)
            except Exception as e:
                logger.warning(f"[Sentiment] RSS error ({url.split('/')[2]}): {e}")
        return headlines

    def get_cryptopanic_sentiment(self) -> float:
        """Fetch sentiment from CryptoPanic API."""
        if not self.cryptopanic_key: return 0.0
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_key}&currencies=BTC"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            # CryptoPanic doesn't give a direct score, we parse results
            total = len(data.get("results", []))
            return 0.5 # Placeholder for refined parsing
        except Exception as e:
            logger.error(f"[Sentiment] CryptoPanic error: {e}")
            return 0.0

    def get_overall_sentiment(self) -> dict:
        """Combines all sources into a final score."""
        headlines = self.fetch_rss_headlines()
        score = 0.5
        
        # Simple keyword scoring
        bullish_kw = ["surge", "bull", "moon", "buy", "record", "high", "adoption", "etf", "halving"]
        bearish_kw = ["crash", "dump", "bear", "sell", "ban", "hack", "low", "dip", "regulation"]

        bull_count = sum(1 for h in headlines if any(kw in h.lower() for kw in bullish_kw))
        bear_count = sum(1 for h in headlines if any(kw in h.lower() for kw in bearish_kw))

        if bull_count > bear_count:
            score = 0.7
        elif bear_count > bull_count:
            score = 0.3

        label = "NEUTRAL"
        if score > 0.6: label = "BULLISH"
        elif score < 0.4: label = "BEARISH"

        return {
            "score": score,
            "label": label,
            "headline_count": len(headlines)
        }
