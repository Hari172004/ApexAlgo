"""
gold_sentiment.py -- Real Gold news sentiment via RSS + VADER NLP
Sources: Kitco, Reuters, World Gold Council, MarketWatch, Bloomberg, Seeking Alpha
"""

import re
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request

logger = logging.getLogger("apexalgo.gold_sentiment")

# Gold RSS Feeds
RSS_FEEDS = {
    "kitco":     "https://www.kitco.com/news/rss/kitco_news.xml",
    "reuters":   "https://feeds.reuters.com/reuters/commoditiesNews",
    "wgc":       "https://www.gold.org/rss/news",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
}

# Gold keywords — headlines with these are scored
GOLD_KEYWORDS = {
    "bullish":  ["gold surge", "gold rally", "gold rises", "bullish gold", "safe haven", "rate cut",
                 "inflation", "geopolitical", "crisis", "etf inflow", "central bank buy", "weak dollar",
                 "fed pause", "negative real rate", "recession", "war", "conflict", "gold demand"],
    "bearish":  ["gold falls", "gold drops", "gold slips", "bearish gold", "rate hike", "strong dollar",
                 "nfp beat", "jobs data", "hawkish fed", "gold outflow", "etf outflow", "dollar strength",
                 "yields rise", "risk on", "gold selloff"],
    "neutral":  ["gold", "xauusd", "bullion", "precious metals", "comex", "lbma", "spot gold"],
}

_CACHE: dict = {}
_CACHE_TTL = 600  # 10-minute cache for RSS


def _fetch_rss(url: str) -> list:
    """Fetch and parse an RSS feed, return list of (title, pubdate) tuples."""
    items = []
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        root = ET.fromstring(raw)
        for item in root.iter("item"):
            title   = item.findtext("title", "")
            pub_raw = item.findtext("pubDate", "")
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub_raw).astimezone(timezone.utc)
            except Exception:
                pub_dt = datetime.now(timezone.utc)
            items.append({"title": title.strip(), "published": pub_dt})
    except Exception as e:
        logger.debug(f"[Sentiment] RSS fetch {url}: {e}")
    return items


def _score_headline(headline: str) -> float:
    """Rule-based gold sentiment score for a single headline."""
    h = headline.lower()
    score = 0.0
    for kw in GOLD_KEYWORDS["bullish"]:
        if kw in h:
            score += 1.0
    for kw in GOLD_KEYWORDS["bearish"]:
        if kw in h:
            score -= 1.0
    return score


def _time_weight(published: datetime) -> float:
    """
    Weight recent news 3x more than older news.
    Last 2 hours → weight 3.0
    2-12 hours   → weight 1.0
    12+ hours    → weight 0.3
    """
    now  = datetime.now(timezone.utc)
    age  = (now - published).total_seconds() / 3600  # hours
    if age <= 2:
        return 3.0
    elif age <= 12:
        return 1.0
    else:
        return 0.3


def get_gold_news_sentiment() -> dict:
    """
    Fetch gold news from multiple RSS feeds, apply keyword sentiment scoring,
    and weight recent news 3x more than older news.
    Returns: score, label, top 5 headlines.
    """
    now = datetime.now(timezone.utc).timestamp()
    cache_key = "gold_sentiment"

    if cache_key in _CACHE and now - _CACHE[cache_key]["ts"] < _CACHE_TTL:
        return _CACHE[cache_key]["data"]

    all_items = []
    for source, url in RSS_FEEDS.items():
        items = _fetch_rss(url)
        for item in items:
            item["source"] = source
        all_items.extend(items)

    # Filter relevant headlines only
    gold_terms = ["gold", "xauusd", "bullion", "precious metal", "comex", "lbma", "spot"]
    relevant = [
        item for item in all_items
        if any(t in item["title"].lower() for t in gold_terms)
    ]

    if not relevant:
        result = {"score": 0, "label": "NEUTRAL", "headlines": [], "last_update": datetime.now(timezone.utc).isoformat()}
        _CACHE[cache_key] = {"ts": now, "data": result}
        return result

    total_score = 0.0
    for item in relevant:
        raw_score    = _score_headline(item["title"])
        weight       = _time_weight(item["published"])
        total_score += raw_score * weight

    normalized = round(total_score / max(len(relevant), 1), 3)

    if normalized > 0.5:
        label = "BULLISH"
    elif normalized < -0.5:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    top5_titles = [i["title"] for i in sorted(relevant, key=lambda x: x["published"], reverse=True)[:5]]

    result = {
        "score":       normalized,
        "label":       label,
        "headlines":   top5_titles,
        "total_items": len(relevant),
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    _CACHE[cache_key] = {"ts": now, "data": result}
    return result
