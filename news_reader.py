"""
news_reader.py — Live News Fetcher + NLP Sentiment Analyzer
===========================================================
Fetches from NewsAPI (REST) and ForexFactory (RSS) every 15 minutes.
Scores articles using VADER + TextBlob and returns Bullish/Bearish/Neutral.
"""

import os
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import threading

logger = logging.getLogger("apexalgo.news")

FOREXFACTORY_RSS = "https://www.forexfactory.com/ff_calendar_thisweek.xml"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# Keywords that affect XAUUSD and BTC
RELEVANT_KEYWORDS = [
    "gold", "xau", "bitcoin", "btc", "crypto", "fed", "inflation", "interest rate",
    "cpi", "nfp", "fomc", "usd", "dollar", "central bank", "risk off", "safe haven",
    "recession", "quantitative", "jerome powell", "bank of england", "ecb"
]

IMPACT_WEIGHT = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


class NewsReader:
    def __init__(self, newsapi_key: str, fetch_interval_minutes: int = 15):
        self.newsapi_key = newsapi_key
        self.fetch_interval = fetch_interval_minutes * 60
        self.vader = SentimentIntensityAnalyzer()
        self._articles: list = []
        self._events: list = []
        self._last_fetch: float = 0
        self._lock = threading.Lock()

    # ── Fetching ──────────────────────────────────────────────

    def fetch_all(self):
        """Fetch from both sources and store internally."""
        articles = self._fetch_newsapi()
        events   = self._fetch_forexfactory()
        with self._lock:
            self._articles = articles
            self._events   = events
            self._last_fetch = time.time()
        logger.info(f"[News] Fetched {len(articles)} articles, {len(events)} FF events")

    def _fetch_newsapi(self) -> list:
        if not self.newsapi_key:
            return []
        try:
            params = {
                "q":        "gold OR bitcoin OR forex OR FED OR inflation",
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 30,
                "apiKey":   self.newsapi_key,
            }
            resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {
                    "source":    a.get("source", {}).get("name", ""),
                    "title":     a.get("title", ""),
                    "body":      a.get("description", "") or "",
                    "published": a.get("publishedAt", ""),
                    "url":       a.get("url", ""),
                }
                for a in articles
            ]
        except Exception as e:
            logger.error(f"[News] NewsAPI error: {e}")
            return []

    def _fetch_forexfactory(self) -> list:
        """Parse ForexFactory RSS calendar for high-impact events."""
        try:
            resp = requests.get(FOREXFACTORY_RSS, timeout=10,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            events = []
            for item in root.iter("event"):
                title  = item.findtext("title", "")
                impact = item.findtext("impact", "LOW").upper()
                date_str = item.findtext("date", "")
                time_str = item.findtext("time", "")
                try:
                    event_dt = datetime.strptime(f"{date_str} {time_str}", "%m-%d-%Y %I:%M%p")
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    event_dt = datetime.now(timezone.utc)
                events.append({"event": title, "impact": impact, "time": event_dt})
            return events
        except Exception as e:
            logger.error(f"[News] ForexFactory error: {e}")
            return []

    # ── Sentiment Analysis ────────────────────────────────────

    def _score_text(self, text: str) -> float:
        """Combined VADER + TextBlob score. Returns -1.0 to +1.0."""
        if not text:
            return 0.0
        vader_score = self.vader.polarity_scores(text)["compound"]
        blob_score  = TextBlob(text).sentiment.polarity  # type: ignore
        combined    = (vader_score * 0.6) + (blob_score * 0.4)
        return round(combined, 3)

    def _is_relevant(self, text: str) -> bool:
        text_l = text.lower()
        return any(kw in text_l for kw in RELEVANT_KEYWORDS)

    def get_sentiment(self, symbol: str = "XAUUSD") -> dict:
        """
        Aggregate sentiment for XAUUSD or BTCUSD from recent articles.
        Returns:
            {
              'label':  'BULLISH' | 'BEARISH' | 'NEUTRAL',
              'score':  float (-1 to +1),
              'articles_used': int,
              'high_impact_events': list
            }
        """
        with self._lock:
            articles = list(self._articles)
            events   = list(self._events)

        scores = []
        used   = 0
        for art in articles:
            text = f"{art['title']} {art['body']}"
            if not self._is_relevant(text):
                continue
            score = self._score_text(text)
            scores.append(score)
            used += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Determine label
        if avg_score >= 0.05:
            label = "BULLISH"
        elif avg_score <= -0.05:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        # High-impact upcoming events
        now = datetime.now(timezone.utc)
        upcoming_high = [
            e for e in events
            if e.get("impact") == "HIGH"
            and 0 <= (e["time"] - now).total_seconds() <= 4 * 3600
        ]

        result = {
            "label":                label,
            "score":                round(float(avg_score), 3),  # type: ignore[arg-type]
            "articles_used":        used,
            "high_impact_events":   upcoming_high,
        }
        logger.info(f"[News] Sentiment={label} ({avg_score:+.3f}) | HighImpact={len(upcoming_high)}")
        return result

    def needs_refresh(self) -> bool:
        return time.time() - self._last_fetch >= self.fetch_interval

    def start_background_refresh(self):
        """Spawn a daemon thread that auto-refreshes news every interval."""
        def _loop():
            while True:
                try:
                    self.fetch_all()
                except Exception as e:
                    logger.error(f"[News] Background refresh error: {e}")
                time.sleep(self.fetch_interval)

        t = threading.Thread(target=_loop, daemon=True, name="news-refresh")
        t.start()
        logger.info("[News] Background refresh started (every "
                    f"{self.fetch_interval//60} min)")
