"""
gold_fundamentals.py -- Real-time macro monitoring for Gold (XAUUSD)
Fetches: DXY, US 10Y Yield, VIX, Gold ETF flows via Yahoo Finance (free)
Scores: Combined bullish/bearish fundamental bias
"""

import logging
import json
from datetime import datetime, timezone
from urllib.request import urlopen, Request

logger = logging.getLogger("apexalgo.gold_fundamentals")

# Yahoo Finance symbols
YF_DXY   = "DX-Y.NYB"   # US Dollar Index
YF_10Y   = "%5ETNX"     # US 10Y Treasury Yield
YF_VIX   = "%5EVIX"     # CBOE VIX Fear Index
YF_GLD   = "GLD"        # SPDR Gold ETF
YF_GOLD  = "GC%3DF"     # Gold Futures

_CACHE: dict = {}
_CACHE_TTL  = 300  # 5 minutes


def _yf_fetch(symbol: str) -> dict:
    """Fetch real-time quote from Yahoo Finance unofficial JSON endpoint."""
    cache_key = f"yf_{symbol}"
    now = datetime.now(timezone.utc).timestamp()
    if cache_key in _CACHE and now - _CACHE[cache_key]["ts"] < _CACHE_TTL:
        return _CACHE[cache_key]["data"]

    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=8) as resp:
            raw  = json.loads(resp.read().decode())
            meta = raw["chart"]["result"][0]["meta"]
            data = {
                "price":  float(meta.get("regularMarketPrice", 0)),
                "prev":   float(meta.get("chartPreviousClose", meta.get("previousClose", 0))),
                "change": 0.0,
            }
            if data["prev"] != 0:
                data["change"] = round(float((data["price"] - data["prev"]) / data["prev"] * 100), 3)
            _CACHE[cache_key] = {"ts": now, "data": data}
            return data
    except Exception as e:
        logger.debug(f"[Fundamentals] {symbol} fetch error: {e}")
        return {"price": 0.0, "prev": 0.0, "change": 0.0}


def get_gold_fundamental_score() -> dict:
    """
    Composite Gold fundamental bias based on:
    - DXY direction  (inverse to Gold)
    - 10Y Yield direction (inverse to Gold)
    - VIX level (safe-haven demand)
    - GLD ETF 1-day flow proxy
    Returns score (-100 to +100) and bias label.
    """
    dxy_data = _yf_fetch(YF_DXY)
    yield_data = _yf_fetch(YF_10Y)
    vix_data  = _yf_fetch(YF_VIX)
    gld_data  = _yf_fetch(YF_GLD)

    score = 0.0
    details = {}

    # 1. DXY — Gold inverse (DXY up = Gold down)
    dxy_change = dxy_data.get("change", 0.0)
    dxy_score  = -dxy_change * 10  # each 1% DXY move = 10 score pts bearish for gold
    dxy_score  = max(-50, min(50, dxy_score))
    score += dxy_score
    details["dxy_price"]  = dxy_data.get("price", 0.0)
    details["dxy_change"] = dxy_change
    details["dxy_score"]  = dxy_score

    # 2. 10Y Yield — Gold inverse (yield up = Gold down)
    yield_change = yield_data.get("change", 0.0)
    yield_score  = -yield_change * 10
    yield_score  = max(-40, min(40, yield_score))
    score += yield_score
    details["us10y_price"]  = yield_data.get("price", 0.0)
    details["us10y_change"] = yield_change
    details["yield_score"]  = yield_score

    # 3. VIX — High VIX = safe haven demand = bullish gold
    vix_price = vix_data.get("price", 15.0)
    if vix_price > 30:
        vix_score = 20
    elif vix_price > 20:
        vix_score = 10
    else:
        vix_score = 0
    score += vix_score
    details["vix_price"] = vix_price
    details["vix_score"] = vix_score

    # 4. GLD ETF flow proxy (1-day change in GLD price)
    gld_change = gld_data.get("change", 0.0)
    gld_score  = gld_change * 5
    gld_score  = max(-15, min(15, gld_score))
    score += gld_score
    details["gld_price"]  = gld_data.get("price", 0.0)
    details["gld_change"] = gld_change
    details["gld_score"]  = gld_score

    score = round(float(score), 2)
    if score > 25:
        bias = "BULLISH"
    elif score < -25:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "score":   score,
        "bias":    bias,
        "details": details,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
