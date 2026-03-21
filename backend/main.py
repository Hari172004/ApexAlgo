"""
main.py — ApexAlgo FastAPI Backend
======================================
Runs as the cloud API on Oracle Cloud VPS.
All Android app and web dashboard routes are here.
"""

import os
import threading
import logging
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from backend.models import (
    BotConfigRequest, BotStartRequest, BotStatusResponse,
    CheckoutRequest, FundedSetupRequest, FundedReportResponse,
    TradeHistory,
)
from backend.auth     import init_firebase, get_current_user, require_valid_license, require_pro_or_elite
from backend.database import (
    get_trades, get_trade_stats, get_license, get_all_users,
    get_all_licenses, save_funded_report, get_funded_report,
)
from backend.payments  import create_checkout_session, handle_webhook
from history_store     import HistoryStore, SYMBOL_MAP, TIMEFRAME_MAP

_history_store = HistoryStore()

load_dotenv()
logger = logging.getLogger("apexalgo.api")

# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title="ApexAlgo API",
    description="Cloud backend for the ApexAlgo SaaS trading bot platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production set to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot registry: one bot instance per user session
_bot_registry: dict = {}   # user_id → ApexAlgoBot
_bot_threads:  dict = {}   # user_id → Thread


@app.on_event("startup")
async def startup():
    init_firebase()
    logger.info("[API] ApexAlgo API started.")


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "service": "apexalgo-api"}


# ── Auth / License ────────────────────────────────────────────

@app.get("/license")
def get_my_license(user: dict = Depends(get_current_user)):
    lic = get_license(user["uid"])
    if not lic:
        return {"active": False, "plan": None}
    return lic


# ── Bot Control ───────────────────────────────────────────────

@app.post("/bot/start")
def start_bot(req: BotStartRequest, user: dict = Depends(get_current_user)):
    """Start a bot instance for the authenticated user."""
    uid = user["uid"]
    plan = require_valid_license(user=user)

    # Import here to avoid circular at module level
    from core import ApexAlgoBot, BotConfig, MODE_DEMO, MODE_FUNDED

    # Funded mode requires Pro or Elite
    if req.config.mode and req.config.mode.value == "FUNDED":
        require_pro_or_elite(user)

    if uid in _bot_registry and _bot_registry[uid]._running:
        raise HTTPException(status_code=400, detail="Bot already running. Stop it first.")

    cfg_kwargs = {k: v for k, v in req.config.dict().items() if v is not None}
    # Map enum values to strings
    cfg_str = {k: getattr(v, "value", v) for k, v in cfg_kwargs.items()}

    bot = ApexAlgoBot(BotConfig(**cfg_str))
    _bot_registry[uid] = bot

    thread = threading.Thread(target=bot.start, daemon=True, name=f"bot-{uid[:8]}")
    thread.start()
    _bot_threads[uid] = thread

    logger.info(f"[API] Bot started for user={uid} plan={plan}")
    return {"status": "started", "mode": cfg_str.get("mode", "DEMO")}


@app.post("/bot/stop")
def stop_bot(user: dict = Depends(get_current_user)):
    uid = user["uid"]
    bot = _bot_registry.get(uid)
    if not bot:
        raise HTTPException(status_code=404, detail="No running bot found.")
    bot.stop()
    return {"status": "stopped"}


@app.get("/bot/status", response_model=BotStatusResponse)
def bot_status(user: dict = Depends(get_current_user)):
    uid = user["uid"]
    bot = _bot_registry.get(uid)
    if not bot:
        return {
            "running":        False,
            "mode":           "DEMO",
            "strategy":       "AUTO",
            "assets":         "BOTH",
            "balance":        0,
            "equity":         0,
            "open_positions": [],
            "risk_stats":     {"consecutive_losses": 0, "daily_loss": 0, "trade_count_today": 0, "wins_today": 0, "losses_today": 0, "win_rate_today": 0, "paused": False, "pause_reason": ""},
            "funded_report":  None,
            "history":        {},
            "last_update":    datetime.utcnow().isoformat(),
        }
    return bot.get_status()


@app.patch("/bot/config")
def update_bot_config(req: BotConfigRequest, user: dict = Depends(get_current_user)):
    """Update bot settings in real-time (e.g., switch mode from app)."""
    uid = user["uid"]
    bot = _bot_registry.get(uid)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not running.")
    updates = {k: getattr(v, "value", v)
               for k, v in req.dict(exclude_none=True).items()}
    bot.update_config(**updates)
    return {"status": "updated", "applied": updates}


# ── Trades ────────────────────────────────────────────────────

@app.get("/trades")
def get_my_trades(limit: int = 100, user: dict = Depends(get_current_user)):
    uid = user["uid"]
    trades = get_trades(uid, limit=limit)
    stats  = get_trade_stats(uid)
    return {
        "trades":    trades,
        **stats,
    }


# ── Historical Market Data ────────────────────────────────────

@app.get("/history/candles")
def history_candles(
    symbol:    str = "XAUUSD",
    timeframe: str = "H1",
    limit:     int = 200,
):
    """
    Return cached OHLCV candles for charting.
    Automatically fetches from yfinance if the cache is empty or stale.
    Public endpoint — no authentication required.
    """
    # Validate inputs
    if symbol not in SYMBOL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown symbol '{symbol}'. Valid: {list(SYMBOL_MAP.keys())}",
        )
    if timeframe not in TIMEFRAME_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown timeframe '{timeframe}'. Valid: {list(TIMEFRAME_MAP.keys())}",
        )
    limit = max(1, min(limit, 1000))  # clamp 1–1000

    # If cache is stale or empty, fetch synchronously (first request)
    if _history_store._is_stale(symbol, timeframe):
        _history_store.fetch_and_cache(symbol, timeframe)

    candles = _history_store.get_candles_json(symbol, timeframe, limit)
    return {
        "symbol":    symbol,
        "timeframe": timeframe,
        "count":     len(candles),
        "candles":   candles,
    }


@app.get("/history/stats")
def history_stats(
    symbol:    str = "XAUUSD",
    timeframe: str = "H1",
):
    """
    Return price statistics and cache metadata for a symbol/timeframe.
    Useful for the dashboard header (last price, range, cache freshness).
    """
    if symbol not in SYMBOL_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown symbol '{symbol}'")
    if timeframe not in TIMEFRAME_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown timeframe '{timeframe}'")

    info = _history_store.cache_info(symbol, timeframe)
    last_close = _history_store.get_last_close(symbol, timeframe)
    return {**info, "last_close": last_close}


# ── Funded Mode ───────────────────────────────────────────────

@app.post("/funded/setup")
def setup_funded(req: FundedSetupRequest, user: dict = Depends(get_current_user)):
    """Configure prop firm rules for a user's funded account."""
    require_pro_or_elite(user)
    uid = user["uid"]
    bot = _bot_registry.get(uid)
    if bot:
        from funded_mode import FundedModeEngine
        bot.funded_engine = FundedModeEngine(
            firm=req.firm.value,
            phase=req.phase,
            starting_balance=req.starting_balance,
            custom_rules=req.custom_rules,
        )
    return {"status": "funded_configured", "firm": req.firm, "phase": req.phase}


@app.get("/funded/report")
def funded_report(user: dict = Depends(get_current_user)):
    """Get current funded account progress report."""
    uid = user["uid"]
    bot = _bot_registry.get(uid)
    if bot and bot.funded_engine:
        report = bot.funded_engine.daily_report()
        save_funded_report(uid, report)
        return report
    # Fallback: fetch from DB
    return get_funded_report(uid)


# ── Subscriptions ─────────────────────────────────────────────

@app.post("/subscribe/checkout")
def start_checkout(req: CheckoutRequest, user: dict = Depends(get_current_user)):
    base = os.getenv("FRONTEND_URL", "https://apexalgo.vercel.app")
    return create_checkout_session(
        user_id     = user["uid"],
        plan        = req.plan.value,
        email       = req.email,
        success_url = f"{base}/dashboard?subscribed=true",
        cancel_url  = f"{base}/pricing",
    )


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    body = await request.body()
    result = handle_webhook(body, stripe_signature)
    return JSONResponse(content=result)


# ── Admin ─────────────────────────────────────────────────────

@app.get("/admin/users")
def admin_users(user: dict = Depends(get_current_user)):
    _require_admin(user)
    return {"users": get_all_users()}


@app.get("/admin/licenses")
def admin_licenses(user: dict = Depends(get_current_user)):
    _require_admin(user)
    return {"licenses": get_all_licenses()}


def _require_admin(user: dict):
    admin_uids = os.getenv("ADMIN_UIDS", "").split(",")
    if user.get("uid") not in admin_uids:
        raise HTTPException(status_code=403, detail="Admin only.")


# ── Run locally ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
