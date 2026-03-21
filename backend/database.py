"""
database.py — Supabase PostgreSQL Client for ApexAlgo
"""

import os
import logging
from supabase import create_client, Client

logger = logging.getLogger("apexalgo.db")

_supabase_client: Client = None  # type: ignore


def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
        _supabase_client = create_client(url, key)
        logger.info("[DB] Supabase client initialized.")
    return _supabase_client


# ── User ──────────────────────────────────────────────────────

def get_user(user_id: str) -> dict:
    sb = get_supabase()
    result = sb.table("users").select("*").eq("id", user_id).single().execute()
    return result.data or {}  # type: ignore


def upsert_user(user_id: str, data: dict):
    sb = get_supabase()
    sb.table("users").upsert({"id": user_id, **data}).execute()


# ── Trades ────────────────────────────────────────────────────

def insert_trade(trade: dict):
    sb = get_supabase()
    return sb.table("trades").insert(trade).execute()


def get_trades(user_id: str, limit: int = 100) -> list:
    sb = get_supabase()
    result = (
        sb.table("trades")
        .select("*")
        .eq("user_id", user_id)
        .order("opened_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_trade_stats(user_id: str) -> dict:
    trades = get_trades(user_id, limit=1000)
    total  = len(trades)
    wins   = sum(1 for t in trades if t.get("win"))
    losses = total - wins
    total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
    _win_rate: float = (wins / total * 100) if total > 0 else 0.0
    _total_pnl: float = float(total_pnl)
    return {
        "total":    total,
        "wins":     wins,
        "losses":   losses,
        "win_rate": round(_win_rate, 1),  # type: ignore[arg-type]
        "total_pnl": round(_total_pnl, 2),  # type: ignore[arg-type]
    }


# ── Licenses ──────────────────────────────────────────────────

def get_license(user_id: str) -> dict:
    sb = get_supabase()
    result = (
        sb.table("licenses")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else {}  # type: ignore


def create_license(user_id: str, plan: str, key: str, expires_at: str):
    sb = get_supabase()
    sb.table("licenses").insert({
        "user_id":    user_id,
        "plan":       plan,
        "key":        key,
        "active":     True,
        "expires_at": expires_at,
    }).execute()


def deactivate_license(user_id: str):
    sb = get_supabase()
    sb.table("licenses").update({"active": False}).eq("user_id", user_id).execute()


def is_license_valid(user_id: str) -> tuple[bool, str]:
    """Returns (is_valid, plan_name)."""
    from datetime import datetime, timezone
    lic = get_license(user_id)
    if not lic:
        return False, ""
    if not lic.get("active"):
        return False, ""
    expires = lic.get("expires_at")
    if expires:
        exp_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > exp_dt:
            deactivate_license(user_id)
            return False, ""
    return True, lic.get("plan", "")


# ── Funded Progress ───────────────────────────────────────────

def save_funded_report(user_id: str, report: dict):
    sb = get_supabase()
    sb.table("funded_progress").upsert({"user_id": user_id, **report}).execute()


def get_funded_report(user_id: str) -> dict:
    sb = get_supabase()
    result = (
        sb.table("funded_progress")
        .select("*")
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return result.data or {}  # type: ignore


# ── Admin ─────────────────────────────────────────────────────

def get_all_users(limit: int = 200) -> list:
    sb = get_supabase()
    result = sb.table("users").select("*").limit(limit).execute()
    return result.data or []


def get_all_licenses() -> list:
    sb = get_supabase()
    result = sb.table("licenses").select("*").order("created_at", desc=True).execute()
    return result.data or []
