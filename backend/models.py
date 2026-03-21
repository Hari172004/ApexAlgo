"""
models.py — Pydantic data models for the ApexAlgo backend API
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TradingMode(str, Enum):
    DEMO   = "DEMO"
    REAL   = "REAL"
    FUNDED = "FUNDED"


class StrategyMode(str, Enum):
    SCALP = "SCALP"
    SWING = "SWING"
    AUTO  = "AUTO"


class AssetMode(str, Enum):
    XAUUSD = "XAUUSD"
    BTCUSD = "BTCUSD"
    BOTH   = "BOTH"


class PropFirm(str, Enum):
    FTMO         = "FTMO"
    MyForexFunds = "MyForexFunds"
    The5ers      = "The5ers"
    Apex         = "Apex"
    TrueForex    = "TrueForex"
    CUSTOM       = "CUSTOM"


class SubscriptionPlan(str, Enum):
    STARTER = "STARTER"
    PRO     = "PRO"
    ELITE   = "ELITE"


# ── Auth ──────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email:    str
    password: str
    name:     str


class UserLogin(BaseModel):
    email:    str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    plan:         Optional[str] = None
    license_key:  Optional[str] = None


# ── Bot Control ───────────────────────────────────────────────

class BotConfigRequest(BaseModel):
    mode:          Optional[TradingMode]   = None
    strategy:      Optional[StrategyMode]  = None
    assets:        Optional[AssetMode]     = None
    risk_pct:      Optional[float]         = Field(None, ge=0.1, le=5.0)
    firm:          Optional[PropFirm]      = None
    firm_phase:    Optional[str]           = None
    firm_balance:  Optional[float]         = None
    mt5_account:   Optional[int]           = None
    mt5_password:  Optional[str]           = None
    mt5_server:    Optional[str]           = None
    exchange:      Optional[str]           = None
    ccxt_key:      Optional[str]           = None
    ccxt_secret:   Optional[str]           = None
    ccxt_testnet:  Optional[bool]          = None


class BotStartRequest(BaseModel):
    user_id:    str
    config:     BotConfigRequest


class BotStatusResponse(BaseModel):
    running:         bool
    mode:            str
    strategy:        str
    assets:          str
    balance:         float
    equity:          float
    open_positions:  list
    risk_stats:      dict
    funded_report:   Optional[dict] = None
    last_update:     str


# ── Trades ────────────────────────────────────────────────────

class TradeRecord(BaseModel):
    id:           Optional[str]
    user_id:      str
    symbol:       str
    strategy:     str
    mode:         str
    direction:    str
    entry_price:  float
    exit_price:   Optional[float] = None
    sl:           float
    tp:           float
    volume:       float
    pnl:          Optional[float] = None
    win:          Optional[bool]  = None
    exit_reason:  Optional[str]   = None
    sentiment:    Optional[str]   = None
    opened_at:    Optional[str]   = None
    closed_at:    Optional[str]   = None


class TradeHistory(BaseModel):
    trades:     List[TradeRecord]
    total:      int
    wins:       int
    losses:     int
    win_rate:   float
    total_pnl:  float


# ── Subscriptions ─────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    user_id: str
    plan:    SubscriptionPlan
    email:   str


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id:   str


class LicenseInfo(BaseModel):
    user_id:     str
    plan:        str
    license_key: str
    active:      bool
    expires_at:  Optional[str]


# ── Prop Firm ─────────────────────────────────────────────────

class FundedSetupRequest(BaseModel):
    user_id:        str
    firm:           PropFirm
    phase:          str
    starting_balance: float
    custom_rules:   Optional[dict] = None


class FundedReportResponse(BaseModel):
    firm:                  str
    phase:                 str
    current_balance:       float
    total_profit:          float
    profit_progress_pct:   float
    daily_loss_used_pct:   float
    drawdown_used_pct:     float
    days_remaining:        int
    halted:                bool
    phase_passed:          bool
    phase_failed:          bool
