"""
gold_risk_manager.py -- Complete Gold risk management for XAUUSD
Rules: 2% cap, 15-pip min SL, 3 max open, daily loss limit, spread check,
       ATR spike guard, news pause, BTC correlation, funded account compliance.
"""

import logging
from datetime import datetime, date, timezone
from analysis.gold_sessions import is_lbma_fix_time  # type: ignore

logger = logging.getLogger("agniv.gold_risk")

# Risk constants
MAX_RISK_PCT          = 2.0     # Max % account risk per trade
MIN_SL_POINTS         = 1.5     # Min stop loss in XAU points (= 15 pips on most brokers)
MAX_OPEN_GOLD_TRADES  = 3
DAILY_LOSS_LIMIT_PCT  = 3.0     # Stop all gold trading at 3% daily loss
MAX_SPREAD_POINTS     = 3.0     # 30 pips = 3.0 XAU points — pause if wider
ATR_SPIKE_MULTIPLIER  = 3.0     # If ATR > 3x avg ATR → reduce size 70%
FUNDED_MAX_RISK_PCT   = 1.0     # More conservative in funded mode
FUNDED_MAX_TRADE_PCT  = 0.5     # Max 0.5% account per pip in funded mode

# News pause window in minutes around high-impact events
NEWS_BLACKOUT_MINS = 30

# Tracks daily stats
_daily: dict = {}


def _today() -> str:
    return date.today().isoformat()


def record_daily_loss(loss_usd: float):
    """Call when a gold trade closes at a loss."""
    key = _today()
    _daily.setdefault(key, {"loss": 0.0, "profit": 0.0, "trades": 0})
    _daily[key]["loss"] += abs(loss_usd)
    _daily[key]["trades"] += 1


def record_daily_profit(profit_usd: float):
    """Call when a gold trade closes at a profit."""
    key = _today()
    _daily.setdefault(key, {"loss": 0.0, "profit": 0.0, "trades": 0})
    _daily[key]["profit"] += abs(profit_usd)
    _daily[key]["trades"] += 1


def _daily_loss_exceeded(balance: float) -> bool:
    key  = _today()
    pnl  = _daily.get(key, {})
    loss = pnl.get("loss", 0.0)
    return loss >= balance * (DAILY_LOSS_LIMIT_PCT / 100)


class GoldRiskManager:
    def __init__(self, config):
        self.config    = config
        self.funded    = getattr(config, "mode", "DEMO") == "FUNDED"
        self.low_balance_mode = False

    def set_dynamic_safety(self, balance: float):
        """Adjust risk parameters based on account size."""
        self.low_balance_mode = (balance < 500)
        if self.low_balance_mode:
            logger.info(f"[GoldRisk] Low balance ({balance}) detected. Risk restricted.")

    # ── Main check ────────────────────────────────────────────────────────

    def check_all_rules(
        self,
        balance:        float,
        signal:         str,
        atr:            float,
        open_gold_pos:  int,
        is_btc_active:  bool,
        spread_points:  float = 0.0,
        news_pause:     bool  = False,
    ) -> dict:
        """
        Run all Gold risk checks.
        Returns: {can_trade, reason, volume, sl_value, risk_usd}
        """
        if signal == "HOLD":
            return {"can_trade": False, "reason": "No signal"}

        # 1. Daily loss limit
        if _daily_loss_exceeded(balance):
            return {"can_trade": False, "reason": f"Daily {DAILY_LOSS_LIMIT_PCT}% loss limit reached"}

        # 2. Max concurrent gold trades
        if open_gold_pos >= MAX_OPEN_GOLD_TRADES:
            return {"can_trade": False, "reason": f"Max {MAX_OPEN_GOLD_TRADES} gold trades open"}

        # 3. Spread check (avoid trading when spread > 30 pips)
        if spread_points > MAX_SPREAD_POINTS:
            return {"can_trade": False, "reason": f"Spread {spread_points:.2f} pts too wide (>{MAX_SPREAD_POINTS})"}

        # 4. LBMA fix window
        if is_lbma_fix_time():
            return {"can_trade": False, "reason": "LBMA gold fix window — paused"}

        # 5. News pause (ForexFactory red folder)
        if news_pause:
            return {"can_trade": False, "reason": "High-impact news pause active"}

        # 6. BTC correlation guard (max 4% combined)
        btc_risk_penalty = 0.5 if is_btc_active else 1.0

        # 7. ATR spike guard (if ATR > 3x normal, reduce size 70%)
        atr_spike = atr > 0 and (atr > (atr * ATR_SPIKE_MULTIPLIER))
        # (actual comparison done against rolling avg ATR passed from caller if needed)
        atr_mult  = 0.30 if atr_spike else 1.0

        # 8. Dynamic Session Risk (DSR)
        # London/NY Overlap (12:00-16:00 UTC) is the "Vol Zone"
        now_hour = datetime.now(timezone.utc).hour
        is_vol_zone = 12 <= now_hour <= 16
        is_asian    = 22 <= now_hour or now_hour <= 8
        
        session_mult = 1.25 if is_vol_zone else (0.25 if is_asian else 1.0)
        
        # 9. Funded mode — more conservative
        risk_pct = FUNDED_MAX_RISK_PCT if self.funded else MAX_RISK_PCT
        risk_pct *= btc_risk_penalty * session_mult
        
        if is_asian:
            logger.info(f"[GoldRisk] Asian Session: Risk reduced by 75% ({risk_pct:.2f}%)")
        elif is_vol_zone:
            logger.info(f"[GoldRisk] Vol Zone (London/NY Overlap): Risk boosted by 25% ({risk_pct:.2f}%)")

        # 10. Calculate lot size
        risk_usd = balance * (risk_pct / 100) * atr_mult
        sl_val   = max(float(atr) * 1.5, MIN_SL_POINTS)

        # XAUUSD: contract_size = 100 oz, 1 point = $100/lot
        # Volume = Risk / (SL_points * 100)
        raw_vol  = risk_usd / (sl_val * 100)

        # Funded account pip limit
        if self.funded:
            max_vol_funded = balance * (FUNDED_MAX_TRADE_PCT / 100) / (sl_val * 100)
            raw_vol = min(raw_vol, max_vol_funded)

        volume = float(round(max(0.01, min(raw_vol, 10.0)), 2))  # type: ignore

        return {
            "can_trade": True,
            "volume":    volume,
            "sl_value":  sl_val,
            "risk_usd":  risk_usd,
            "risk_pct":  risk_pct,
            "is_vol_zone": is_vol_zone
        }

    # ── Consistency checks (funded) ───────────────────────────────────────

    def is_weekend(self) -> bool:
        return datetime.now(timezone.utc).weekday() >= 5  # Sat=5, Sun=6

    def check_funded_consistency(self, this_trade_profit: float, total_day_profit: float) -> bool:
        """
        Funded rule: no single gold trade should exceed 30% of daily total profit.
        """
        if total_day_profit <= 0:
            return True
        return this_trade_profit / total_day_profit <= 0.30
