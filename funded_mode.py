"""
funded_mode.py — Prop Firm Rule Engine for ApexAlgo
=======================================================
Enforces all prop firm rules from FTMO, MyForexFunds, The5ers, Apex, TrueForex.
Supports: Phase 1 (Challenge), Phase 2 (Verification), Live Funded.
Safety buffer of 0.5% added on top of all hard limits.
"""

import logging
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional
import json

logger = logging.getLogger("apexalgo.funded_mode")


def _r2(x: float) -> float:
    """Round x to 2 decimal places (explicit float helper for Pyre2 compat)."""
    return round(x, 2)  # type: ignore[arg-type]

# ──────────────────────────────────────────────────────────────
# Prop Firm Presets
# ──────────────────────────────────────────────────────────────

PROP_FIRM_PRESETS = {
    "FTMO": {
        "daily_loss_limit_pct":    5.0,
        "max_total_drawdown_pct": 10.0,
        "profit_target_pct":      10.0,   # Phase 1
        "phase2_profit_target_pct": 5.0,  # Phase 2
        "min_trading_days":         4,
        "max_trading_days":        30,
        "max_lot_per_10k":          2.0,
        "no_weekend_holding":      True,
        "no_news_trading_mins":    30,
        "consistency_rule_pct":    40.0,  # No single trade > 40% of total profit
        "no_martingale":           True,
    },
    "MyForexFunds": {
        "daily_loss_limit_pct":    5.0,
        "max_total_drawdown_pct": 12.0,
        "profit_target_pct":       8.0,
        "phase2_profit_target_pct": 5.0,
        "min_trading_days":         5,
        "max_trading_days":        30,
        "max_lot_per_10k":          1.5,
        "no_weekend_holding":      True,
        "no_news_trading_mins":    30,
        "consistency_rule_pct":    40.0,
        "no_martingale":           True,
    },
    "The5ers": {
        "daily_loss_limit_pct":    4.0,
        "max_total_drawdown_pct":  6.0,
        "profit_target_pct":       6.0,
        "phase2_profit_target_pct": 6.0,
        "min_trading_days":         3,
        "max_trading_days":        60,
        "max_lot_per_10k":          1.0,
        "no_weekend_holding":      True,
        "no_news_trading_mins":    30,
        "consistency_rule_pct":    50.0,
        "no_martingale":           True,
    },
    "Apex": {
        "daily_loss_limit_pct":    3.0,
        "max_total_drawdown_pct":  6.0,
        "profit_target_pct":       9.0,
        "phase2_profit_target_pct": 9.0,
        "min_trading_days":         5,
        "max_trading_days":        30,
        "max_lot_per_10k":          2.0,
        "no_weekend_holding":      True,
        "no_news_trading_mins":    30,
        "consistency_rule_pct":    40.0,
        "no_martingale":           True,
    },
    "TrueForex": {
        "daily_loss_limit_pct":    5.0,
        "max_total_drawdown_pct": 10.0,
        "profit_target_pct":      10.0,
        "phase2_profit_target_pct": 5.0,
        "min_trading_days":         4,
        "max_trading_days":        30,
        "max_lot_per_10k":          2.0,
        "no_weekend_holding":      True,
        "no_news_trading_mins":    30,
        "consistency_rule_pct":    40.0,
        "no_martingale":           True,
    },
    "CUSTOM": {
        "daily_loss_limit_pct":    5.0,
        "max_total_drawdown_pct": 10.0,
        "profit_target_pct":      10.0,
        "phase2_profit_target_pct": 5.0,
        "min_trading_days":         4,
        "max_trading_days":        30,
        "max_lot_per_10k":          2.0,
        "no_weekend_holding":      True,
        "no_news_trading_mins":    30,
        "consistency_rule_pct":    40.0,
        "no_martingale":           True,
    },
}

SAFETY_BUFFER_PCT = 0.5  # Stop 0.5% BEFORE hitting any hard limit


# ──────────────────────────────────────────────────────────────
# Phase Enum
# ──────────────────────────────────────────────────────────────

class Phase:
    CHALLENGE    = "PHASE_1_CHALLENGE"
    VERIFICATION = "PHASE_2_VERIFICATION"
    LIVE_FUNDED  = "LIVE_FUNDED"


# ──────────────────────────────────────────────────────────────
# Funded Account State
# ──────────────────────────────────────────────────────────────

@dataclass
class FundedAccountState:
    firm: str                       = "FTMO"
    phase: str                      = Phase.CHALLENGE
    starting_balance: float         = 10_000.0
    current_balance: float          = 10_000.0
    peak_balance: float             = 10_000.0
    daily_start_balance: float      = 10_000.0
    total_profit: float             = 0.0
    today_profit: float             = 0.0
    trading_days: int               = 0
    start_date: date                = field(default_factory=date.today)
    current_date: date              = field(default_factory=date.today)
    daily_trades_profit: list       = field(default_factory=list)  # profit per trade today
    halted: bool                    = False
    halt_reason: str                = ""
    phase_passed: bool              = False
    phase_failed: bool              = False


# ──────────────────────────────────────────────────────────────
# FundedModeEngine
# ──────────────────────────────────────────────────────────────

class FundedModeEngine:
    """
    Core prop firm rule enforcement engine.
    Call check_can_trade() before EVERY trade.
    Call update_after_trade() after EVERY trade closes.
    Call on_new_day() at the start of every trading day.
    """

    def __init__(self, firm: str = "FTMO", phase: str = Phase.CHALLENGE,
                 starting_balance: float = 10_000.0, custom_rules: Optional[dict] = None):
        self.state = FundedAccountState(
            firm=firm,
            phase=phase,
            starting_balance=starting_balance,
            current_balance=starting_balance,
            peak_balance=starting_balance,
            daily_start_balance=starting_balance,
            start_date=date.today(),
            current_date=date.today(),
        )
        rules = PROP_FIRM_PRESETS.get(firm, PROP_FIRM_PRESETS["CUSTOM"])
        if custom_rules:
            rules.update(custom_rules)
        self.rules = rules
        logger.info(
            f"[FundedMode] Initialised | Firm={firm} | Phase={phase} "
            f"| Balance=${starting_balance:,.2f} | Rules={json.dumps(rules)}"
        )

    # ── Calculated Properties ──────────────────────────────────

    @property
    def effective_daily_limit_pct(self) -> float:
        return float(self.rules.get("daily_loss_limit_pct", 0.0)) - SAFETY_BUFFER_PCT

    @property
    def effective_drawdown_limit_pct(self) -> float:
        return float(self.rules.get("max_total_drawdown_pct", 0.0)) - SAFETY_BUFFER_PCT

    @property
    def profit_target_pct(self) -> float:
        if self.state.phase == Phase.VERIFICATION:
            return float(self.rules.get("phase2_profit_target_pct", 0.0))
        return float(self.rules.get("profit_target_pct", 0.0))

    @property
    def profit_target_amount(self) -> float:
        return self.state.starting_balance * (self.profit_target_pct / 100)

    @property
    def daily_loss_limit_amount(self) -> float:
        return self.state.daily_start_balance * (self.effective_daily_limit_pct / 100)

    @property
    def max_drawdown_amount(self) -> float:
        return self.state.starting_balance * (self.effective_drawdown_limit_pct / 100)

    @property
    def total_drawdown(self) -> float:
        return self.state.starting_balance - self.state.current_balance

    @property
    def today_loss(self) -> float:
        _diff: float = self.state.daily_start_balance - self.state.current_balance
        return max(0.0, _diff)

    @property
    def days_elapsed(self) -> int:
        return (date.today() - self.state.start_date).days + 1

    @property
    def days_remaining(self) -> int:
        return max(0, int(self.rules.get("max_trading_days", 0)) - self.days_elapsed)

    # ── Max Lot Size Calculation ───────────────────────────────

    def max_lot_size(self, symbol: str) -> float:
        """
        Auto-calculate maximum safe lot size based on:
        - Prop firm per-10k-lot rule
        - Current drawdown (reduce lots as drawdown grows)
        """
        base_lots = (self.state.current_balance / 10_000) * float(self.rules.get("max_lot_per_10k", 0.0))
        drawdown_used_pct = (self.total_drawdown / self.state.starting_balance) * 100
        drawdown_limit = float(self.rules.get("max_total_drawdown_pct", 0.0))
        # Scale down lot size as we approach the drawdown limit
        if drawdown_limit > 0:
            safety_factor = max(0.1, 1.0 - (drawdown_used_pct / drawdown_limit))
        else:
            safety_factor = 1.0
        _final_v: float = float(base_lots * safety_factor)
        final = round(_final_v, 2)  # type: ignore[arg-type]
        final = max(0.01, float(final))  # Never go below 0.01
        logger.debug(f"[FundedMode] MaxLot={final} | DrawdownUsed={drawdown_used_pct:.2f}% | SafetyFactor={safety_factor:.2f}")
        return float(final)

    # ── News Guard ─────────────────────────────────────────────

    def is_near_news(self, upcoming_events: list) -> bool:
        """
        upcoming_events: list of dicts with 'time' (datetime) and 'impact' ('HIGH'|'MEDIUM'|'LOW')
        Returns True if within the no-news window.
        """
        window_mins = int(self.rules.get("no_news_trading_mins", 0))
        now = datetime.utcnow()
        for event in upcoming_events:
            if event.get("impact", "LOW") != "HIGH":
                continue
            event_time = event["time"]
            delta = abs((event_time - now).total_seconds() / 60)
            if delta <= window_mins:
                logger.warning(
                    f"[FundedMode] NEWS GUARD active — {event.get('event','?')} "
                    f"in {delta:.0f} min — no trading."
                )
                return True
        return False

    # ── Weekend Guard ──────────────────────────────────────────

    def is_weekend(self) -> bool:
        """Returns True on Saturday (5) and Sunday (6) UTC."""
        return datetime.utcnow().weekday() >= 5

    def has_open_trades_over_weekend(self, open_positions: list) -> bool:
        return self.is_weekend() and len(open_positions) > 0

    # ── Consistency Rule ───────────────────────────────────────

    def violates_consistency_rule(self, proposed_trade_profit_estimate: float) -> bool:
        """
        No single trade should contribute more than X% of total profit.
        Check BEFORE entering a trade by estimating potential profit.
        """
        limit_pct = float(self.rules.get("consistency_rule_pct", 0.0)) / 100
        if self.state.total_profit <= 0:
            return False  # No profit yet — rule doesn't apply
        return proposed_trade_profit_estimate > (self.state.total_profit * limit_pct)

    # ── Main Gate ─────────────────────────────────────────────

    def check_can_trade(self, upcoming_news: Optional[list] = None,
                        open_positions: Optional[list] = None,
                        estimated_trade_profit: float = 0) -> tuple[bool, str]:
        """
        Master check — call before every trade.
        Returns: (can_trade: bool, reason: str)
        """
        s = self.state

        if s.halted:
            return False, f"Bot halted: {s.halt_reason}"

        if s.phase_failed:
            return False, "Challenge FAILED — account rules breached."

        if s.phase_passed:
            return False, "Phase already passed — waiting for next phase setup."

        # 1. Weekend check
        if self.rules.get("no_weekend_holding") and self.is_weekend():
            return False, "Weekend trading prohibited by prop firm rules."

        # 2. News guard
        if upcoming_news and self.is_near_news(upcoming_news):
            return False, "High-impact news window — trading paused."

        # 3. Daily loss limit
        if self.today_loss >= self.daily_loss_limit_amount:
            reason = (
                f"Daily loss limit reached | Lost: ${self.today_loss:.2f} "
                f"| Limit: ${self.daily_loss_limit_amount:.2f}"
            )
            self._halt(reason)
            return False, reason

        # 4. Total drawdown
        if self.total_drawdown >= self.max_drawdown_amount:
            reason = (
                f"Max drawdown breached | Drawdown: ${self.total_drawdown:.2f} "
                f"| Limit: ${self.max_drawdown_amount:.2f}"
            )
            self._fail_phase(reason)
            return False, reason

        # 5. Time limit
        if self.days_elapsed > int(self.rules.get("max_trading_days", 0)):
            reason = f"Time limit expired ({self.rules.get('max_trading_days', 0)} days)"
            self._fail_phase(reason)
            return False, reason

        # 6. Consistency rule
        if self.violates_consistency_rule(estimated_trade_profit):
            return False, "Consistency rule: single trade too large relative to total profit."

        # ── Approach warning (close to limits) ─────────────────
        daily_loss_pct_used = (self.today_loss / (s.daily_start_balance * float(self.rules.get("daily_loss_limit_pct", 0.0)) / 100)) * 100 if s.daily_start_balance > 0 else 0.0
        if daily_loss_pct_used >= 80:
            logger.warning(f"[FundedMode] ⚠️  Approaching daily loss limit ({daily_loss_pct_used:.1f}% used)")

        total_dd_pct_used = (self.total_drawdown / (s.starting_balance * float(self.rules.get("max_total_drawdown_pct", 0.0)) / 100)) * 100 if s.starting_balance > 0 else 0.0
        if total_dd_pct_used >= 80:
            logger.warning(f"[FundedMode] ⚠️  Approaching max drawdown ({total_dd_pct_used:.1f}% used)")

        return True, "OK"

    # ── State Updates ──────────────────────────────────────────

    def update_after_trade(self, pnl: float):
        """Call when a trade closes. pnl = profit or loss in account currency."""
        s = self.state
        s.current_balance += pnl
        s.today_profit += pnl
        s.total_profit += pnl
        if pnl > 0:
            s.daily_trades_profit.append(pnl)
        # Track peak
        if s.current_balance > s.peak_balance:
            s.peak_balance = s.current_balance
        logger.info(
            f"[FundedMode] Trade closed | PnL=${pnl:+.2f} | "
            f"Balance=${s.current_balance:.2f} | "
            f"TotalProfit=${s.total_profit:.2f} | "
            f"TodayLoss=${self.today_loss:.2f}"
        )
        # Check for phase pass
        self._check_phase_completion()
        # Check for failure
        self._check_phase_failure()

    def on_new_day(self, current_balance: float):
        """Call at start of each new trading day."""
        s = self.state
        s.daily_start_balance = current_balance
        s.current_balance = current_balance
        s.today_profit = 0.0
        s.daily_trades_profit = []
        s.current_date = date.today()
        if datetime.utcnow().weekday() < 5:  # Mon–Fri
            s.trading_days += 1
        if s.halted and self.today_loss == 0:
            s.halted = False
            s.halt_reason = ""
            logger.info("[FundedMode] Daily halt lifted — new day reset.")
        logger.info(
            f"[FundedMode] New day | TradingDay={s.trading_days} "
            f"| DaysLeft={self.days_remaining} | Balance=${current_balance:.2f}"
        )

    def _halt(self, reason: str):
        self.state.halted = True
        self.state.halt_reason = reason
        logger.error(f"[FundedMode] 🚨 BOT HALTED: {reason}")

    def _fail_phase(self, reason: str):
        self.state.phase_failed = True
        self.state.halt_reason = reason
        logger.error(f"[FundedMode] ❌ PHASE FAILED: {reason}")

    def _check_phase_completion(self):
        """Check if profit target has been hit."""
        s = self.state
        if s.phase in (Phase.CHALLENGE, Phase.VERIFICATION):
            if s.total_profit >= self.profit_target_amount:
                if s.trading_days >= self.rules["min_trading_days"]:
                    logger.info(
                        f"[FundedMode] ✅ {s.phase} PASSED! "
                        f"Profit=${s.total_profit:.2f} / Target=${self.profit_target_amount:.2f}"
                    )
                    s.phase_passed = True
                else:
                    logger.info(
                        f"[FundedMode] Target hit but min trading days not met "
                        f"({s.trading_days}/{self.rules['min_trading_days']}). Keep going."
                    )

    def _check_phase_failure(self):
        """Check rule breaches that cause failure."""
        s = self.state
        if self.today_loss >= self.daily_loss_limit_amount:
            self._fail_phase(
                f"Daily loss exceeded ${self.today_loss:.2f} / limit ${self.daily_loss_limit_amount:.2f}"
            )
        if self.total_drawdown >= self.max_drawdown_amount:
            self._fail_phase(
                f"Drawdown ${self.total_drawdown:.2f} exceeded limit ${self.max_drawdown_amount:.2f}"
            )

    # ── Daily Report ───────────────────────────────────────────

    def daily_report(self) -> dict:
        """Generate a structured daily progress report."""
        s = self.state
        return {
            "firm":               s.firm,
            "phase":              s.phase,
            "date":               str(date.today()),
            "starting_balance":   float(s.starting_balance),
            "current_balance":    _r2(s.current_balance),
            "total_profit":       _r2(s.total_profit),
            "total_profit_pct":   _r2((s.total_profit / s.starting_balance) * 100),
            "profit_target":      _r2(self.profit_target_amount),
            "profit_target_pct":  float(self.profit_target_pct),
            "profit_progress_pct": _r2(min(100.0, (s.total_profit / max(1.0, self.profit_target_amount)) * 100)),
            "today_pnl":          _r2(s.today_profit),
            "today_loss":         _r2(self.today_loss),
            "daily_loss_limit":   _r2(self.daily_loss_limit_amount),
            "daily_loss_used_pct": _r2((self.today_loss / max(1.0, self.daily_loss_limit_amount)) * 100),
            "total_drawdown":     _r2(self.total_drawdown),
            "max_drawdown_limit": _r2(self.max_drawdown_amount),
            "drawdown_used_pct":  _r2((self.total_drawdown / max(1.0, self.max_drawdown_amount)) * 100),
            "trading_days":       s.trading_days,
            "days_elapsed":       self.days_elapsed,
            "days_remaining":     self.days_remaining,
            "min_trading_days":   int(self.rules.get("min_trading_days", 0)),
            "max_trading_days":   int(self.rules.get("max_trading_days", 0)),
            "halted":             s.halted,
            "halt_reason":        s.halt_reason,
            "phase_passed":       s.phase_passed,
            "phase_failed":       s.phase_failed,
        }

    def advance_phase(self):
        """Move from Phase 1 → Phase 2 → Live Funded."""
        current = self.state.phase
        if current == Phase.CHALLENGE:
            self.state.phase = Phase.VERIFICATION
        elif current == Phase.VERIFICATION:
            self.state.phase = Phase.LIVE_FUNDED
        self.state.phase_passed = False
        self.state.phase_failed = False
        self.state.total_profit = 0.0
        self.state.start_date = date.today()
        self.state.trading_days = 0
        logger.info(f"[FundedMode] Phase advanced to: {self.state.phase}")
