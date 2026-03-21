"""
risk_manager.py — Smart Risk Management Engine for ApexAlgo
==============================================================
Handles: position sizing, SL/TP, breakeven, consecutive-loss stops,
daily loss limits, and reduction of risk as drawdown increases.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("apexalgo.risk_manager")


@dataclass
class RiskState:
    consecutive_losses: int   = 0
    daily_loss: float         = 0.0
    daily_starting_balance: float = 0.0
    trade_count_today: int    = 0
    wins_today: int           = 0
    losses_today: int         = 0
    paused: bool              = False
    pause_reason: str         = ""


class RiskManager:
    """
    Shared risk engine used by all trading modes (Demo, Real, Funded).
    Funded mode layered ON TOP via FundedModeEngine.
    """

    def __init__(self,
                 max_risk_pct: float = 2.0,
                 max_daily_loss_pct: float = 5.0,
                 max_consecutive_losses: int = 3,
                 breakeven_at_r: float = 1.0):
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.breakeven_at_r = breakeven_at_r
        self.state = RiskState()

    # ── Lot Size Calculation ───────────────────────────────────

    def calculate_lot_size(self, balance: float, sl_pips: float,
                           pip_value: float, symbol: str) -> float:
        """
        Calculate lot size so that max_risk_pct % of balance is at risk.
        pip_value = value in account currency per 1 pip on 1 standard lot.
        sl_pips   = distance to stop-loss in pips.
        """
        if sl_pips <= 0 or pip_value <= 0:
            logger.warning("[RiskMgr] Invalid SL or pip_value — using minimum lot 0.01")
            return 0.01

        risk_amount = balance * (self.max_risk_pct / 100)
        lot_size = risk_amount / (sl_pips * pip_value)
        lot_size = round(float(max(0.01, min(lot_size, 10.0))), 2)  # type: ignore[arg-type]  Clamp 0.01–10
        logger.debug(
            f"[RiskMgr] LotSize={lot_size} | Balance=${balance:.2f} "
            f"| Risk=${risk_amount:.2f} | SL={sl_pips}pips"
        )
        return lot_size

    # ── ATR-Based SL / TP ─────────────────────────────────────

    def calculate_sl_tp(self, entry_price: float, atr: float,
                        direction: str, sl_multiplier: float = 1.5,
                        tp_multiplier: float = 2.5) -> tuple[float, float]:
        """
        direction: 'BUY' or 'SELL'
        Returns (stop_loss_price, take_profit_price)
        """
        sl_distance = atr * sl_multiplier
        tp_distance = atr * tp_multiplier

        if direction.upper() == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        sl = round(float(sl), 5)  # type: ignore[arg-type]
        tp = round(float(tp), 5)  # type: ignore[arg-type]
        logger.debug(f"[RiskMgr] ATR={atr:.5f} SL={sl} TP={tp} [{direction}]")
        return sl, tp

    # ── Trailing Stop Loss Logic ───────────────────────────────

    def should_update_sl(self, entry_price: float, current_price: float,
                         current_sl: float, initial_sl: float, direction: str) -> tuple[bool, float]:
        """
        Returns (should_move: bool, new_sl: float).
        Implements a Trailing Stop Loss. Activates when profit >= 1R (initial SL distance).
        Once active, it maintains exactly a 1R distance behind the current price.
        """
        # We don't trail if the initial SL is invalid or missing
        if not initial_sl or initial_sl == entry_price:
            return False, current_sl

        if direction.upper() == "BUY":
            initial_risk = entry_price - initial_sl
            profit = current_price - entry_price

            # Activation at 1R profit
            if profit >= (initial_risk * self.breakeven_at_r):
                # Trail by 1R distance behind current price
                new_sl = current_price - initial_risk
                new_sl = round(float(new_sl), 5) # type: ignore[arg-type]
                # Only move if it locks in more profit than current SL
                if new_sl > current_sl:
                    return True, new_sl

        else:  # SELL
            initial_risk = initial_sl - entry_price
            profit = entry_price - current_price

            if profit >= (initial_risk * self.breakeven_at_r):
                new_sl = current_price + initial_risk
                new_sl = round(float(new_sl), 5) # type: ignore[arg-type]
                # Only move if it locks in more profit than current SL (lower is better for SL in SELL)
                if new_sl < current_sl:
                    return True, new_sl

        return False, current_sl

    # ── Consecutive Loss / Daily Loss Guards ───────────────────

    def check_can_trade(self, current_balance: float) -> tuple[bool, str]:
        """
        Base risk check — call BEFORE every trade.
        Returns (can_trade: bool, reason: str)
        """
        s = self.state

        if s.paused:
            return False, f"Risk pause: {s.pause_reason}"

        if s.consecutive_losses >= self.max_consecutive_losses:
            reason = f"Max consecutive losses ({self.max_consecutive_losses}) reached — pausing."
            self._pause(reason)
            return False, reason

        if s.daily_starting_balance > 0:
            daily_loss_pct = (s.daily_loss / s.daily_starting_balance) * 100
            if daily_loss_pct >= self.max_daily_loss_pct:
                reason = (
                    f"Daily max loss reached: {daily_loss_pct:.2f}% "
                    f"/ limit {self.max_daily_loss_pct}%"
                )
                self._pause(reason)
                return False, reason

        return True, "OK"

    def update_after_trade(self, pnl: float):
        """Update internal state after each trade closes."""
        s = self.state
        s.trade_count_today += 1
        if pnl >= 0:
            s.consecutive_losses = 0
            s.wins_today += 1
        else:
            s.consecutive_losses += 1
            s.daily_loss += abs(pnl)
            s.losses_today += 1
        logger.info(
            f"[RiskMgr] Trade result: PnL=${pnl:+.2f} | "
            f"ConsecLosses={s.consecutive_losses} | "
            f"DailyLoss=${s.daily_loss:.2f}"
        )

    def on_new_day(self, current_balance: float):
        """Reset daily counters at the start of each session."""
        s = self.state
        s.consecutive_losses = 0
        s.daily_loss = 0.0
        s.daily_starting_balance = current_balance
        s.trade_count_today = 0
        s.wins_today = 0
        s.losses_today = 0
        if s.paused:
            s.paused = False
            s.pause_reason = ""
            logger.info("[RiskMgr] Daily reset — pause lifted.")

    def _pause(self, reason: str):
        self.state.paused = True
        self.state.pause_reason = reason
        logger.error(f"[RiskMgr] 🚨 TRADING PAUSED: {reason}")

    def resume(self):
        """Manually resume (e.g. from the app)."""
        self.state.paused = False
        self.state.pause_reason = ""
        logger.info("[RiskMgr] Trading resumed manually.")

    def stats(self) -> dict:
        s = self.state
        total = s.wins_today + s.losses_today
        return {
            "consecutive_losses": s.consecutive_losses,
            "daily_loss":         round(float(s.daily_loss), 2),  # type: ignore[arg-type]
            "trade_count_today":  s.trade_count_today,
            "wins_today":         s.wins_today,
            "losses_today":       s.losses_today,
            "win_rate_today":     round(float((s.wins_today / total * 100) if total > 0 else 0.0), 1),  # type: ignore[arg-type]
            "paused":             s.paused,
            "pause_reason":       s.pause_reason,
        }
