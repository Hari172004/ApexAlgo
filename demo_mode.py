"""
demo_mode.py — Paper Trading Engine for ApexAlgo
=====================================================
Simulates trades using real market data without risking real money.
Tracks virtual balance, open positions, trade history entirely in memory.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("apexalgo.demo")


@dataclass
class VirtualPosition:
    id:         str
    symbol:     str
    direction:  str          # 'BUY' or 'SELL'
    volume:     float
    open_price: float
    sl:         float
    tp:         float
    open_time:  datetime
    comment:    str          = ""
    pnl:        float        = 0.0
    pip_value:  float        = 10.0  # $ per pip per lot (XAUUSD = 1/point)


@dataclass
class DemoAccount:
    balance:   float = 10_000.0
    equity:    float = 10_000.0
    positions: list[Any]  = field(default_factory=list)
    history:   list[dict[str, Any]] = field(default_factory=list)
    wins:      int   = 0
    losses:    int   = 0


class DemoMode:
    """
    Full paper trading engine.
    Receives real price ticks to update PnL on open positions.
    """

    def __init__(self, starting_balance: float = 10_000.0):
        self.account = DemoAccount(
            balance=starting_balance,
            equity=starting_balance,
        )
        logger.info(f"[Demo] Paper account initialised | Balance=${starting_balance:,.2f}")

    # ── Order Management ──────────────────────────────────────

    def open_position(self, symbol: str, direction: str, volume: float,
                      price: float, sl: float = 0.0, tp: float = 0.0,
                      comment: str = "demo") -> dict:
        pos = VirtualPosition(
            id=uuid.uuid4().hex[:8],
            symbol=symbol,
            direction=direction,
            volume=volume,
            open_price=price,
            sl=sl,
            tp=tp,
            open_time=datetime.utcnow(),
            comment=comment,
        )
        self.account.positions.append(pos)
        logger.info(
            f"[Demo] ➕ Open {direction} {volume} {symbol} @ {price:.5f} "
            f"| SL={sl:.5f} TP={tp:.5f} | #{pos.id}"
        )
        return {"id": pos.id, "price": price, "volume": volume, "direction": direction}

    def close_position(self, position_id: str, close_price: float) -> Optional[dict]:
        pos = next((p for p in self.account.positions if p.id == position_id), None)
        if pos is None:
            logger.warning(f"[Demo] Position {position_id} not found")
            return None

        # Calculate PnL
        if pos.direction == "BUY":
            pnl = (close_price - pos.open_price) * pos.volume * 100  # simplified
        else:
            pnl = (pos.open_price - close_price) * pos.volume * 100

        pnl: float = round(float(pnl), 2)
        self.account.balance += pnl
        self.account.equity   = self.account.balance

        if pnl >= 0:
            self.account.wins += 1
        else:
            self.account.losses += 1

        trade_record = {
            "id":          pos.id,
            "symbol":      pos.symbol,
            "direction":   pos.direction,
            "volume":      pos.volume,
            "open_price":  pos.open_price,
            "close_price": close_price,
            "sl":          pos.sl,
            "tp":          pos.tp,
            "pnl":         pnl,
            "open_time":   pos.open_time.isoformat(),
            "close_time":  datetime.utcnow().isoformat(),
            "comment":     pos.comment,
        }
        self.account.history.append(trade_record)
        self.account.positions.remove(pos)

        logger.info(
            f"[Demo] ✅ Close #{pos.id} {pos.direction} {pos.symbol} "
            f"@ {close_price:.5f} | PnL=${pnl:+.2f} | Balance=${self.account.balance:.2f}"
        )
        return trade_record

    def modify_sl(self, position_id: str, new_sl: float) -> bool:
        pos = next((p for p in self.account.positions if p.id == position_id), None)
        if pos:
            pos.sl = new_sl
            return True
        return False

    # ── Tick Update ───────────────────────────────────────────

    def on_tick(self, symbol: str, bid: float, ask: float) -> list:
        """
        Update all open positions with current price.
        Auto-closes if SL or TP is hit.
        Returns list of closed trade records.
        """
        closed = []
        for pos in list(self.account.positions):
            if pos.symbol != symbol:
                continue
            current = bid if pos.direction == "BUY" else ask
            # Update floating PnL
            if pos.direction == "BUY":
                pos.pnl = (bid - pos.open_price) * pos.volume * 100
            else:
                pos.pnl = (pos.open_price - ask) * pos.volume * 100

            # SL check
            if pos.sl > 0:
                if (pos.direction == "BUY" and bid <= pos.sl) or \
                   (pos.direction == "SELL" and ask >= pos.sl):
                    record = self.close_position(pos.id, bid if pos.direction == "BUY" else ask)
                    if record:
                        record["exit_reason"] = "SL"
                        closed.append(record)
                    continue

            # TP check
            if pos.tp > 0:
                if (pos.direction == "BUY" and bid >= pos.tp) or \
                   (pos.direction == "SELL" and ask <= pos.tp):
                    record = self.close_position(pos.id, bid if pos.direction == "BUY" else ask)
                    if record:
                        record["exit_reason"] = "TP"
                        closed.append(record)

        # Update equity
        floating_pnl = sum(p.pnl for p in self.account.positions)
        self.account.equity = self.account.balance + floating_pnl
        return closed

    # ── Account Info ──────────────────────────────────────────

    def get_account_info(self) -> dict:
        total = self.account.wins + self.account.losses
        return {
            "balance":        round(float(self.account.balance), ndigits=2),
            "equity":         round(float(self.account.equity), ndigits=2),
            "open_positions": len(self.account.positions),
            "floating_pnl":   round(float(sum(p.pnl for p in self.account.positions)), ndigits=2),
            "total_trades":   total,
            "wins":           self.account.wins,
            "losses":         self.account.losses,
            "win_rate":       round(float((self.account.wins / total * 100) if total > 0 else 0.0), ndigits=1),
        }

    def get_open_positions(self) -> list:
        return [
            {
                "id":         p.id,
                "symbol":     p.symbol,
                "direction":  p.direction,
                "volume":     p.volume,
                "open_price": p.open_price,
                "sl":         p.sl,
                "tp":         p.tp,
                "pnl":        round(float(p.pnl), ndigits=2),
                "open_time":  p.open_time.isoformat(),
            }
            for p in self.account.positions
        ]

    def get_trade_history(self, limit: int = 100) -> list[dict[str, Any]]:
        n: int = int(limit)
        history: list[dict[str, Any]] = self.account.history
        return history[-n:]

    def get_last_close(self, symbol: str) -> Optional[float]:
        """
        Return the most recent close price seen for a symbol from trade history,
        or None if no trades have been recorded yet for that symbol.
        Used by core.py to seed a realistic entry price in demo mode.
        """
        for record in reversed(self.account.history):
            if record.get("symbol") == symbol:
                close = record.get("close_price")
                if close is not None:
                    return float(close)
        return None
