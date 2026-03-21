"""
gold_alerts.py -- Full Telegram alert system for Gold (XAUUSD)
Alert types: Signal, DXY warning, News pause, Kill Zone open,
             Fundamental score, Spread warning, ETF flow, Geopolitical, Daily report.
"""

import logging
from datetime import datetime, timezone, date

logger = logging.getLogger("apexalgo.gold_alerts")

EMOJI = {
    "gold":         "🥇",
    "dxy":          "💵",
    "news":         "🚨",
    "lightning":    "⚡",
    "chart":        "📊",
    "warning":      "⚠️",
    "etf":          "📈",
    "globe":        "🌍",
    "report":       "📋",
    "buy":          "🟢",
    "sell":         "🔴",
    "hold":         "⏸️",
}


class GoldAlerts:
    def __init__(self, alert_manager):
        """
        alert_manager: existing AlertManager instance from logger.py.
        Wraps it to send Gold-specific Telegram messages.
        """
        self.alerts = alert_manager

    # ── 1. Trade Signal ───────────────────────────────────────────────────

    def signal_alert(self, symbol: str, signal: str, strategy: str, reason: str,
                     entry: float = 0.0, sl: float = 0.0, tp: float = 0.0):
        """Fire on every Gold BUY/SELL signal."""
        direction_emoji = EMOJI["buy"] if signal == "BUY" else EMOJI["sell"]
        msg = (
            f"{EMOJI['gold']} *ApexAlgo Gold Signal {EMOJI['gold']}*\n"
            f"{'─' * 30}\n"
            f"{direction_emoji} Symbol:   `{symbol}`\n"
            f"{direction_emoji} Action:   `{signal}`\n"
            f"📐 Strategy: `{strategy}`\n"
            f"📝 Reason:   _{reason}_\n"
        )
        if entry > 0:
            msg += (
                f"💰 Entry: `{entry:.3f}`\n"
                f"🛑 SL:    `{sl:.3f}`\n"
                f"🎯 TP:    `{tp:.3f}`\n"
            )
        msg += f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        self._send(msg)

    # ── 2. DXY Warning ────────────────────────────────────────────────────

    def dxy_warning(self, dxy_val: float, change_pct: float):
        """Fire when DXY spikes significantly — caution for Gold longs."""
        severity = "Spiking" if change_pct > 0.3 else "Rising"
        msg = (
            f"{EMOJI['dxy']} *DXY {severity} — Gold Caution Mode Active*\n"
            f"DXY: `{dxy_val:.2f}` ({'+' if change_pct > 0 else ''}{change_pct:.2f}%)\n"
            f"⚠️ Gold long positions at risk. Review open trades."
        )
        self._send(msg, alert=True)

    # ── 3. News Pause ─────────────────────────────────────────────────────

    def news_pause_alert(self, event_name: str, minutes_to: int = 30):
        """Fire when high-impact news is approaching."""
        msg = (
            f"{EMOJI['news']} *High Impact News — Gold Bot Paused for Safety*\n"
            f"Event: `{event_name}`\n"
            f"Pausing gold trading for `{minutes_to}` minutes."
        )
        self._send(msg, alert=True)

    def news_resume_alert(self, event_name: str):
        """Fire when news blackout window ends."""
        msg = (
            f"✅ *Gold Trading Resumed* — `{event_name}` data released.\n"
            f"ApexAlgo scanning gold markets again."
        )
        self._send(msg)

    # ── 4. Kill Zone Open ─────────────────────────────────────────────────

    def session_alert(self, session: str, is_killzone: bool):
        """Fire when London or NY Kill Zone opens."""
        if not is_killzone:
            return
        session_display = "London" if session == "LONDON" else "New York"
        msg = (
            f"{EMOJI['lightning']} *{session_display} Kill Zone Open*\n"
            f"ApexAlgo scanning gold opportunities.\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%H:%M')} UTC"
        )
        self._send(msg)

    # ── 5. Fundamental Alert ──────────────────────────────────────────────

    def fundamental_alert(self, score: float, bias: str, dxy: float, us10y: float, vix: float):
        """Fire when macro bias shifts to bearish — reducing position size."""
        if bias == "NEUTRAL":
            return
        indicator_emoji = EMOJI["chart"] if bias == "BULLISH" else EMOJI["warning"]
        msg = (
            f"{indicator_emoji} *Gold Fundamental Score: {bias}*\n"
            f"{'─' * 28}\n"
            f"Score: `{score:+.0f}` | DXY: `{dxy:.2f}` | US10Y: `{us10y:.2f}%` | VIX: `{vix:.1f}`\n"
        )
        if bias == "BEARISH":
            msg += "⬇️ Reducing gold position size automatically."
        else:
            msg += "⬆️ Gold macro conditions bullish — normal sizing."
        self._send(msg)

    # ── 6. Spread Warning ─────────────────────────────────────────────────

    def spread_alert(self, spread_points: float, threshold: float = 3.0):
        """Fire when gold spread is too wide for profitable trading."""
        msg = (
            f"{EMOJI['warning']} *Gold Spread Too High — Paused*\n"
            f"Current spread: `{spread_points:.2f}` pts (`{spread_points * 10:.0f}` pips)\n"
            f"Threshold: `{threshold:.2f}` pts. Waiting for normal spread to resume."
        )
        self._send(msg, alert=True)

    # ── 7. ETF Flow Alert ─────────────────────────────────────────────────

    def etf_flow_alert(self, flow_pct: float, direction: str):
        """Fire when GLD ETF shows significant inflow or outflow."""
        flow_label = "Inflow" if direction == "IN" else "Outflow"
        bias_label = "Gold Bullish Bias Active" if direction == "IN" else "Gold Risk — Smart Money Exiting"
        msg = (
            f"{EMOJI['etf']} *GLD ETF {flow_label} Detected*\n"
            f"Change: `{flow_pct:+.2f}%` today\n"
            f"📌 {bias_label}"
        )
        self._send(msg)

    # ── 8. Geopolitical Alert ─────────────────────────────────────────────

    def geopolitical_alert(self, event: str, severity: str = "HIGH"):
        """Fire when geopolitical risk triggers safe-haven demand for gold."""
        msg = (
            f"{EMOJI['globe']} *Geopolitical Risk Detected [{severity}]*\n"
            f"Event: _{event}_\n"
            f"🛡️ Gold Safe Haven Mode Active. Bias shifted Bullish."
        )
        self._send(msg, alert=True)

    # ── 9. Daily Report ───────────────────────────────────────────────────

    def daily_report(self, trades: int, pnl: float, win_rate: float,
                     fund_bias: str, sentiment: str, session_summary: str):
        """End-of-day summary report."""
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{EMOJI['gold']} *ApexAlgo Gold Daily Report {EMOJI['gold']}*\n"
            f"{'─' * 32}\n"
            f"📅 Date:       `{date.today().isoformat()}`\n"
            f"{pnl_emoji} PnL:        `{'+'if pnl>=0 else ''}{pnl:.2f}` USD\n"
            f"📊 Win Rate:   `{win_rate:.0f}%`\n"
            f"🔢 Trades:    `{trades}`\n"
            f"🌍 Fund Bias:  `{fund_bias}`\n"
            f"📰 Sentiment:  `{sentiment}`\n"
            f"🕐 Sessions:   _{session_summary}_"
        )
        self._send(msg)

    # ── Internal sender ───────────────────────────────────────────────────

    def _send(self, message: str, alert: bool = False):
        """Delegate to the AlertManager — supports both Telegram and email."""
        try:
            if hasattr(self.alerts, "send_telegram"):
                self.alerts.send_telegram(message)
            else:
                logger.info(f"[GoldAlerts] {message}")
        except Exception as e:
            logger.error(f"[GoldAlerts] Failed to send alert: {e}")
