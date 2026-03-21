"""
logger.py — ApexAlgo Trade Logger & Alerts
=============================================
Sends alerts via Telegram and Email.
Logs every trade to Supabase. Sends daily reports.
"""

import os
import smtplib
import logging
import requests # type: ignore
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger("apexalgo.alerts")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class AlertManager:
    def __init__(self,
                 telegram_token: str  = "",
                 telegram_chat_id: str = "",
                 gmail_user: str      = "",
                 gmail_password: str  = "",
                 supabase_client      = None):
        self.telegram_token   = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.gmail_user       = gmail_user
        self.gmail_password   = gmail_password
        self.supabase         = supabase_client

    # ── Telegram ──────────────────────────────────────────────

    def send_telegram(self, message: str, is_alert: bool = False) -> bool:
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        prefix = "⚠️ *ApexAlgo Risk Alert*\n" if is_alert else "🚀 *ApexAlgo Signal*\n"
        text = prefix + message
        url  = TELEGRAM_API.format(token=self.telegram_token)
        try:
            resp = requests.post(url, json={
                "chat_id":    self.telegram_chat_id,
                "text":       text,
                "parse_mode": "Markdown",
            }, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"[Alert] Telegram error: {e}")
            return False

    # ── Email ─────────────────────────────────────────────────

    def send_email(self, subject: str, body: str, to: Optional[str] = None) -> bool:
        if not self.gmail_user or not self.gmail_password:
            return False
        recipient = to or self.gmail_user
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"ApexAlgo Bot <{self.gmail_user}>"
        msg["To"]      = recipient
        msg.attach(MIMEText(body, "html"))
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.gmail_user, self.gmail_password)
                server.sendmail(self.gmail_user, recipient, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"[Alert] Email error: {e}")
            return False

    # ── Trade Alert ───────────────────────────────────────────

    def trade_opened(self, trade: dict, sentiment: str = "NEUTRAL"):
        symbol    = trade.get("symbol", "")
        direction = trade.get("direction", "")
        price     = trade.get("price", 0)
        sl        = trade.get("sl", 0)
        tp        = trade.get("tp", 0)
        strategy  = trade.get("strategy", "")
        mode      = trade.get("mode", "")
        emoji     = "🟢" if direction == "BUY" else "🔴"
        msg = (
            f"{emoji} *{direction} {symbol}*\n"
            f"Strategy: `{strategy}` | Mode: `{mode}`\n"
            f"Entry: `{price:.5f}`\n"
            f"SL: `{sl:.5f}` | TP: `{tp:.5f}`\n"
            f"Sentiment: `{sentiment}`\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.send_telegram(msg)

    def trade_closed(self, trade: dict):
        symbol    = trade.get("symbol", "")
        pnl       = trade.get("pnl", 0)
        exit_rsn  = trade.get("exit_reason", "manual")
        emoji     = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} *CLOSED {symbol}*\n"
            f"PnL: `${pnl:+.2f}`\n"
            f"Exit: `{exit_rsn}`\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.send_telegram(msg)

    def risk_alert(self, message: str):
        logger.warning(f"[Alert] ⚠️ RISK: {message}")
        self.send_telegram(message, is_alert=True)

    # ── Supabase Logging ──────────────────────────────────────

    def log_trade_to_db(self, trade: dict, user_id: str = "system"):
        if not self.supabase:
            return
        try:
            payload = {
                "user_id":      user_id,
                "symbol":       trade.get("symbol"),
                "strategy":     trade.get("strategy"),
                "mode":         trade.get("mode"),
                "direction":    trade.get("direction"),
                "entry_price":  trade.get("price"),
                "exit_price":   trade.get("close_price"),
                "sl":           trade.get("sl"),
                "tp":           trade.get("tp"),
                "volume":       trade.get("volume"),
                "pnl":          trade.get("pnl"),
                "win":          trade.get("pnl", 0) >= 0,
                "exit_reason":  trade.get("exit_reason"),
                "sentiment":    trade.get("sentiment"),
                "opened_at":    trade.get("open_time"),
                "closed_at":    trade.get("close_time"),
            }
            self.supabase.table("trades").insert(payload).execute()
        except Exception as e:
            logger.error(f"[Alert] DB log error: {e}")

    # ── Daily Report ──────────────────────────────────────────

    def send_daily_report(self, stats: dict, funded_report: Optional[dict] = None,
                          recipient_email: Optional[str] = None):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        lines = [
            f"<h2>📊 ApexAlgo Daily Report — {now}</h2>",
            f"<b>Balance:</b> ${stats.get('balance', 0):,.2f}<br>",
            f"<b>Today PnL:</b> ${stats.get('today_pnl', 0):+.2f}<br>",
            f"<b>Trades:</b> {stats.get('trade_count_today', 0)} "
            f"(W:{stats.get('wins_today',0)} / L:{stats.get('losses_today',0)})<br>",
            f"<b>Win Rate:</b> {stats.get('win_rate_today', 0):.1f}%<br>",
        ]
        if funded_report:
            lines += [
                "<hr><h3>🏦 Funded Account Progress</h3>",
                f"<b>Firm:</b> {funded_report.get('firm')} — {funded_report.get('phase')}<br>",
                f"<b>Profit Progress:</b> {funded_report.get('profit_progress_pct', 0):.1f}% toward target<br>",
                f"<b>Drawdown Used:</b> {funded_report.get('drawdown_used_pct', 0):.1f}%<br>",
                f"<b>Days Remaining:</b> {funded_report.get('days_remaining', 0)}<br>",
            ]
        body = "\n".join(lines)
        subject = f"ApexAlgo Daily Report — {now}"
        self.send_email(subject, body, recipient_email)

        tg_msg = (
            f"📊 *Daily Report — {now}*\n"
            f"Balance: `${stats.get('balance',0):,.2f}`\n"
            f"Today PnL: `${stats.get('today_pnl',0):+.2f}`\n"
            f"Win Rate: `{stats.get('win_rate_today',0):.1f}%`"
        )
        if funded_report:
            tg_msg += (
                f"\n🏦 *{funded_report.get('firm')}* | {funded_report.get('phase')}\n"
                f"Progress: `{funded_report.get('profit_progress_pct',0):.1f}%` to target\n"
                f"Drawdown: `{funded_report.get('drawdown_used_pct',0):.1f}%` used"
            )
        self.send_telegram(tg_msg)
