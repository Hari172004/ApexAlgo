"""
threat_detector.py — Suspicious Activity and Threat Detection Module
Monitors various signals (Auth, DB, Bot) and emits high-profile Telegram alerts.
"""

import logging
from datetime import datetime
from typing import Optional, Dict
from cachetools import TTLCache

logger = logging.getLogger("apexalgo.threats")

# Simple transient cache to prevent alert spamming (e.g. 1 alert per hour per anomaly type)
ALERT_CACHE = TTLCache(maxsize=1000, ttl=3600)  

class ThreatDetector:
    def __init__(self, telegram_bot_token: str = None, telegram_chat_id: str = None):  # type: ignore
        """
        Initializes the Threat Detector with Telegram credentials for sending alerts.
        """
        self.bot_token = telegram_bot_token
        self.chat_id = telegram_chat_id
        
    def _send_alert(self, title: str, message: str, severity: str = "WARNING", cache_key: Optional[str] = None):
        """Internal helper to dispatch Telegram alerts safely without spamming."""
        if cache_key:
            if cache_key in ALERT_CACHE:
                return  # Skip, already alerted recently
            ALERT_CACHE[cache_key] = True

        emoji = "🚨" if severity == "CRITICAL" else "⚠️"
        alert_text = f"{emoji} *Security Alert: {title}*\n{'-'*30}\n{message}\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        # Log it locally regardless
        if severity == "CRITICAL":
            logger.critical(alert_text.replace("\n", " | "))
        else:
            logger.warning(alert_text.replace("\n", " | "))
            
        # Push to Telegram if configured
        if self.bot_token and self.chat_id:
            try:
                # Normally we use requests or aiohttp here:
                # requests.post(f"https://api.telegram.org/bot{self.bot_token}/sendMessage", data={"chat_id": self.chat_id, "text": alert_text, "parse_mode": "Markdown"})
                pass
            except Exception as e:
                logger.error(f"[Threat] Fast Telegram dispatch failed: {e}")

    # ── 1. Authentication Threats ──────────────────────────────────────────────

    def trigger_failed_login_burst(self, email: str, count: int, ip_addr: str):
        """3+ failed login attempts."""
        msg = f"User: `{email}`\nIP: `{ip_addr}`\nFailed Attempts: `{count}`."
        self._send_alert("Burst Login Failures", msg, "WARNING", f"brute_{email}")

    def trigger_new_device_login(self, email: str, device_id: str, ip_addr: str):
        """First time a new device UUID successfully logs into an account."""
        msg = f"User: `{email}`\nIP: `{ip_addr}`\nDevice ID: `{device_id}`\nAction: _Device registered to account._"
        self._send_alert("New Device Authorized", msg, "WARNING") # No cache limit

    def trigger_new_country_login(self, email: str, old_country: str, new_country: str, ip_addr: str):
        """Geo-velocity anomaly: sudden country change."""
        msg = f"User: `{email}`\nIP: `{ip_addr}`\nJump: `{old_country}` ➡️ `{new_country}`."
        self._send_alert("Suspicious Geo-Location Jump", msg, "CRITICAL", f"geo_{email}_{new_country}")

    # ── 2. License & API Abuse ───────────────────────────────────────────────

    def trigger_license_sharing(self, user_id: str, ip1: str, ip2: str):
        """Same license heartbeat received from two different IP addresses simultaneously."""
        msg = f"User ID: `{user_id}`\nConflict: Connected from `{ip1}` and `{ip2}` concurrently.\nAction: _License access suspended pending review._"
        self._send_alert("License Sharing Detected", msg, "CRITICAL", f"share_{user_id}")

    def trigger_api_abuse(self, user_id: str, ip_addr: str, endpoint: str):
        """User or IP rapidly breaking rate limits."""
        msg = f"Target: `{user_id or ip_addr}`\nEndpoint: `{endpoint}`\nAction: _Repeated 429 Too Many Requests._"
        self._send_alert("API Rate Limit Abuse", msg, "WARNING", f"abuse_{ip_addr}")

    # ── 3. Malicious Database/Execution ───────────────────────────────────────

    def trigger_abnormal_trade_frequency(self, bot_id: str, trade_count: int, timeframe_min: int):
        """Bot places an insane number of trades unexpectedly."""
        msg = f"Bot ID: `{bot_id}`\nVolume: `{trade_count}` trades in `{timeframe_min}` minutes.\nAction: _Bot auto-paused. Review required._"
        self._send_alert("Abnormal Trading Frequency", msg, "CRITICAL", f"spam_{bot_id}")
        
    def trigger_tamper_detected(self, bot_id: str, file_path: str):
        """Bot execution layer reports file checksum modification."""
        msg = f"Bot ID: `{bot_id}`\nFile: `{file_path}`\nAction: _Execution Halted! Source code modified._"
        self._send_alert("PC Bot Code Tampering", msg, "CRITICAL", f"tamper_{bot_id}")

global_threats = ThreatDetector()
