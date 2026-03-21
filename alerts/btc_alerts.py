"""
btc_alerts.py — BTC Telegram Alerts
===================================
Handles all Bitcoin-specific notifications.
"""

import logging
from datetime import datetime

logger = logging.getLogger("apexalgo.btc_alerts")

class BTCAlerts:
    """Sends BTC-specific alerts to Telegram."""

    def __init__(self, alert_manager):
        self.alerts = alert_manager

    def signal_detected(self, strategy: str, direction: str, price: float, reason: str):
        emoji = "🟡"
        msg = (
            f"{emoji} *ApexAlgo BTC Signal*\n"
            f"Strategy: `{strategy}` | Direction: `{direction}`\n"
            f"Price: `{price:.2f}`\n"
            f"Reason: _{reason}_\n"
            f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        )
        self.alerts.send_telegram(msg)

    def whale_alert(self, amount: float, source: str):
        msg = (
            f"🐋 *Whale Move Detected*\n"
            f"Amount: `{amount:.2f} BTC` sent to `{source}`\n"
            f"⚠️ Caution recommended for new BTC trades."
        )
        self.alerts.send_telegram(msg, is_alert=True)

    def extreme_sentiment_alert(self, score: float, label: str):
        msg = (
            f"🚨 *BTC Sentiment Alert*\n"
            f"Mood: `{label}` (Score: {score:.2f})\n"
            f"⚠️ Market extremes detected. Reducing trade sizes."
        )
        self.alerts.send_telegram(msg, is_alert=True)
