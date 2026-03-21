"""
session_filter.py — Strict Kill Zone & Liquidity Enforcement
Blocks all trades outside of precise algorithmic time windows to avoid fakeouts and dead markets.
"""

import logging
from datetime import datetime, time as dtime
import pytz

logger = logging.getLogger("apexalgo.filters.session")

class SessionFilter:
    def __init__(self):
        self.gmt_tz = pytz.timezone("GMT")

    def _is_time_in_range(self, t: dtime, start: dtime, end: dtime) -> bool:
        """Returns True if time t is strictly between start and end."""
        if start <= end:
            return start <= t <= end
        else:
            return start <= t or t <= end

    def is_kill_zone_active(self, symbol: str) -> bool:
        """Checks if current GMT time falls inside the allowed Kill Zones."""
        now_gmt = datetime.now(self.gmt_tz)
        current_time = now_gmt.time()
        
        is_xau = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        is_btc = "BTC" in symbol.upper() or "BITCOIN" in symbol.upper()

        if is_xau:
            # Gold Kill Zones (GMT/UTC):
            # London: 07:00 to 10:00
            # NY: 12:00 to 15:00
            # London Close: 14:00 to 16:00
            london_kz = self._is_time_in_range(current_time, dtime(7, 0), dtime(10, 0))
            ny_kz     = self._is_time_in_range(current_time, dtime(12, 0), dtime(15, 0))
            ldn_close = self._is_time_in_range(current_time, dtime(14, 0), dtime(16, 0))
            
            if london_kz or ny_kz or ldn_close:
                return True
                
        elif is_btc:
            # BTC Kill Zones (GMT/UTC):
            # NY Open: 13:00 to 17:00
            # London Open: 07:00 to 10:00
            # Asian Breakout: 00:00 to 03:00
            ny_open    = self._is_time_in_range(current_time, dtime(13, 0), dtime(17, 0))
            ldn_open   = self._is_time_in_range(current_time, dtime(7, 0), dtime(10, 0))
            asian_open = self._is_time_in_range(current_time, dtime(0, 0), dtime(3, 0))
            
            if ny_open or ldn_open or asian_open:
                return True

        return False

    def is_liquidity_safe(self, symbol: str) -> bool:
        """
        Detects low liquidity periods:
        1. Asian Session for Gold (22:00 to 06:00 GMT)
        2. Sunday first 2 hours (21:00 to 23:00 GMT depending on DST)
        3. Friday close (21:30 to 22:00 GMT)
        """
        now_gmt = datetime.now(self.gmt_tz)
        current_time = now_gmt.time()
        weekday = now_gmt.weekday()

        # 1. Asian Session for Gold
        is_xau = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        if is_xau:
            if self._is_time_in_range(current_time, dtime(22, 0), dtime(6, 0)):
                logger.warning(f"[SESSION] {symbol} inside Asian session dead zone. Blocking.")
                return False

        # 2. Sunday Open
        if weekday == 6: # Sunday
            if self._is_time_in_range(current_time, dtime(21, 0), dtime(23, 0)):
                logger.warning(f"[SESSION] Sunday market open erratic gap zone. Blocking.")
                return False

        # 3. Friday Close
        if weekday == 4: # Friday
            if self._is_time_in_range(current_time, dtime(21, 30), dtime(23, 59)):
                logger.warning(f"[SESSION] Friday market close dumping zone. Blocking.")
                return False

        # Note: Bank holiday calendars would be checked here via ForexFactory API
        # but to keep it self-contained without risking third-party API keys today,
        # we rely on the strict time checks above.
        
        return True

    def validate_trade_window(self, symbol: str) -> bool:
        """Master check. Both strict Kill Zone and safe liquidity."""
        if not self.is_liquidity_safe(symbol):
            return False
            
        if not self.is_kill_zone_active(symbol):
            logger.warning(f"[SESSION] {symbol} is completely outside Kill Zones. No trades allowed.")
            return False
            
        logger.info(f"[SESSION] {symbol} Kill Zone verified. Trade execution permitted.")
        return True

global_session_filter = SessionFilter()
