"""
binance_connector.py — Binance WebSocket Client for BTC
=======================================================
Provides real-time price, volume, and trade updates for BTCUSDT.
Used for instantaneous scalping triggers.
"""

import json
import logging
import threading
import time
import websocket # type: ignore

logger = logging.getLogger("apexalgo.binance")

class BinanceConnector:
    """
    Connects to Binance Public WebSocket Streams.
    Stream: btc_usdt@ticker or btc_usdt@aggTrade
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@ticker"
        self.ws = None
        self.latest_tick = {}
        self._running = False
        self._thread = None

    def start(self):
        """Starts the WebSocket connection in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()
        logger.info(f"[Binance] WebSocket started for {self.symbol}")

    def stop(self):
        """Stops the WebSocket connection."""
        self._running = False
        if self.ws:
            self.ws.close()
        logger.info("[Binance] WebSocket stopped.")

    def _run_ws(self):
        while self._running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"[Binance] Connection error: {e}")
            
            if self._running:
                logger.info("[Binance] Reconnecting in 5 seconds...")
                time.sleep(5)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # data from @ticker has: 'c' (close/last price), 'v' (volume), 'h' (high), 'l' (low)
            self.latest_tick = {
                "price": float(data.get("c", 0)),
                "volume": float(data.get("v", 0)),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "symbol": data.get("s", "BTCUSDT"),
                "ts": data.get("E", time.time()*1000)
            }
        except Exception as e:
            logger.error(f"[Binance] Error parsing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"[Binance] WS Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"[Binance] WS Closed: {close_status_code} - {close_msg}")

    def get_latest_price(self) -> float:
        return self.latest_tick.get("price", 0.0)

    def get_tick(self) -> dict:
        return self.latest_tick

if __name__ == "__main__":
    # Diagnostic Test
    logging.basicConfig(level=logging.INFO)
    connector = BinanceConnector()
    connector.start()
    try:
        for _ in range(10):
            time.sleep(2)
            print(f"BTC Price: {connector.get_latest_price()}")
    except KeyboardInterrupt:
        pass
    connector.stop()
