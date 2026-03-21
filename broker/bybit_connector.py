"""
bybit_connector.py — Bybit WebSocket Client for BTC (Backup)
===========================================================
Provides real-time price and volume updates for BTCUSDT from Bybit.
Acts as a failover if Binance is unreachable.
"""

import json
import logging
import threading
import time
import websocket # type: ignore

logger = logging.getLogger("apexalgo.bybit")

class BybitConnector:
    """
    Connects to Bybit Public WebSocket Streams (V5).
    Stream: publicV5/tickers.BTCUSDT
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"
        self.ws = None
        self.latest_tick = {}
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()
        logger.info(f"[Bybit] WebSocket started for {self.symbol}")

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()
        logger.info("[Bybit] WebSocket stopped.")

    def _run_ws(self):
        while self._running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"[Bybit] Connection error: {e}")
            
            if self._running:
                time.sleep(5)

    def _on_open(self, ws):
        # Subscribe to tickers
        sub_msg = {
            "op": "subscribe",
            "args": [f"tickers.{self.symbol}"]
        }
        ws.send(json.dumps(sub_msg))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "data" in data:
                ticker = data["data"]
                # data corresponds to Bybit V5 ticker response
                self.latest_tick = {
                    "price": float(ticker.get("lastPrice", 0)),
                    "volume": float(ticker.get("volume24h", 0)),
                    "high": float(ticker.get("highPrice24h", 0)),
                    "low": float(ticker.get("lowPrice24h", 0)),
                    "symbol": ticker.get("symbol", "BTCUSDT"),
                    "ts": data.get("ts", time.time()*1000)
                }
        except Exception as e:
            logger.error(f"[Bybit] Error parsing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"[Bybit] WS Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"[Bybit] WS Closed: {close_status_code} - {close_msg}")

    def get_latest_price(self) -> float:
        return self.latest_tick.get("price", 0.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = BybitConnector()
    connector.start()
    try:
        for _ in range(10):
            time.sleep(2)
            print(f"Bybit BTC Price: {connector.get_latest_price()}")
    except KeyboardInterrupt:
        pass
    connector.stop()
