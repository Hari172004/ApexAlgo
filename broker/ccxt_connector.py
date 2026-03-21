"""
ccxt_connector.py — Binance / ByBit connector via CCXT (Crypto exchanges)
"""

import logging
import ccxt
import pandas as pd
from typing import Optional

logger = logging.getLogger("apexalgo.ccxt")

SYMBOL_MAP = {
    "BTCUSD":  "BTC/USDT",
    "XAUUSD":  "XAU/USDT",  # If exchange supports it
}


class CCXTConnector:
    def __init__(self, exchange_name: str, api_key: str, secret: str, testnet: bool = False):
        exchange_class = getattr(ccxt, exchange_name.lower(), None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{exchange_name}' not supported by CCXT")
        self.exchange: ccxt.Exchange = exchange_class({
            "apiKey":          api_key,
            "secret":          secret,
            "enableRateLimit": True,
            "options":         {"defaultType": "future"},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        self.name = exchange_name.lower()
        self.exchange.load_markets()
        logger.info(f"CCXT | {exchange_name} | Testnet={testnet}")

    def _resolve(self, symbol: str) -> str:
        return SYMBOL_MAP.get(symbol, symbol)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """timeframe: '1m', '5m', '1h', '4h', '1d'"""
        try:
            raw = self.exchange.fetch_ohlcv(self._resolve(symbol), timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df.set_index("time", inplace=True)
            return df.astype(float)
        except Exception as e:
            logger.error(f"CCXT get_ohlcv error: {e}")
            return pd.DataFrame()

    def get_ticker(self, symbol: str) -> dict:
        try:
            t = self.exchange.fetch_ticker(self._resolve(symbol))
            return {"bid": t["bid"], "ask": t["ask"], "last": t["last"]}
        except Exception as e:
            logger.error(f"CCXT get_ticker error: {e}")
            return {}

    def get_balance(self) -> dict:
        try:
            b = self.exchange.fetch_balance()
            return {"free": b["USDT"]["free"], "total": b["USDT"]["total"]}
        except Exception as e:
            logger.error(f"CCXT get_balance error: {e}")
            return {}

    def place_market_order(self, symbol: str, direction: str,
                           amount: float, sl: Optional[float] = None, tp: Optional[float] = None) -> dict:
        """direction: 'BUY' or 'SELL'. amount in contracts/base currency."""
        side = direction.lower()
        try:
            params = {}
            if sl:
                params["stopLoss"] = {"type": "market", "price": sl}
            if tp:
                params["takeProfit"] = {"type": "market", "price": tp}
            order = self.exchange.create_market_order(self._resolve(symbol), side, amount, params=params)  # type: ignore
            logger.info(f"✅ CCXT | {side.upper()} {amount} {symbol} | ID={order['id']}")
            return {"id": order["id"], "price": order.get("average", 0), "amount": amount, "direction": direction}
        except Exception as e:
            logger.error(f"CCXT place_market_order error: {e}")
            return {"error": str(e)}

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        try:
            syms = [self._resolve(symbol)] if symbol else None
            positions = self.exchange.fetch_positions(syms)
            return [p for p in positions if abs(p.get("contractSize", p.get("contracts", 0))) > 0]
        except Exception as e:
            logger.error(f"CCXT get_open_positions error: {e}")
            return []

    def close_position(self, symbol: str, direction: str, amount: float) -> dict:
        """Close by placing a reverse market order."""
        close_dir = "SELL" if direction.upper() == "BUY" else "BUY"
        return self.place_market_order(symbol, close_dir, amount)
