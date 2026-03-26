"""
mt5_connector.py — MetaTrader 5 Broker Interface
"""

import logging
from typing import Optional
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

logger = logging.getLogger("agniv.mt5")

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# Symbol mapping: internal name → MT5 symbol name
SYMBOL_MAP = {
    "XAUUSD": "XAUUSD",
    "BTCUSD": "BTCUSD",
}


class MT5Connector:
    def __init__(self):
        self.connected = False

    def connect(self, account: int, password: str, server: str) -> bool:
        if not mt5.initialize():  # type: ignore
            logger.error("MT5 initialize() failed. Is MetaTrader 5 running?")
            return False
        if not mt5.login(account, password=password, server=server):  # type: ignore
            logger.error(f"MT5 login failed: {mt5.last_error()}")  # type: ignore
            mt5.shutdown()  # type: ignore
            return False
        info = mt5.account_info()  # type: ignore
        logger.info(f"MT5 Connected | #{info.login} | ${info.balance:.2f} | {info.server}")
        self.connected = True
        self.auto_discover_symbols()
        return True

    def auto_discover_symbols(self):
        """
        Scans broker's available symbols and maps generic 'XAUUSD' and 'BTCUSD'
        to their exact broker-specific names (e.g., 'XAUUSD.a', 'GOLD', 'BTCUSDm').
        """
        all_symbols = mt5.symbols_get()  # type: ignore
        if not all_symbols:
            return

        gold_candidates = ["GAUUSD", "XAUUSD", "GOLD", "XAUUSD.a", "XAUUSDm", "XAUUSD.r", "XAUUSD.pro"]
        btc_candidates  = ["BTCUSD", "BTCUSDT", "BITCOIN", "BTCUSD.a", "BTCUSDm", "BTCUSD.pro"]

        available_names = [s.name.upper() for s in all_symbols]
        raw_names = [s.name for s in all_symbols]

        # Map Gold
        for cand in gold_candidates:
            if cand.upper() in available_names:
                exact_name = raw_names[available_names.index(cand.upper())]
                mt5.symbol_select(exact_name, True)  # type: ignore
                SYMBOL_MAP["XAUUSD"] = exact_name
                print(f"[Broker] Universal Mapper: mapped XAUUSD -> {exact_name}")
                logger.info(f"[Broker] Universal Mapper: mapped XAUUSD -> {exact_name}")
                break

        # Map BTC
        for cand in btc_candidates:
            if cand.upper() in available_names:
                exact_name = raw_names[available_names.index(cand.upper())]
                mt5.symbol_select(exact_name, True)  # type: ignore
                SYMBOL_MAP["BTCUSD"] = exact_name
                logger.info(f"[Broker] Universal Mapper: mapped BTCUSD -> {exact_name}")
                break

    def disconnect(self):
        mt5.shutdown()  # type: ignore
        self.connected = False

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        tf = TIMEFRAME_MAP.get(timeframe.upper(), mt5.TIMEFRAME_H1)
        mt5_symbol = SYMBOL_MAP.get(symbol, symbol)
        rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, count)  # type: ignore
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    def get_tick(self, symbol: str) -> dict:
        tick = mt5.symbol_info_tick(SYMBOL_MAP.get(symbol, symbol))  # type: ignore
        if tick is None:
            return {}
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

    def get_account_info(self) -> dict:
        info = mt5.account_info()  # type: ignore
        if not info:
            return {}
        return {
            "balance":      info.balance,
            "equity":       info.equity,
            "margin_free":  info.margin_free,
            "profit":       info.profit,
            "currency":     info.currency,
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        mt5_symbol = self.map_symbol(symbol) if symbol else None
        positions = mt5.positions_get(symbol=mt5_symbol) if mt5_symbol else mt5.positions_get()  # type: ignore
        return [p._asdict() for p in positions] if positions else []

    def map_symbol(self, symbol: str) -> str:
        """Returns the broker-specific symbol name (e.g. 'GOLD' for 'XAUUSD')."""
        return SYMBOL_MAP.get(symbol, symbol)

    def place_market_order(self, symbol: str, direction: str, volume: float,
                           sl: float = 0.0, tp: float = 0.0, comment: str = "agniv") -> dict:
        mt5_symbol = SYMBOL_MAP.get(symbol, symbol)
        tick = mt5.symbol_info_tick(mt5_symbol)  # type: ignore
        if tick is None:
            return {"error": "no_tick"}
        is_buy = direction.upper() == "BUY"
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       mt5_symbol,
            "volume":       float(f"{float(volume):.2f}"),  # type-safe volume rounding
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        202601,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # ── Send directly to the broker (User explicitly requested no manual check) ─────
        result = mt5.order_send(request)  # type: ignore
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: [{result.retcode}] {result.comment}")
            return {"error": result.comment, "retcode": result.retcode}
        logger.info(f"✅ Order | {direction} {volume} {symbol} @ {price:.5f} | Ticket#{result.order}")
        return {"ticket": result.order, "price": price, "volume": volume, "direction": direction}

    def close_position(self, ticket: int, volume: float = None) -> bool:
        pos = mt5.positions_get(ticket=ticket)  # type: ignore
        if not pos:
            return False
        p = pos[0]
        
        # If volume not specified, close full position
        close_vol = volume if volume is not None else p.volume
        # Clamp to available volume
        close_vol = min(close_vol, p.volume)
        
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)  # type: ignore
        close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        result = mt5.order_send({  # type: ignore
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       p.symbol,
            "volume":       float(close_vol),
            "type":         close_type,
            "position":     ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        202601,
            "comment":      "ag_close_partial" if volume else "ag_close_full",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Modifies both SL and TP for an existing position. MT5 requires both to keep them."""
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       float(sl),
            "tp":       float(tp),
        }
        result = mt5.order_send(request)  # type: ignore
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5] Modification failed: {result.comment}")
            return False
        return True

    def get_closed_trade_info(self, ticket: int) -> dict:
        """Fetches profit and closing reason for a specific ticket from history."""
        # Look back 1 hour to find the closing deal
        from datetime import datetime, timedelta
        from_time = datetime.now() - timedelta(hours=1)
        deals = mt5.history_deals_get(from_date=from_time, ticket=ticket) # type: ignore
        
        if deals and len(deals) > 0:
            # Usually the last deal for a ticket is the closing one or contains the cumulative profit
            deal = deals[-1]
            return {
                "profit": float(deal.profit),
                "symbol": deal.symbol,
                "magic":  deal.magic,
                "reason": "TP/SL" if deal.reason in [mt5.DEAL_REASON_SL, mt5.DEAL_REASON_TP] else "Manual/Other" # type: ignore
            }
        return {}

    def pip_value(self, symbol: str, lot: float = 1.0) -> float:
        """Approximate pip value in account currency for 1 lot."""
        info = mt5.symbol_info(SYMBOL_MAP.get(symbol, symbol))  # type: ignore
        if info is None:
            return 10.0  # fallback
        # For XAUUSD: 1 pip = 0.01, contract size typically 100
        # For BTCUSD: varies
        return info.trade_tick_value * lot
