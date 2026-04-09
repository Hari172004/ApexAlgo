"""
backtester.py — Agni-V Backtrader Module
=============================================
Tests both Scalping and Swing strategies on historical XAUUSD data.
Simulates funded account rules during backtesting.
Reports: win rate, profit factor, max drawdown, Sharpe ratio, prop firm pass/fail.
"""

import logging
import backtrader as bt
import backtrader.analyzers as btanalyzers
import yfinance as yf
import pandas as pd
from datetime import datetime

logger = logging.getLogger("agniv.backtest")


# ──────────────────────────────────────────────────────────────
# Strategy 1: Scalping (RSI + EMA crossover)
# ──────────────────────────────────────────────────────────────

class ScalpStrategy(bt.Strategy):
    params = {
        "ema_fast": 9, "ema_slow": 21,
        "rsi_period": 14, "rsi_ob": 70, "rsi_os": 30,
        "risk_pct": 1.0,
        "daily_loss_limit_pct": 5.0,
        "max_drawdown_pct": 10.0,
    }

    def __init__(self):
        self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)  # type: ignore
        self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)  # type: ignore
        self.rsi      = bt.ind.RSI(period=self.p.rsi_period)  # type: ignore
        self.bb       = bt.ind.BollingerBands(period=20, devfactor=2)  # type: ignore
        self.macd     = bt.ind.MACD()
        self.order    = None
        self.daily_start_value = None
        self.start_value = None

    def start(self):
        self.start_value = self.broker.getvalue()
        self.daily_start_value = self.start_value

    def next(self):
        val = self.broker.getvalue()

        # Daily loss limit check
        daily_loss_pct = (1 - val / self.daily_start_value) * 100 if self.daily_start_value else 0
        total_dd_pct   = (1 - val / self.start_value) * 100 if self.start_value else 0

        if daily_loss_pct >= self.p.daily_loss_limit_pct:
            if self.position:
                self.close()
            return
        if total_dd_pct >= self.p.max_drawdown_pct:
            if self.position:
                self.close()
            return

        if self.order:
            return

        buy_signal  = (
            self.ema_fast[0] > self.ema_slow[0] and
            self.rsi[0] < self.p.rsi_os and
            self.macd.macd[0] > self.macd.signal[0]
        )
        sell_signal = (
            self.ema_fast[0] < self.ema_slow[0] and
            self.rsi[0] > self.p.rsi_ob and
            self.macd.macd[0] < self.macd.signal[0]
        )

        risk_amount = val * (self.p.risk_pct / 100)
        size = max(1, int(risk_amount / (self.data.close[0] * 0.01)))

        if not self.position:
            if buy_signal:
                self.order = self.buy(size=size)
            elif sell_signal:
                self.order = self.sell(size=size)
        else:
            if self.position.size > 0 and sell_signal:
                self.order = self.close()
            elif self.position.size < 0 and buy_signal:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            if trade.pnlcomm > 0:
                logger.debug(f"WIN  ${trade.pnlcomm:+.2f}")
            else:
                logger.debug(f"LOSS ${trade.pnlcomm:+.2f}")


# ──────────────────────────────────────────────────────────────
# Strategy 2: Swing (EMA 50 trend + ATR)
# ──────────────────────────────────────────────────────────────

class SwingStrategy(bt.Strategy):
    params = {
        "ema_trend": 50, "ema_fast": 20,
        "atr_period": 14, "adx_period": 14,
        "risk_pct": 1.0,
        "daily_loss_limit_pct": 5.0,
        "max_drawdown_pct": 10.0,
    }

    def __init__(self):
        self.ema_trend = bt.ind.EMA(period=self.p.ema_trend)  # type: ignore
        self.ema_fast  = bt.ind.EMA(period=self.p.ema_fast)  # type: ignore
        self.atr       = bt.ind.ATR(period=self.p.atr_period)  # type: ignore
        self.adx       = bt.ind.DirectionalMovement(period=self.p.adx_period)  # type: ignore
        self.order     = None
        self.start_value = None
        self.daily_start_value = None
        self.entry_price = None

    def start(self):
        self.start_value = self.broker.getvalue()
        self.daily_start_value = self.start_value

    def next(self):
        val = self.broker.getvalue()
        daily_dd = (1 - val / self.daily_start_value) * 100 if self.daily_start_value else 0
        total_dd = (1 - val / self.start_value) * 100 if self.start_value else 0

        if daily_dd >= self.p.daily_loss_limit_pct or total_dd >= self.p.max_drawdown_pct:
            if self.position:
                self.close()
            return

        if self.order:
            return

        adx_val = self.adx.adx[0] if hasattr(self.adx, "adx") else 25

        in_uptrend   = self.data.close[0] > self.ema_trend[0] and self.ema_fast[0] > self.ema_trend[0]
        in_downtrend = self.data.close[0] < self.ema_trend[0] and self.ema_fast[0] < self.ema_trend[0]

        risk_amount = val * (self.p.risk_pct / 100)
        atr_val = max(self.atr[0], 0.0001)
        size = max(1, int(risk_amount / (atr_val * 100)))

        if not self.position and adx_val > 20:
            if in_uptrend:
                self.order = self.buy(size=size)
                self.entry_price = self.data.close[0]
            elif in_downtrend:
                self.order = self.sell(size=size)
                self.entry_price = self.data.close[0]
        elif self.position:
            # Move to breakeven at 1R
            if self.position.size > 0 and self.entry_price:
                profit = self.data.close[0] - self.entry_price
                if profit >= atr_val * 1.5 and in_downtrend:
                    self.order = self.close()
            elif self.position.size < 0 and self.entry_price:
                profit = self.entry_price - self.data.close[0]
                if profit >= atr_val * 1.5 and in_uptrend:
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


# ──────────────────────────────────────────────────────────────
# Data Loader
# ──────────────────────────────────────────────────────────────

def load_data(symbol: str, period: str = "2y", interval: str = "1h") -> bt.feeds.PandasData:
    """
    symbol: 'GC=F' (XAUUSD)
    interval: '5m', '1h', '1d'
    """
    ticker_map = {
        "XAUUSD": "GC=F",
        "GC=F":   "GC=F",
    }
    yf_symbol = ticker_map.get(symbol, symbol)
    logger.info(f"[Backtest] Downloading {yf_symbol} | Period={period} | Interval={interval}")
    df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
    df.columns = [c.lower() for c in df.columns]  # type: ignore
    df = df.rename(columns={"adj close": "close"})  # type: ignore
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    feed = bt.feeds.PandasData(dataname=df)  # type: ignore
    return feed


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

def run_backtest(symbol: str = "XAUUSD",
                 strategy_name: str = "scalp",
                 starting_cash: float = 10_000.0,
                 period: str = "2y",
                 interval: str = "1h",
                 risk_pct: float = 1.0,
                 daily_loss_limit_pct: float = 5.0,
                 max_drawdown_pct: float = 10.0) -> dict:
    """
    Full backtest runner.
    Returns a dict with all metrics including prop firm pass/fail simulation.
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.0002)  # 0.02% commission

    data = load_data(symbol, period, interval)
    cerebro.adddata(data)

    StratClass = ScalpStrategy if strategy_name == "scalp" else SwingStrategy
    cerebro.addstrategy(
        StratClass,
        risk_pct=risk_pct,
        daily_loss_limit_pct=daily_loss_limit_pct,
        max_drawdown_pct=max_drawdown_pct,
    )

    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio,    _name="sharpe",   riskfreerate=0.02)
    cerebro.addanalyzer(btanalyzers.DrawDown,        _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer,   _name="trades")
    cerebro.addanalyzer(btanalyzers.Returns,         _name="returns")

    logger.info(f"[Backtest] Running {strategy_name} on {symbol} | Cash=${starting_cash:,.2f}")
    results = cerebro.run()
    strat   = results[0]

    final_value = cerebro.broker.getvalue()
    net_profit  = final_value - starting_cash
    net_pnl_pct = (net_profit / starting_cash) * 100

    # Trade metrics
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("total", 0)
    won   = ta.get("won",  {}).get("total", 0)
    lost  = ta.get("lost", {}).get("total", 0)
    win_rate = round((won / total_trades * 100) if total_trades > 0 else 0, 1)

    gross_profit = ta.get("won",  {}).get("pnl", {}).get("total", 0)
    gross_loss   = abs(ta.get("lost", {}).get("pnl", {}).get("total", 1))
    profit_factor = round(gross_profit / gross_loss if gross_loss > 0 else 0, 2)

    # Drawdown and Sharpe
    dd  = strat.analyzers.drawdown.get_analysis()
    max_dd_pct = round(dd.get("max", {}).get("drawdown", 0), 2)
    sharpe_raw = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe = round(sharpe_raw, 3) if sharpe_raw else 0.0

    # Prop firm simulation
    prop_pass = (
        net_pnl_pct >= 10.0 and
        max_dd_pct <= max_drawdown_pct and
        win_rate >= 40
    )

    report = {
        "symbol":           symbol,
        "strategy":         strategy_name,
        "starting_cash":    starting_cash,
        "final_value":      round(final_value, 2),
        "net_profit":       round(net_profit, 2),
        "net_pnl_pct":      round(net_pnl_pct, 2),
        "total_trades":     total_trades,
        "wins":             won,
        "losses":           lost,
        "win_rate_pct":     win_rate,
        "profit_factor":    profit_factor,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio":     sharpe,
        "prop_firm_pass":   prop_pass,
        "period":           period,
        "interval":         interval,
    }

    logger.info(
        f"[Backtest] DONE | {symbol} {strategy_name} | "
        f"PnL={net_pnl_pct:+.2f}% | WinRate={win_rate:.1f}% | "
        f"Drawdown={max_dd_pct:.2f}% | Sharpe={sharpe:.3f} | "
        f"PropFirmPass={'✅' if prop_pass else '❌'}"
    )
    return report


# ──────────────────────────────────────────────────────────────
# CLI Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    for sym in ["XAUUSD"]:
        for strat in ["scalp", "swing"]:
            result = run_backtest(symbol=sym, strategy_name=strat, period="2y", interval="1h")
            print(json.dumps(result, indent=2))
            print("─" * 60)
