"""
backtest/optimize.py — Strategy Optimizer
=============================================
Runs a Parameter Grid Search over the trading strategies using Backtrader.
Ensures that any optimized parameter set does not violate Funded Account rules
(e.g. 10% max drawdown). Saves the best parameters to `data/optimized_params.json`
so the live bot can automatically load and use them.
"""

import os
import json
import logging
import pandas as pd # type: ignore
import yfinance as yf # type: ignore
import backtrader as bt # type: ignore
import backtrader.analyzers as btanalyzers # type: ignore

# Import the backtrader strategy implementations from backtester.py
from backtest.backtester import ScalpStrategy, SwingStrategy # type: ignore

# Suppress debug logs from the strategies during the heavy optimization loop
logging.getLogger("apexalgo").setLevel(logging.WARNING)
logger = logging.getLogger("apexalgo.optimizer")
logger.setLevel(logging.INFO)

# Stream handler for console
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(ch)

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized_params.json")

def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    logger.info(f"Downloading {period} of {interval} data for {symbol}...")
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if df.empty:  # type: ignore
        raise ValueError(f"Failed to fetch {symbol} data from yfinance")
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]  # type: ignore
    if "adj close" in df.columns:  # type: ignore
        df = df.rename(columns={"adj close": "close"})  # type: ignore
    df = df[["open", "high", "low", "close", "volume"]].dropna()  # type: ignore
    return df

def run_optimization(strategy_class, data: pd.DataFrame, param_grid: dict):
    cerebro = bt.Cerebro(optreturn=False)  # type: ignore
    
    # Configure exact same starting conditions as funded account simulator
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)

    # Load data
    data_feed = bt.feeds.PandasData(dataname=data)  # type: ignore
    cerebro.adddata(data_feed)

    # Add analyzers to evaluate funded rules (Max Drawdown) and profitability
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")

    # Add the strategy with the grid of parameters
    cerebro.optstrategy(strategy_class, **param_grid)

    logger.info(f"Starting Grid Search for {strategy_class.__name__}...")
    opt_runs = cerebro.run(maxcpus=1) # using maxcpus=1 for safety with SQLite/logging if any
    
    best_pnl = -float('inf')
    best_params = None
    best_stats = {}

    for run in opt_runs:
        strat = run[0]
        params = strat.p._getkwargs()
        
        # Extract analyzer results
        dd_analyzer = strat.analyzers.drawdown.get_analysis()
        ret_analyzer = strat.analyzers.returns.get_analysis()
        trade_analyzer = strat.analyzers.trades.get_analysis()
        
        # Some runs might not place any trades
        if 'pnl' in trade_analyzer and 'net' in trade_analyzer.pnl:
            net_pnl = trade_analyzer.pnl.net.total
        else:
            net_pnl = 0.0

        max_dd = dd_analyzer.get("max", {}).get("drawdown", 0.0)

        # FUNDED ACCOUNT RULE CHECK
        # Discard any strategy that hits 10% max drawdown
        if max_dd >= params.get('max_drawdown_pct', 10.0):
            continue

        if net_pnl > best_pnl:
            best_pnl = net_pnl
            best_params = params
            best_stats = {
                "net_pnl": net_pnl,
                "max_drawdown": max_dd,
                "total_trades": trade_analyzer.get('total', {}).get('closed', 0)
            }

    if best_params:
        logger.info(f"🏆 Best {strategy_class.__name__} found!")
        logger.info(f"Params: {best_params}")
        logger.info(f"Stats: PnL=${best_stats['net_pnl']:.2f}, MaxDD={best_stats['max_drawdown']:.2f}%")
        return best_params
    else:
        logger.warning(f"No profitable parameter set survived the funded rules for {strategy_class.__name__}.")
        return None


def main():
    logger.info("Initializing Parameter Optimization...")
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

    # Load existing to not overwrite other strategy
    saved_params: dict[str, dict] = {}
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, 'r') as f:
                saved_params = json.load(f)
        except Exception:
            pass

    # --- 1. Optimize Scalping ---
    scalp_data = fetch_data("GC=F", "5m", "10d") # Using 10 days of 5m data for swift optimization
    # Small grid to keep optimization time reasonable
    scalp_grid = {
        "ema_fast": range(5, 12, 3),    # 5, 8, 11
        "ema_slow": range(20, 31, 5),   # 20, 25, 30
        "rsi_period": [14],
        "rsi_ob": [70, 75],             # 70, 75
        "rsi_os": [25, 30],             # 25, 30
        "risk_pct": [1.0]               # Static risk
    }
    
    best_scalp_params = run_optimization(ScalpStrategy, scalp_data, scalp_grid)
    if best_scalp_params:
        # Strip backtrader specific params we don't need in core logic
        clean_scalp = {k:v for k,v in best_scalp_params.items() if k not in ['risk_pct', 'daily_loss_limit_pct', 'max_drawdown_pct']}
        saved_params["scalping"] = clean_scalp # type: ignore

    # --- 2. Optimize Swing ---
    swing_data = fetch_data("BTC-USD", "1h", "60d") # 60 days of 1h data
    swing_grid = {
        "ema_trend": [50, 100],         # 50, 100
        "ema_fast": range(15, 26, 5),   # 15, 20, 25
        "atr_period": [14],
        "adx_period": [14],
        "risk_pct": [1.0]
    }

    best_swing_params = run_optimization(SwingStrategy, swing_data, swing_grid)
    if best_swing_params:
        clean_swing = {k:v for k,v in best_swing_params.items() if k not in ['risk_pct', 'daily_loss_limit_pct', 'max_drawdown_pct']}
        saved_params["swing"] = clean_swing # type: ignore

    # Save to JSON
    with open(PARAMS_PATH, 'w') as f:
        json.dump(saved_params, f, indent=4)
        
    logger.info(f"✅ Optimization complete. Saved parameters to {PARAMS_PATH}.")


if __name__ == "__main__":
    main()
