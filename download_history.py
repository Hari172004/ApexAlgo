import os
import logging
from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5 # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_history(symbol: str, timeframe: str = "D1", start_year: int = 2001):
    """Download historical data for a symbol from MT5."""
    if not mt5.initialize(): # type: ignore
        logger.error("MT5 initialize() failed")
        return

    # Map universal symbol to broker symbol
    mt5_symbol = symbol
    
    # Try to find the exact symbol name if not found
    if not mt5.symbol_select(mt5_symbol, True): # type: ignore
        candidates = [symbol, symbol+".a", symbol+"m", "GOLD" if "XAU" in symbol else "XAUUSD"]
        for cand in candidates:
            if mt5.symbol_select(cand, True): # type: ignore
                mt5_symbol = cand
                break
    
    start_date = datetime(start_year, 1, 1)
    end_date   = datetime.now()
    
    tf_map = {
        "D1": mt5.TIMEFRAME_D1, 
        "H1": mt5.TIMEFRAME_H1, 
        "M15": mt5.TIMEFRAME_M15,
        "M5": mt5.TIMEFRAME_M5
    }
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_D1)
    
    logger.info(f"Downloading {symbol} ({mt5_symbol}) {timeframe} from {start_year}...")
    rates = mt5.copy_rates_range(mt5_symbol, mt5_tf, start_date, end_date) # type: ignore
    
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get {symbol} history: {mt5.last_error()}") # type: ignore
        # mt5.shutdown() # don't shutdown here if we want to call it multiple times accurately
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Ensure standard column names
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'volume'})
    
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol.replace('/', '')}_{timeframe}_history.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Successfully saved {len(df)} rows to {filename}")

if __name__ == "__main__":
    # Download Gold (2001+)
    download_history("XAUUSD", "D1", 2001)
    
    # Download History Logic Complete

    
    mt5.shutdown() # type: ignore
