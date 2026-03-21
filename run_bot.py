"""
run_bot.py — Local Headless Runner
==================================
Use this script to run the bot directly on your PC 
without needing to connect through the mobile app or localtunnel.
"""

import time
import logging
from core import ApexAlgoBot, BotConfig # type: ignore

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    
    print("\n" + "="*50)
    print("Welcome to the ApexAlgo PC Runner!")
    print(f"Target Mode: {os.getenv('BOT_MODE', 'REAL')}")
    print("Which asset would you like to trade today?")
    print("[1] Gold (XAUUSD)\n[2] Bitcoin (BTCUSD)\n[3] Both (Default)")
    choice = input("Enter 1, 2, or 3 (or press Enter for Both): ").strip()
    
    selected_assets = "BOTH"
    if choice == "1":
        selected_assets = "XAUUSD"
    elif choice == "2":
        selected_assets = "BTCUSD"

    # Load All Config from .env
    config = BotConfig(
        mode         = os.getenv("BOT_MODE", "REAL"),
        assets       = selected_assets,
        strategy     = os.getenv("BOT_STRATEGY", "AUTO"),
        risk_pct     = float(os.getenv("BOT_RISK_PCT", "1.0")),
        mt5_account  = int(os.getenv("MT5_ACCOUNT", "0")),
        mt5_password = os.getenv("MT5_PASSWORD", ""),
        mt5_server   = os.getenv("MT5_SERVER", ""),
        # Prop firm settings (only if mode=FUNDED)
        firm         = os.getenv("FUNDED_FIRM", "FTMO"),
        firm_balance = float(os.getenv("FUNDED_BALANCE", "10000")),
    )
    
    bot = ApexAlgoBot(config)
    bot.start()
    
    logging.info(f"Bot is running locally! Mode: {config.mode} | Assets: {config.assets}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping Bot...")
        bot.stop()
