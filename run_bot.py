"""
run_bot.py — Local Headless Runner
==================================
Use this script to run the bot directly on your PC 
without needing to connect through the mobile app or localtunnel.
"""

import time
import logging
import threading
from core import AgniVBot, BotConfig # type: ignore

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv # type: ignore
    load_dotenv(override=True)  # Load .env file
    
    print("Which asset would you like to trade today?")
    print("[1] Gold (XAUUSD)\n[2] Bitcoin (BTCUSD)\n[3] Both (Default)")
    choice = input("Enter 1, 2, or 3 [Default 3]: ").strip()
    
    selected_assets = "BOTH"
    if choice == "1":
        selected_assets = "XAUUSD"  # Will be mapped to GAUUSD in core.py
    elif choice == "2":
        selected_assets = "BTCUSD"
    else:
        selected_assets = "BOTH"

    print("\nSelect your Trading Strategy:")
    print("[1] ⚡ SCALPER\n[2] 🎯 SWING\n[3] 🤖 AUTO (Default)")
    strat_choice = input("Enter 1, 2, or 3 [Default 3]: ").strip()

    selected_strategy = "AUTO"
    if strat_choice == "1":
        selected_strategy = "SCALP"
    elif strat_choice == "2":
        selected_strategy = "SWING"
    else:
        selected_strategy = "AUTO"

    # Load All Config from .env
    config = BotConfig(
        mode         = os.getenv("BOT_MODE", "REAL"),
        assets       = selected_assets,
        strategy     = selected_strategy,
        risk_pct     = float(os.getenv("BOT_RISK_PCT", "1.0")),
        mt5_account  = int(os.getenv("MT5_ACCOUNT", "0")),
        mt5_password = os.getenv("MT5_PASSWORD", ""),
        mt5_server   = os.getenv("MT5_SERVER", ""),
        # Prop firm settings (only if mode=FUNDED)
        firm         = os.getenv("FUNDED_FIRM", "FTMO"),
        firm_balance = float(os.getenv("FUNDED_BALANCE", "10000")),
        use_ai_confirmation = os.getenv("USE_AI_CONFIRMATION", "true").lower() == "true",
        sniper_mode         = os.getenv("SNIPER_MODE", "false").lower() == "true",
    )
    
    bot = AgniVBot(config)

    # ── Start Telegram Command Handler ────────────────────────
    from telegram_bot import TelegramCommandHandler  # type: ignore
    tg_handler = TelegramCommandHandler(
        token         = os.getenv("TELEGRAM_BOT_TOKEN", ""),
        owner_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").split(",")[0].strip(),
        allowed_ids   = os.getenv("TELEGRAM_ALLOWED_IDS", os.getenv("TELEGRAM_CHAT_ID", "")),
    )
    tg_handler.set_bot(bot)
    tg_handler.start()

    # Override the alert chat ID so new /start subscribers receive signals too
    bot.alerts.telegram_chat_id = ",".join(tg_handler.subscribers)

    # Start the bot in a background thread so we can maintain the Telegram sync loop
    bot_thread = threading.Thread(target=bot.start, daemon=True, name="AgniVCore")
    bot_thread.start()

    # 2. Notify user bot is ON (Non-blocking)
    def _async_startup_notify():
        asset_label    = {"XAUUSD": "Gold 🥇", "BTCUSD": "Bitcoin ₿", "BOTH": "Gold 🥇 + Bitcoin ₿"}.get(selected_assets, selected_assets)
        strat_label    = {"SCALP": "⚡ SCALPER (M1/M5)", "SWING": "🎯 SWING (H1/H4)", "AUTO": "🤖 AUTO-PILOT"}.get(selected_strategy, selected_strategy)
        startup_msg    = (
            f"🟢 <b>Agni-V Bot ONLINE</b>\n"
            f"{'─' * 28}\n"
            f"📈 Asset:    <code>{asset_label}</code>\n"
            f"🎯 Strategy: <code>{strat_label}</code>\n"
            f"⚙️ Mode:     <code>{config.mode}</code>\n"
            f"🔫 Sniper:   <code>{'ON' if config.sniper_mode else 'OFF'}</code>\n\n"
            f"Live signals active! 🚀🎯"
        )
        time.sleep(2) # Give components time to breathe
        try:
            bot.alerts.send_telegram(startup_msg)
        except:
            pass

    threading.Thread(target=_async_startup_notify, daemon=True).start()

    logging.info(f"Bot is running locally! Mode: {config.mode} | Assets: {config.assets}")
    
    try:
        while True:
            # Keep subscriber list in sync with alert manager
            if tg_handler.subscribers:
                bot.alerts.telegram_chat_id = ",".join(tg_handler.subscribers)
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Stopping Bot...")
        tg_handler.stop()
        bot.stop()
