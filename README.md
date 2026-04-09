# 🚀 Agni-V Trading Bot | Gold Scalper Edition

> **Trade Above the Market** — AI-powered, proprietary Gold (XAUUSD) trading bot built strictly for high-frequency scalping and swing trading via MetaTrader 5.

---

## 🔥 Current Capabilities (v2.0)

Agni-V has been entirely rebuilt into a hardened, high-performance XAUUSD trading engine:

* **100% Gold Focus:** Optimized exclusively for XAUUSD structure and liquidity. (Legacy BTC/Crypto support removed).
* **Aggressive Pyramid Scaling:** Automatically scales up to 5 compounded layers on extreme intra-bar momentum spikes.
* **Macro-Aware Killswitches:** Automatically blocks trading exactly 45 minutes before major news (CPI, NFP, FOMC) via ForexFactory RSS.
* **RL-Powered Vetting:** Sub-second Proximal Policy Optimization (PPO) signal vetting running locally to assess trade win probability based on market regime.
* **World Monitor API:** Scrapes geopolitical escalation events globally to pause trading.
* **Anti-Martingale:** Halves the risk after maximum successive hits.
* **Dynamic Spread Locks:** Cuts order transmission if broker raw spreads diverge > 20 points.
* **Anti-Martingale Compounding Risk:** Dynamically grows position sizing as account balance grows.
  * *$10 Nano/Micro Accounts fully supported via structured dynamic tiers.*
  * 1.5x Reward-to-Risk ratio locked in for scalps, ensuring a ~40% win-rate breakeven mathematically.
* **Live Telegram Control center:** Subscribed users get automatic heartbeat reports, entry/exit pings, PnL updates, and live command executions.

---

## 📁 System Architecture

```text
c:/project/bot/
├── core.py                     → Main execution engine & lifecycle pipeline
├── run_bot.py                  → The startup script & Telegram long-polling loop
├── gold_risk_manager.py        → New Advanced Risk Manager / Compounding engine
├── logger.py                   → Unified logging & Telegram broadcaster
├── broker/
│   └── mt5_connector.py        → MetaTrader 5 inter-process communication
├── strategies/
│   ├── gold_scalp.py           → Micro-Scalp Signal Generator (M1 / M5)
│   ├── diy_custom_builder.py   → Dynamic filter queue builder
│   └── ...
├── analysis/
│   ├── gold_market_structure.py→ Smart Money Concepts (ICT/Fvg) detection
│   └── gold_sessions.py        → Auto-detects NY/London Open & LBMA fix windows
├── .env                        → MT5 Credentials & API Keys
└── start_bot.bat               → Virtual-Environment Auto-Boot script
```

---

## 🔰 How to Run

1. **Prerequisites**: Python 3.9+ and an installed version of an XM Global MT5 Terminal.
2. **Setup Credentials**: Copy your MT5 Account ID, Password, and Server Name exactly into the `.env` file under `MT5_ACCOUNT`, `MT5_PASSWORD`, and `MT5_SERVER`.
3. **Boot Sequence**: Double click or run `.\start_bot.bat` inside the terminal.
4. **Select Mode**: Press `1` for Scalp mode, and let it run.

Have your Telegram open to receive live PnL and trade setups!
