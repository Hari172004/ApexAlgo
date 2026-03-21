# 🚀 ApexAlgo Trading Bot Platform

> **Trade Above the Market** — AI-powered SaaS trading bot with prop firm support, live signals, and global deployment.

---

## 📁 Project Structure

```text
c:/project/bot/
├── core.py                     → Main bot engine (start here)
├── funded_mode.py              → Prop firm rule engine (FTMO, MFF, The5ers...)
├── risk_manager.py             → ATR-based SL/TP, position sizing, daily limits
├── demo_mode.py                → Paper trading engine
├── news_reader.py              → NewsAPI + ForexFactory + VADER/TextBlob sentiment
├── logger.py                   → Telegram, Email, Supabase logging
├── broker/
│   ├── mt5_connector.py        → MetaTrader 5 connector
│   └── ccxt_connector.py       → Binance / ByBit connector
├── strategies/
│   ├── scalping.py             → 1m/5m RSI + EMA + BB + MACD
│   └── swing.py                → 1H/4H trend + ATR + S/R
├── backend/
│   ├── main.py                 → FastAPI cloud API
│   ├── models.py               → Pydantic schemas
│   ├── database.py             → Supabase client
│   ├── auth.py                 → Firebase Auth
│   └── payments.py             → Stripe integration
├── backtest/
│   └── backtester.py           → Backtrader module
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example                → All environment variables
└── requirements.txt
```

---

## 🔰 Step 1 — Install Python & Dependencies

> **You need Python 3.11+**

1. Download [Python 3.11](https://www.python.org/downloads/)
2. Open a terminal and run:

```bash
cd c:\project\bot
pip install -r requirements.txt
```

---

## 🔰 Step 2 — Set Up Environment Variables

1. Copy `.env.example` to `.env`:

   ```bash
   copy .env.example .env
   ```

2. Open `.env` and fill in each value:
   - `MT5_ACCOUNT`, `MT5_PASSWORD`, `MT5_SERVER` — from your XM Global account
   - `SUPABASE_URL`, `SUPABASE_KEY` — from [Supabase](https://supabase.com) (free)
   - `FIREBASE_CONFIG_JSON` — from [Firebase](https://console.firebase.google.com) (free)
   - `STRIPE_SECRET_KEY` — from [Stripe](https://dashboard.stripe.com) (free)
   - `NEWS_API_KEY` — from [NewsAPI](https://newsapi.org) (free)
   - `TELEGRAM_BOT_TOKEN` — create a bot at [BotFather](https://t.me/BotFather) (free)

---

## 🔰 Step 3 — Create Supabase Tables

Run these SQL statements in your Supabase SQL editor:

```sql
-- Users table
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  name TEXT,
  email TEXT UNIQUE,
  plan TEXT,
  license_key TEXT,
  active BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trades table
CREATE TABLE trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT REFERENCES users(id),
  symbol TEXT, strategy TEXT, mode TEXT,
  direction TEXT, entry_price FLOAT, exit_price FLOAT,
  sl FLOAT, tp FLOAT, volume FLOAT,
  pnl FLOAT, win BOOLEAN,
  exit_reason TEXT, sentiment TEXT,
  opened_at TIMESTAMPTZ, closed_at TIMESTAMPTZ
);

-- Licenses table
CREATE TABLE licenses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT REFERENCES users(id),
  plan TEXT, key TEXT UNIQUE,
  active BOOLEAN DEFAULT true,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Funded progress table
CREATE TABLE funded_progress (
  user_id TEXT PRIMARY KEY,
  firm TEXT, phase TEXT,
  current_balance FLOAT, total_profit FLOAT,
  profit_progress_pct FLOAT, daily_loss_used_pct FLOAT,
  drawdown_used_pct FLOAT, days_remaining INT,
  halted BOOLEAN, phase_passed BOOLEAN, phase_failed BOOLEAN,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 🔰 Step 4 — Run in Demo Mode (Paper Trading — Safe!)

```bash
cd c:\project\bot
set BOT_MODE=DEMO
python core.py
```

You will see live signals printed to the screen and saved to a log file.

---

## 🔰 Step 5 — Run the FastAPI Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open [API endpoints](http://localhost:8000/docs) to see all API endpoints with auto-generated docs.

---

## 🔰 Step 6 — Run the Backtester

```bash
python backtest/backtester.py
```

This downloads 2 years of XAUUSD and BTC data and tests both strategies.
Results are printed showing: win rate, profit factor, max drawdown, Sharpe ratio.

---

## 🌐 Step 7 — Deploy to Oracle Cloud VPS (Free, 24/7)

1. Create a free [Oracle Cloud account](https://cloud.oracle.com) (always-free tier)
2. Create an **ARM VM** (4 CPU / 24GB RAM — completely free)
3. Connect via SSH and install Docker:

   ```bash
   sudo apt update && sudo apt install docker.io docker-compose -y
   ```

4. Upload your project and `.env` file, then:

   ```bash
   cd docker
   docker-compose up -d
   ```

5. Your API will be live at `http://YOUR_SERVER_IP:8000`

---

## 🏦 Prop Firm / Funded Account Mode

Set these in `.env`:

```ini
BOT_MODE=FUNDED
FUNDED_FIRM=FTMO
FUNDED_PHASE=PHASE_1_CHALLENGE
FUNDED_BALANCE=10000
```

The bot will:

- Never trade during high-impact news
- Auto-stop if daily loss limit is near
- Close all positions before weekend
- Show daily progress: `% toward profit target`, `% drawdown used`

---

## 💳 ApexAlgo Subscription Plans

| Plan    | Price  | Assets       | Modes                |
| :------ | :----- | :----------- | :------------------- |
| Starter | $29/mo | XAUUSD only  | Demo + Real          |
| Pro     | $59/mo | XAUUSD + BTC | Demo + Real + Funded |
| Elite   | $99/mo | All assets   | All modes + Priority |

Set up Stripe Price IDs in the Stripe Dashboard and paste them in `.env`.

---

## 🆓 All Tools Are Free

| Tool         | Use            | Cost               |
| :----------- | :------------- | :----------------- |
| Python       | Bot language   | Free               |
| MetaTrader5  | XAUUSD trading | Free               |
| CCXT         | BTC trading    | Free               |
| Oracle Cloud | 24/7 server    | Free               |
| Supabase     | Database       | Free               |
| Firebase     | Login          | Free               |
| Stripe       | Payments setup | Free               |
| Telegram     | Alerts         | Free               |
| NewsAPI      | News           | Free (100 req/day) |

---

## ❓ Beginner FAQ

**Q: Will it trade real money immediately?**
No. By default `BOT_MODE=DEMO` — it only paper trades.

**Q: How do I switch to real money?**
Change `BOT_MODE=REAL` in your `.env` and add your MT5 credentials.

**Q: Can it pass a prop firm challenge?**
The funded mode is designed to respect all FTMO/MFF rules strictly.
Always test in demo first. Past performance doesn't guarantee future results.

**Q: Is this safe?**
The risk manager limits each trade to 1-2% of your balance. The funded engine stops the bot before any hard limit is hit. But trading always carries risk.
