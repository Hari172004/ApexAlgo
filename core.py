"""
core.py — Agni-V Main Bot Engine
======================================
This is the central orchestrator. It:
  - Manages three trading modes: Demo, Real, Funded
  - Runs both Scalping and Swing strategies simultaneously
  - Blends technical signals with news sentiment
  - Enforces risk and prop firm rules before every trade
  - Logs and alerts on every event
  - Handles breakeven management and daily resets

Usage:
    from core import AgniVBot
    bot = AgniVBot(config)
    bot.start()
"""

import os
import time
import logging
import threading
import winsound
from datetime import datetime, date
from typing import cast
from dotenv import load_dotenv # type: ignore

from broker.mt5_connector  import MT5Connector # type: ignore
from broker.ccxt_connector import CCXTConnector # type: ignore
from strategies.scalping   import ScalpingStrategy # type: ignore
from strategies.swing      import SwingStrategy # type: ignore
from news_reader           import NewsReader # type: ignore
from risk_manager          import RiskManager # type: ignore
from funded_mode           import FundedModeEngine, Phase, PROP_FIRM_PRESETS # type: ignore
from demo_mode             import DemoMode # type: ignore
from logger                import AlertManager # type: ignore
from history_store         import HistoryStore # type: ignore
from backend.correlation   import CorrelationEngine # type: ignore
from strategies.smc        import SMCEngine # type: ignore
from ml.signal_classifier import SignalClassifier # type: ignore

# ── BTC Modules ──────────────────────────────────────
from broker.binance_connector import BinanceConnector # type: ignore
from broker.bybit_connector   import BybitConnector   # type: ignore
from strategies.btc_scalp      import BTCScalpStrategy # type: ignore
from strategies.btc_swing      import BTCSwingStrategy # type: ignore
from btc_risk_manager          import BTCRiskManager   # type: ignore
from alerts.btc_alerts         import BTCAlerts        # type: ignore

# ── Gold Modules ─────────────────────────────────────
from strategies.gold_scalp      import GoldScalpStrategy # type: ignore
from strategies.gold_swing      import GoldSwingStrategy # type: ignore
from gold_risk_manager          import GoldRiskManager    # type: ignore
from alerts.gold_alerts         import GoldAlerts         # type: ignore
from analysis.gold_sessions     import get_current_gold_session # type: ignore

load_dotenv(override=True)

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agniv_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("agniv.core")

# ──────────────────────────────────────────────────────────────
# Mode constants
# ──────────────────────────────────────────────────────────────
MODE_DEMO   = "DEMO"
MODE_REAL   = "REAL"
MODE_FUNDED = "FUNDED"

STRATEGY_SCALP     = "SCALP"
STRATEGY_SWING     = "SWING"
STRATEGY_AUTO      = "AUTO"   # bot decides based on market

ASSETS_XAUUSD = os.getenv("GOLD_SYMBOL", "XAUUSD")
ASSETS_BTC    = "BTCUSD"
ASSETS_BOTH   = "BOTH"

# MT5 Timeframe strings for each strategy
SCALP_TIMEFRAMES = ["M1", "M5"]
SWING_TIMEFRAMES = ["H1", "H4"]

# How many pips the scalp strategy uses for pip value calc (XAUUSD = $0.1/tick)
PIP_VALUE_XAUUSD = 0.1
PIP_VALUE_BTC    = 1.0


class BotConfig:
    """All configurable settings — can be updated live from the Android app."""
    mode:        str   = MODE_DEMO
    strategy:    str   = STRATEGY_AUTO
    assets:      str   = ASSETS_BOTH
    risk_pct:    float = 1.0          # % per trade (will be overridden in funded mode)
    firm:        str   = "FTMO"
    firm_phase:  str   = Phase.CHALLENGE
    firm_balance: float = 10_000.0
    mt5_account: int   = 0
    mt5_password: str  = ""
    mt5_server:   str  = ""
    exchange:     str  = "binance"    # 'binance' or 'bybit'
    ccxt_key:     str  = ""
    ccxt_secret:  str  = ""
    ccxt_testnet: bool = True
    use_ai_confirmation: bool = True
    sniper_mode: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class AgniVBot:
    """
    The main Agni-V trading bot.
    Thread-safe — mode and settings can be updated from the API at runtime.
    """

    def __init__(self, config: BotConfig):
        # ── Asset Mapping (Standardise Input) ───────────────
        # Ensure 'XAUUSD' input maps to the actual broker symbol (GAUUSD)
        if config.assets == "XAUUSD":
            config.assets = ASSETS_XAUUSD
        elif config.assets == "BTCUSD":
            config.assets = ASSETS_BTC

        self.config   = config
        self._running = False
        self._lock    = threading.Lock()
        self._last_daily_reset = date.today()

        # ── Components ───────────────────────────────────────
        self.mt5: MT5Connector = MT5Connector()
        self.ccxt      = None  # connected lazily if needed
        self.scalping  = ScalpingStrategy()
        self.swing     = SwingStrategy()
        self.news      = NewsReader(newsapi_key=os.getenv("NEWS_API_KEY", ""))
        self.risk_mgr  = RiskManager(
            max_risk_pct          = config.risk_pct,
            max_daily_loss_pct    = 5.0,
            max_consecutive_losses= 3,
        )
        self.smc_engine = SMCEngine()
        self.funded_engine: FundedModeEngine = None  # type: ignore
        self.demo_account: DemoMode          = None  # type: ignore
        self.alerts    = AlertManager(
            telegram_token   = os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", ""),
            gmail_user       = os.getenv("GMAIL_USER", ""),
            gmail_password   = os.getenv("GMAIL_APP_PASSWORD", ""),
        )
        self.history = HistoryStore()
        self._correlation = CorrelationEngine(self.history)
        self._smc         = SMCEngine() # SMCEngine doesn't take args

        # ── Asset-Specific Components ───────────────────────
        if config.assets in (ASSETS_BTC, ASSETS_BOTH):
            self.btc_binance = BinanceConnector()
            self.btc_bybit   = BybitConnector()
            self.btc_scalp   = BTCScalpStrategy()
            self.btc_swing   = BTCSwingStrategy()
            self.btc_risk    = BTCRiskManager(max_risk_pct=config.risk_pct)
            self.btc_alerts  = BTCAlerts(self.alerts)
            self.btc_ml      = SignalClassifier(symbol="BTCUSD")

        if config.assets in (ASSETS_XAUUSD, ASSETS_BOTH):
            self.gold_scalp   = GoldScalpStrategy()
            self.gold_swing   = GoldSwingStrategy()
            self.gold_risk    = GoldRiskManager(config)
            self.gold_alerts  = GoldAlerts(self.alerts)
            self.gold_ml      = SignalClassifier(symbol="XAUUSD")

        # Open positions tracking for breakeven (real mode)
        self._real_positions: dict = {}  # ticket → {sl, tp, entry, direction}

        logger.info(f"[Core] Agni-V Bot initialised | Mode={config.mode} | Assets={config.assets}")

    def _play_sound(self, action: str):
        """Play system sounds or custom WAV files for trade events."""
        try:
            if action == "entry":
                # Check for custom entry song in the root directory
                song_path = os.path.join(os.getcwd(), "entry_song.wav")
                if os.path.exists(song_path):
                    # Play asynchronously so it doesn't block the bot loop
                    winsound.PlaySound(song_path, winsound.SND_FILENAME | winsound.SND_ASYNC) # type: ignore
                else:
                    # Fallback to high pitch beep
                    winsound.Beep(1000, 500) # type: ignore
            elif action == "exit":
                # Check for custom exit song in the root directory
                exit_song_path = os.path.join(os.getcwd(), "exit_song.wav")
                if os.path.exists(exit_song_path):
                    winsound.PlaySound(exit_song_path, winsound.SND_FILENAME | winsound.SND_ASYNC) # type: ignore
                else:
                    # Lower pitch beep for exit
                    winsound.Beep(500, 500) # type: ignore
        except Exception as e:
            logger.debug(f"Could not play sound: {e}")

    def test_telegram(self):
        """Send a test message to all configured chat IDs."""
        msg = "🧪 <b>Agni-V Telegram Test</b>\nYour connection is active and ready for signals! 🚀🎯"
        return self.alerts.send_telegram(msg)

    # ── Startup ───────────────────────────────────────────────

    def start(self):
        """Starts the bot components and main loop."""
        logger.info("[Core] Starting Agni-V Bot...")
        self._setup_mode(self.config)
        
        # Start BTC Connectors if needed
        if self.config.assets in (ASSETS_BTC, ASSETS_BOTH):
            self.btc_binance.start()

        # Pre-warm historical candle cache in the background so strategies
        # have real data available before the first loop tick.
        refresh_syms = []
        if self.config.assets in (ASSETS_XAUUSD, ASSETS_BOTH):
            refresh_syms.append(ASSETS_XAUUSD)
        if self.config.assets in (ASSETS_BTC, ASSETS_BOTH):
            refresh_syms.append(ASSETS_BTC)
        
        self.history.refresh_all_background(symbols=refresh_syms)
        self._running = True

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("[Core] Stopped by user.")
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown."""
        self._running = False
        
        # 1. Disconnect broker first (High Priority)
        try:
            self.mt5.disconnect()
            self.btc_binance.stop()
            self.btc_bybit.stop()
        except Exception as e:
            logger.error(f"[Core] Error during disconnect: {e}")

        # 2. Notify user bot is OFF (Low Priority, don't block)
        def _silent_notify():
            try:
                self.alerts.send_telegram("🔴 <b>Agni-V Bot OFFLINE</b>\nBot shutting down safely. 👋", timeout=2)
            except:
                pass
        
        threading.Thread(target=_silent_notify).start()
        logger.info("[Core] Bot stopped.")

    # ── Mode Setup ────────────────────────────────────────────

    def _setup_mode(self, config: BotConfig):
        cfg = config
        if cfg.mode in (MODE_REAL, MODE_FUNDED):
            ok = self.mt5.connect(cfg.mt5_account, cfg.mt5_password, cfg.mt5_server)
            if not ok:
                raise RuntimeError("MT5 connection failed. Check credentials.")

            if cfg.assets in (ASSETS_BTC, ASSETS_BOTH) and cfg.ccxt_key:
                self.ccxt = CCXTConnector(
                    cfg.exchange, cfg.ccxt_key, cfg.ccxt_secret, testnet=cfg.ccxt_testnet
                )

        if cfg.mode == MODE_DEMO:
            self.demo_account = DemoMode(starting_balance=cfg.firm_balance)

        if cfg.mode == MODE_FUNDED:
            self.funded_engine = FundedModeEngine(
                firm=cfg.firm,
                phase=cfg.firm_phase,
                starting_balance=cfg.firm_balance,
            )
            # Override risk with funded mode max
            self.risk_mgr.max_risk_pct = 2.0

        info = self._get_balance()
        self.risk_mgr.on_new_day(info.get("balance", cfg.firm_balance))
        logger.info(f"[Core] Mode setup complete | Balance=${info.get('balance', 0):,.2f}")

    # ── Daily Reset ───────────────────────────────────────────

    def _run_daily_reset(self):
        today = date.today()
        if today != self._last_daily_reset:
            self._last_daily_reset = today
            info = self._get_balance()
            balance = info.get("balance", 0)
            self.risk_mgr.on_new_day(balance)
            if self.funded_engine:
                self.funded_engine.on_new_day(balance)
                report = self.funded_engine.daily_report()
                stats  = self.risk_mgr.stats()
                stats["balance"] = balance
                self.alerts.send_daily_report(stats, funded_report=report)
                logger.info(f"[Core] Daily report sent | {report}")
            logger.info(f"[Core] New day reset | {today}")

    # ── Main Loop ─────────────────────────────────────────────

    def _main_loop(self):
        """Tick-level main loop. Checks each symbol every cycle."""
        SYMBOLS = []
        if self.config.assets in (ASSETS_XAUUSD, ASSETS_BOTH):
            SYMBOLS.append(ASSETS_XAUUSD)
        if self.config.assets in (ASSETS_BTC, ASSETS_BOTH):
            SYMBOLS.append(ASSETS_BTC)

        logger.info(f"[Core] Main loop started | Symbols: {SYMBOLS}")

        while self._running:
            # Check for manual resume flag
            if os.path.exists("resume.flag"):
                self.risk_mgr.resume()
                try: os.remove("resume.flag") 
                except: pass
                self.alerts.risk_alert("Trading resumed manually via flag file.")
            self._run_daily_reset()

            for symbol in SYMBOLS:
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"[Core] Error processing {symbol}: {e}", exc_info=True)

            # Trailing Stop Loss checks for real positions
            if self.config.mode in (MODE_REAL, MODE_FUNDED):
                self._check_trailing_sl()

            # Weekend: close all funded positions by Friday 21:00 UTC
            if self.funded_engine and self._is_weekend_close_time():
                self._close_all_positions("Weekend close — prop firm rule")

            time.sleep(30)  # 30-second tick

    # ── Symbol Processing ─────────────────────────────────────

    def _process_symbol(self, symbol: str):
        # Initialize variables to prevent UnboundLocalError
        entry, sl, tp, volume, pip_val = 0.0, 0.0, 0.0, 0.0, 10.0
        
        # 1. Fetch news sentiment
        sentiment = self.news.get_sentiment(symbol)
        upcoming_events = sentiment.get("high_impact_events", [])
        sent_label = sentiment.get("label", "NEUTRAL")

        # 2. Get current price
        price_data = self._get_tick(symbol)
        if not price_data:
            return

        # 3. Determine strategy
        strategy_mode = self._select_strategy(symbol)
        
        # Small Account Turbo: Force SCALP for < $50
        balance_data = self._get_balance()
        current_bal  = balance_data.get("balance", 10_000)
        if current_bal < 50 and symbol == ASSETS_XAUUSD:
            strategy_mode = STRATEGY_SCALP
            logger.info(f"[Core] {symbol} | Small Account Turbo ACTIVE: Forced SCALP mode.")
            
        # 4. Generate signal
        signal_data = self._generate_signal(symbol, strategy_mode, is_nano=(current_bal < 50))
        signal      = signal_data.get("signal", "HOLD")
        atr         = signal_data.get("atr", 0)
        strength    = signal_data.get("strength", 0)

        # 5. Blend with sentiment
        final_signal = self._blend_signal(signal, sent_label)

        logger.info(
            f"[Core] {symbol} | Strategy={strategy_mode} | TechSignal={signal} "
            f"| Sentiment={sent_label} | Final={final_signal} | Strength={strength:.0%} | Reason={signal_data.get('reason', 'N/A')}"
        )

        if final_signal == "HOLD" or strength < 0.5:
            return

        # 5b. Machine Learning Filter
        ml_input = {
            "rsi":            signal_data.get("rsi", 50.0),
            "ema_distance":   signal_data.get("ema_distance", 0.0),
            "atr":            atr,
            "volume_ratio":   signal_data.get("volume_ratio", 1.0),
            "session_id":     get_current_gold_session()["session_id"] if symbol == ASSETS_XAUUSD else 2,
            "news_score":     sentiment.get("score", 0.0),
            "spread":         price_data.get("ask", 0) - price_data.get("bid", 0),
            "mtf_confluence": signal_data.get("mtf_confluence", 3)
        }
        ml_model = self.btc_ml if symbol == ASSETS_BTC else self.gold_ml

        # ML Ultra Mode: Higher threshold for low balance

        # Update risk parameters dynamically
        self.risk_mgr.set_dynamic_safety(current_bal)
        if symbol == ASSETS_XAUUSD:
            self.gold_risk.set_dynamic_safety(current_bal)
        elif symbol == ASSETS_BTC:
            self.btc_risk.set_dynamic_safety(current_bal)

        # Initialize ai_approved
        ai_approved = False
        
        if not self.config.use_ai_confirmation:
            logger.info(f"[Core] {symbol} | AI Confirmation BYPASSED per config.")
        else:
            threshold = 0.75 if current_bal < 500 else 0.70
            ml_res = ml_model.predict_signal(ml_input)
            
            if ml_res["confidence"] < threshold:
                logger.warning(f"[Core] {symbol} | Trade REJECTED by AI (Confidence: {ml_res['confidence']:.1%}, Threshold: {threshold:.0%})")
                return
            
            logger.info(f"[Core] {symbol} | Trade APPROVED by AI (Confidence: {ml_res['confidence']:.1%})")
            logger.info(f"[Core] {symbol} | SNIPER CONFIRMED: Entering High-Precision Trade!")
            ai_approved = True

        # 5c. Spread-to-ATR Safety (New)
        spread = price_data.get("ask", 0) - price_data.get("bid", 0)
        if strategy_mode == STRATEGY_SCALP and atr > 0:
            # Re-read spread for the exact symbol to ensure accuracy
            if spread > (atr * 0.25): # Limit spread to 25% of ATR
                logger.warning(f"[Core] {symbol} | Scalp REJECTED: Spread too wide ({spread:.5f} vs ATR {atr:.5f})")
                return

        # 6. Pre-trade checks
        can_trade, reason = self._check_all_guards(upcoming_events, symbol, final_signal, strategy=strategy_mode, ai_approved=ai_approved)
        if not can_trade:
            logger.warning(f"[Core] Trade blocked: {reason}")
            return

        # 7. Calculate position size & levels
        balance  = self._get_balance().get("balance", 10_000)
        entry, sl, tp, volume = 0.0, 0.0, 0.0, 0.0
        
        if symbol == ASSETS_BTC:
            risk_res = self.btc_risk.check_all_rules(balance, symbol, final_signal, atr,
                                                    is_gold_active=(ASSETS_XAUUSD in self._real_positions))
            volume = risk_res["volume"]
            # Narrow types for the static analyzer
            temp_ask = price_data.get("ask")
            temp_bid = price_data.get("bid")
            
            if (not isinstance(temp_ask, (int, float))) or (not isinstance(temp_bid, (int, float))) or (not isinstance(atr, (int, float))):
                logger.warning(f"[Core] {symbol} | Invalid price/ATR: ask={temp_ask}, bid={temp_bid}, atr={atr}")
                return
            
            # Redeclare with guaranteed types using cast
            f_ask = cast(float, temp_ask)
            f_bid = cast(float, temp_bid)
            f_atr = cast(float, atr)
            
            sl = f_ask - (1.5 * f_atr) if final_signal == "BUY" else f_bid + (1.5 * f_atr)
            tp = f_ask + (3.0 * f_atr) if final_signal == "BUY" else f_bid - (3.0 * f_atr)
            entry = f_ask if final_signal == "BUY" else f_bid
        else:
            pip_val  = PIP_VALUE_XAUUSD
            sl_pips  = (atr / 0.0001) if atr > 0 else 150
            volume   = self.risk_mgr.calculate_lot_size(balance, sl_pips, pip_val, symbol)
            entry = price_data.get("ask") if final_signal == "BUY" else price_data.get("bid")
            if entry is None:
                return
            sl, tp = self.risk_mgr.calculate_sl_tp(entry, atr or (entry * 0.001), final_signal)
        # ── Global Signal Broadcasting (Verified Signals Only) ─────
        # Broadcast the signal with the calculated SL/TP levels
        alert_manager = self.gold_alerts if symbol == ASSETS_XAUUSD else self.btc_alerts
        alert_manager.signal_alert(symbol, final_signal, strategy_mode, signal_data.get("reason", ""),
                                   entry=entry, sl=sl, tp=tp)

        # 8. Place trade
        news_boosted = False
        if sent_label != "NEUTRAL":
            if (tech_signal == "BUY" and sent_label == "BULLISH") or (tech_signal == "SELL" and sent_label == "BEARISH"):
                news_boosted = True
                logger.info(f"[Core] 🚀 News momentum alignment detected for {symbol}. Boosting TP.")

        trade = self._place_trade(symbol, final_signal, volume, entry, sl, tp,
                                  strategy=strategy_mode, sentiment=sent_label,
                                  news_boosted=news_boosted)
        if trade and "error" not in trade:
            self.alerts.trade_opened(
                {**trade, "strategy": strategy_mode, "mode": self.config.mode},
                sentiment=sent_label
            )
        else:
            err_msg = trade.get("error", "Unknown Error") if trade else "Trade rejected"
            logger.error(f"[Core] {symbol} | Trade FAILED to execute: {err_msg}")
            # Also notify Telegram if the user is Subscribed
            self.alerts.risk_alert(f"❌ Trade failed for {symbol}: {err_msg}")

    # ── Strategy Selection ────────────────────────────────────

    def _select_strategy(self, symbol: str) -> str:
        """
        Choose between Scalp and Swing based on volatility or user choice.
        """
        # 1. Overwrite with user manual selection if provided
        if self.config.strategy in (STRATEGY_SCALP, STRATEGY_SWING):
            return self.config.strategy

        # 2. Automated choice based on historical ATR
        df = self.history.get_candles(symbol, "D1", 20)
        if df.empty or len(df) < 5:
            return STRATEGY_SCALP
            
        atr_long = float(df["high"].tail(10).mean() - df["low"].tail(10).mean())
        atr_recent = float(df["high"].tail(3).mean() - df["low"].tail(3).mean())
        if atr_recent > atr_long * 1.2:
            return STRATEGY_SWING
        return STRATEGY_SCALP

    # ── Signal Generation ──────────────────────────────────────
 
    def _generate_signal(self, symbol: str, strategy_mode: str, is_nano: bool = False) -> dict:
        # ── Bitcoin Support ──────────────────────────────────
        if symbol == ASSETS_BTC:
            df = self.history.get_candles(symbol, "M5", 200) # Indicators need 200
            df_h1 = self.history.get_candles(symbol, "H1", 200) if strategy_mode == STRATEGY_SCALP else None
            
            ignore_sess = self.config.sniper_mode or (self.config.strategy == STRATEGY_SCALP)
            is_sniper   = self.config.sniper_mode
            
            if strategy_mode == STRATEGY_SCALP:
                return self.btc_scalp.generate_signal(df, df_h1=df_h1, is_nano=is_nano, ignore_sessions=ignore_sess, is_sniper=is_sniper)
            else:
                # BTC Swing currently doesn't take these, but we'll add them to the call for future-proofing
                # or just call it simply if the class isn't updated yet.
                try:
                    return self.btc_swing.generate_signal(df, is_nano=is_nano, ignore_sessions=ignore_sess, is_sniper=is_sniper)
                except TypeError:
                    return self.btc_swing.generate_signal(df)

        # ── XAUUSD Support ───────────────────────────────────
        if symbol == ASSETS_XAUUSD:
            # 1. Fetch data (MT5 or Cache)
            if self.config.mode == MODE_DEMO and not self.mt5.connected:
                df = self.history.get_candles(symbol, "M5" if strategy_mode == STRATEGY_SCALP else "H1", 300)
            else:
                df = self.mt5.get_ohlcv(symbol, "M5" if strategy_mode == STRATEGY_SCALP else "H1", 300)
            
            if df.empty:
                return {"signal": "HOLD", "atr": 0, "strength": 0}

            # 2. Strategy Analysis
            strat = self.gold_scalp if strategy_mode == STRATEGY_SCALP else self.gold_swing
            
            # Sniper Mode or Manual Scalp selection bypasses session restrictions
            ignore_sess = self.config.sniper_mode or (self.config.strategy == STRATEGY_SCALP)
            is_sniper   = self.config.sniper_mode
            
            if strategy_mode == STRATEGY_SCALP:
                res = self.gold_scalp.generate_signal(df, is_nano=is_nano, ignore_sessions=ignore_sess, is_sniper=is_sniper)
            else:
                res = self.gold_swing.generate_signal(df)
            
            # 3. Session Alerts
            session_info = get_current_gold_session()
            if session_info["is_killzone"]:
                self.gold_alerts.session_alert(session_info["active_kz"], True)
            
        # Signal broadcasting moved to _process_symbol for final verified signals
        return res

        return {"signal": "HOLD", "atr": 0, "strength": 0}

    # ── Sentiment Blending ────────────────────────────────────

    def _blend_signal(self, tech_signal: str, sentiment: str) -> str:
        """
        Combine technical signal and news sentiment.
        Conflicting signals → HOLD (conservative).
        """
        if tech_signal == "HOLD":
            return "HOLD"
        if sentiment == "NEUTRAL":
            return tech_signal
        if tech_signal == "BUY" and sentiment == "BULLISH":
            return "BUY"
        if tech_signal == "SELL" and sentiment == "BEARISH":
            return "SELL"
        # Conflicting: tech says buy but sentiment is bearish (or vice versa)
        logger.info("[Core] Signal/Sentiment conflict → HOLD")
        return "HOLD"

    # ── Pre-Trade Guards ──────────────────────────────────────

    def _check_all_guards(self, upcoming_events: list, symbol: str, direction: str, strategy: str = "SCALP", ai_approved: bool = False) -> tuple[bool, str]:
        # Get all open positions once
        open_pos = self._get_open_positions()
        
        # 1. Asset-Specific Limits (User Request: 2 Gold, 1 BTC, Total 3)
        gold_pos = [p for p in open_pos if p.get("symbol") in (ASSETS_XAUUSD, self.mt5.map_symbol(ASSETS_XAUUSD))]
        btc_pos  = [p for p in open_pos if p.get("symbol") in (ASSETS_BTC, "BTCUSD", "BITCOIN")] # generic check for BTC
        
        # Check sub-limits
        if symbol == ASSETS_XAUUSD:
            if len(gold_pos) >= 2:
                return False, f"Gold sub-limit (2) reached. Already have {len(gold_pos)} Gold positions."
        elif symbol == ASSETS_BTC:
            if len(btc_pos) >= 1:
                return False, f"BTC sub-limit (1) reached. Already have {len(btc_pos)} BTC position."

        # 2. Global Concurrency check: max 3 open positions total
        # (Relaxed slightly if AI approved? User said "place a three order a time", implying a hard cap)
        limit = 5 if ai_approved else 3
        if len(open_pos) >= limit:
            return False, f"Max global concurrent positions ({limit}) reached"
            
        # 3. Market Session Integrity Guard (XAUUSD optimization)
        # Ensure we only trade Gold during high-volume periods (London + NY overlaps)
        # NOTE: Skipped for SWING trades or if AI APPROVED.
        hour = datetime.utcnow().hour
        logger.info(f"[Core] {symbol} | Session Guard Check: hour={hour}, symbol={symbol}, strategy={strategy}, ai_approved={ai_approved}")
        if symbol == ASSETS_XAUUSD and strategy != STRATEGY_SWING and not ai_approved:
            # Avoid trading from 22:00 UTC to 07:00 UTC (Sydney/Tokyo session)
            if hour >= 22 or hour < 7:
                logger.warning(f"[Core] {symbol} | Asian Session block triggered: hour={hour}")
                return False, f"{symbol} is in low-volume Asian Session."

        # Correlation Engine Guard (DXY Check)
        correlation_check = self._correlation.check_correlation_guard(symbol, direction)
        if not correlation_check["safe"]:
            return False, correlation_check["reason"]
            
        # Base risk checks
        balance = self._get_balance().get("balance", 0)
        ok, reason = self.risk_mgr.check_can_trade(balance)
        if not ok:
            return False, reason

        # Funded mode additional checks
        if self.funded_engine:
            ok2, reason2 = self.funded_engine.check_can_trade(
                upcoming_news=upcoming_events,
                open_positions=open_pos,
                skip_news=ai_approved
            )
            if not ok2:
                self.alerts.risk_alert(f"Funded guard: {reason2}")
                return False, reason2

        return True, "OK"

    # ── Trade Execution ───────────────────────────────────────

    def _place_trade(self, symbol, direction, volume, entry, sl, tp,
                     strategy="", sentiment="NEUTRAL") -> dict:
        trade_meta = {
            "symbol":    symbol,
            "direction": direction,
            "volume":    volume,
            "price":     entry,
            "sl":        sl,
            "tp":        tp,
            "strategy":  strategy,
            "mode":      self.config.mode,
            "sentiment": sentiment,
        }

        if self.config.mode == MODE_DEMO:
            result = self.demo_account.open_position(
                symbol, direction, volume, entry, sl, tp, comment="agniv_demo"
            )
        elif self.config.mode in (MODE_REAL, MODE_FUNDED):
            # Apply News Boost to TP if flagged
            final_tp = tp
            if news_boosted:
                risk_dist = abs(entry - sl)
                final_tp = entry + (3.0 * risk_dist) if direction == "BUY" else entry - (3.0 * risk_dist)
            
            result = self.mt5.place_market_order(symbol, direction, volume, sl, final_tp)
            if "ticket" in result:
                self._real_positions[result["ticket"]] = {
                    "sl": sl, "initial_sl": sl, "tp": final_tp, "entry": entry, "direction": direction,
                    "symbol": symbol, "strategy": strategy,
                    "entry_time": time.time(), "partial_closed": False
                }
        else:
            result = {}

        if result and ("ticket" in result or "id" in result):
            self._play_sound("entry")

        return {**trade_meta, **result}

    # ── Trailing Stop Loss Management ─────────────────────────

    def _check_trailing_sl(self):
        if not self.mt5.connected:
            return
            
        # Get active positions directly from MT5 to prune closed tickets
        active_positions = self.mt5.get_open_positions()
        active_tickets = {p["ticket"] for p in active_positions if "ticket" in p}
        
        for ticket, meta in list(self._real_positions.items()):
            if ticket not in active_tickets:
                # Position was closed (TP/SL hit or manually closed), stop tracking it
                info = self.mt5.get_closed_trade_info(ticket)
                pnl = info.get("profit", 0.0)
                
                # Send Alert via Centralised Manager
                self.alerts.trade_closed({
                    "ticket":      ticket,
                    "symbol":      meta['symbol'],
                    "pnl":         pnl,
                    "exit_reason": info.get("reason", "Exit"),
                    "strategy":    meta.get("strategy", "N/A")
                })
                
                # Update Risk Manager
                self.risk_mgr.update_after_trade(pnl)
                
                del self._real_positions[ticket] # type: ignore
                self._play_sound("exit")
                continue
                
            tick = self.mt5.get_tick(meta["symbol"])
            if not tick:
                continue
            # ── [NEW] Agni-V Pulse Guard Logic ───────────────────
            entry_time = meta.get("entry_time", time.time())
            time_elapsed = time.time() - entry_time
            is_scalp = meta.get("strategy") == STRATEGY_SCALP
            
            # 1. Time-Based Exit (TBE): If scalp is open > 20 mins, close remainder
            if is_scalp and time_elapsed > 1200:
                logger.info(f"[Core] ⏱️ TBE: Scalp ticket #{ticket} open > 20 mins. Closing.")
                self.mt5.close_position(ticket)
                continue

            # 2. Risk Mgmt Trailing SL
            be_threshold = 0.5 if is_scalp else None
            should_move, new_sl = self.risk_mgr.should_update_sl(
                meta["entry"], current, meta["sl"], meta.get("initial_sl", meta["sl"]), meta["direction"],
                override_breakeven_r=be_threshold
            )
            
            # 3. Partial Profit Taking (PPT): At 1.0R, close 50% & move to BE
            initial_risk = abs(meta["entry"] - meta.get("initial_sl", meta["sl"]))
            profit_r = abs(current - meta["entry"]) / initial_risk if initial_risk > 0 else 0
            
            if is_scalp and profit_r >= 1.0 and not meta.get("partial_closed"):
                # Move to BE first
                move_ok = self.mt5.modify_sl_tp(ticket, meta["entry"], meta["tp"])
                # Then close 50%
                pos = self.mt5.positions_get(ticket=ticket)
                if pos and len(pos) > 0:
                    half_vol = round(pos[0].volume / 2.0, 2)
                    if half_vol >= 0.01:
                        close_ok = self.mt5.close_position(ticket, volume=half_vol)
                        if close_ok:
                            meta["partial_closed"] = True
                            meta["sl"] = meta["entry"]
                            logger.info(f"[Core] 💰 PPT: Closed 50% of scalp ticket #{ticket} at 1.0R.")
                continue

            # Standard trailing if not already handled by PPT
            if should_move and new_sl != meta["sl"]:
                ok = self.mt5.modify_sl_tp(ticket, new_sl, meta["tp"])
                if ok:
                    self._real_positions[ticket]["sl"] = new_sl
                    logger.info(f"[Core] 🔒 Trailing SL moved | Ticket={ticket} | NewSL={new_sl}")

    # ── Helpers ───────────────────────────────────────────────

    def _get_balance(self) -> dict:
        if self.config.mode == MODE_DEMO and self.demo_account:
            info = self.demo_account.get_account_info()
            return {"balance": info["balance"], "equity": info["equity"]}
        if self.mt5.connected:
            return self.mt5.get_account_info()
        return {"balance": self.config.firm_balance, "equity": self.config.firm_balance}

    def _get_tick(self, symbol: str) -> dict:
        if symbol == ASSETS_BTC:
            tick = self.btc_binance.get_tick()
            if tick and tick.get("price"):
                return {
                    "bid": tick["price"],
                    "ask": tick["price"] * 1.0001,
                    "last": tick["price"],
                    "time": datetime.fromtimestamp(tick["ts"]/1000)
                }

        if self.mt5.connected:
            return self.mt5.get_tick(symbol)
            
        # Demo mode without MT5: fake a tick from history
        if self.config.mode == MODE_DEMO:
            last_close = self.history.get_last_close(symbol, "M5")
            if last_close:
                return {
                    "bid": last_close,
                    "ask": last_close,
                    "last": last_close,
                    "time": datetime.now()
                }
        return {}

    def _get_open_positions(self) -> list:
        if self.config.mode == MODE_DEMO and self.demo_account:
            return self.demo_account.get_open_positions()
        if self.mt5.connected:
            return self.mt5.get_open_positions()
        return []

    def _close_all_positions(self, reason: str = "manual"):
        positions = self._get_open_positions()
        for pos in positions:
            if self.config.mode == MODE_DEMO:
                tick = self.mt5.get_tick(pos["symbol"])
                close_price = tick.get("bid", pos["open_price"])
                self.demo_account.close_position(pos["id"], close_price)
            elif self.mt5.connected:
                ticket = pos.get("ticket")
                if ticket:
                    self.mt5.close_position(ticket)
                    self._play_sound("exit")
        logger.info(f"[Core] All positions closed: {reason}")

    def _is_weekend_close_time(self) -> bool:
        """Friday 20:45 UTC — close before weekend."""
        now = datetime.utcnow()
        return now.weekday() == 4 and now.hour == 20 and now.minute >= 45

    # ── Live Config Update (from Android App) ─────────────────

    def update_config(self, **kwargs):
        """
        Call this from the API to update settings at runtime.
        Example: bot.update_config(mode='REAL', risk_pct=1.5)
        """
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
                    logger.info(f"[Core] Config updated: {k}={v}")
            # If mode changed, re-setup
            if "mode" in kwargs:
                self._setup_mode(self.config)

    def get_status(self) -> dict:
        """Return full bot status for the API and dashboard."""
        info = self._get_balance()
        risk_stats = self.risk_mgr.stats()
        funded_report = self.funded_engine.daily_report() if self.funded_engine else None
        open_pos = self._get_open_positions()
        # Surface last known prices from history cache for the dashboard
        history_info: dict = {}
        target_syms = []
        if self.config.assets in (ASSETS_XAUUSD, ASSETS_BOTH): target_syms.append(ASSETS_XAUUSD)
        if self.config.assets in (ASSETS_BTC, ASSETS_BOTH): target_syms.append(ASSETS_BTC)
        
        for sym in target_syms:
            lc = self.history.get_last_close(sym, "H1")
            if lc is not None:
                history_info[sym] = {"last_close": lc}
        return {
            "running":        self._running,
            "mode":           self.config.mode,
            "strategy":       self.config.strategy,
            "assets":         self.config.assets,
            "balance":        info.get("balance", 0),
            "equity":         info.get("equity", 0),
            "open_positions": open_pos,
            "risk_stats":     risk_stats,
            "funded_report":  funded_report,
            "history":        history_info,
            "last_update":    datetime.utcnow().isoformat(),
        }


# ──────────────────────────────────────────────────────────────
# Entry point (direct run)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    
    print("\n" + "="*50)
    print("Which asset would you like to trade?")
    print("[1] Gold (XAUUSD)\n[2] Bitcoin (BTCUSD)\n[3] Both (Default)")
    choice = input("Enter 1, 2, or 3 (or press Enter for Both): ").strip()
    
    selected_assets = ASSETS_BOTH
    if choice == "1":
        selected_assets = ASSETS_XAUUSD
    elif choice == "2":
        selected_assets = ASSETS_BTC
        
    cfg = BotConfig(
        mode         = os.getenv("BOT_MODE", MODE_DEMO),
        assets       = selected_assets,
        strategy     = os.getenv("BOT_STRATEGY", STRATEGY_AUTO),
        risk_pct     = float(os.getenv("BOT_RISK_PCT", "1.0")),
        mt5_account  = int(os.getenv("MT5_ACCOUNT", "0")),
        mt5_password = os.getenv("MT5_PASSWORD", ""),
        mt5_server   = os.getenv("MT5_SERVER", ""),
        firm         = os.getenv("FUNDED_FIRM", "FTMO"),
        firm_phase   = os.getenv("FUNDED_PHASE", Phase.CHALLENGE),
        firm_balance = float(os.getenv("FUNDED_BALANCE", "10000")),
        ccxt_key     = os.getenv("BINANCE_API_KEY", ""),
        ccxt_secret  = os.getenv("BINANCE_SECRET", ""),
        ccxt_testnet = os.getenv("CCXT_TESTNET", "true").lower() == "true",
    )
    bot = AgniVBot(cfg)
    bot.start()
