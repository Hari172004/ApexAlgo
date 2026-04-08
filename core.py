"""
core.py — Agni-V Gold Bot Engine (XAUUSD Only)
================================================
Gold-only professional trading engine:
  - Supports $10 micro-accounts (standard accounts with leverage)
  - Anti-martingale compounding: lot size grows with balance
  - Pyramid orders: up to 5 simultaneous trades on strong signals
  - DIY Custom Strategy Builder (ZP v1) as primary signal engine
  - Both Scalp (M5) and Swing (H4) modes with configurable filters

Usage:
    from core import AgniVBot, BotConfig
    bot = AgniVBot(config)
    bot.start()
"""

import os
import time
import logging
import threading
import winsound
from datetime import datetime, date
from typing import cast, List, Deque
from collections import deque
from dotenv import load_dotenv # type: ignore

from rich.console import Console
from rich.table import Table
from rich import box
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from broker.mt5_connector  import MT5Connector # type: ignore
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
from trade_journal         import TradeJournal # type: ignore

# ── Gold Modules ─────────────────────────────────────────────────────────
from strategies.gold_scalp         import GoldScalpStrategy # type: ignore
from strategies.gold_swing         import GoldSwingStrategy # type: ignore
from gold_risk_manager             import GoldRiskManager    # type: ignore
from alerts.gold_alerts            import GoldAlerts         # type: ignore
from analysis.gold_sessions        import get_current_gold_session # type: ignore
from strategies.diy_custom_builder import DIYCustomStrategy  # type: ignore
from filters.world_monitor         import WorldMonitorAPI    # type: ignore

load_dotenv(override=True)

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # StreamHandler removed to prevent UI flickering on dashboard
        # NOTE: File logging is handled by run_bot.py via setup_file_logging().
    ],
)
logger = logging.getLogger("agniv.core")

class DashboardLogHandler(logging.Handler):
    """Custom logging handler that keeps the last N lines for the dashboard."""
    def __init__(self, capacity: int = 10):
        super().__init__()
        self.logs: Deque[str] = deque(maxlen=capacity)

    def emit(self, record):
        try:
            msg = self.format(record)
            # Remove timestamp part if it's too long for the dashboard
            if "|" in msg:
                msg = msg.split("|", 1)[1].strip()
            self.logs.append(msg)
        except Exception:
            self.handleError(record)

dashboard_logs = DashboardLogHandler(capacity=10)
dashboard_logs.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("agniv").addHandler(dashboard_logs)

# ──────────────────────────────────────────────────────────────
# Mode constants
# ──────────────────────────────────────────────────────────────
# ── Mode constants ────────────────────────────────────────────────────────────
MODE_DEMO   = "DEMO"
MODE_REAL   = "REAL"
MODE_FUNDED = "FUNDED"

STRATEGY_SCALP = "SCALP"
STRATEGY_SWING = "SWING"
STRATEGY_AUTO  = "AUTO"

ASSETS_XAUUSD = os.getenv("GOLD_SYMBOL", "XAUUSD")

# MT5 timeframe strings per strategy
SCALP_TIMEFRAMES = ["M1", "M5"]
SWING_TIMEFRAMES = ["H1", "H4"]

PIP_VALUE_XAUUSD = 0.1   # $0.1 per tick on XAUUSD


class BotConfig:
    """All configurable settings for the Gold-only bot."""
    mode:          str   = MODE_DEMO
    strategy:      str   = STRATEGY_SCALP
    assets:        str   = ASSETS_XAUUSD
    risk_pct:      float = 2.0
    leverage:      int   = 500            # broker leverage (e.g. 100, 500, 1000)
    firm:          str   = "FTMO"
    firm_phase:    str   = Phase.CHALLENGE
    firm_balance:  float = 10_000.0
    mt5_account:   int   = 0
    mt5_password:  str   = ""
    mt5_server:    str   = ""
    use_ai_confirmation: bool = True
    sniper_mode:   bool  = False
    micro_scalp:   bool  = False
    use_diy_strategy:    bool = True   # Default ON for Gold professional mode
    diy_scalp_config:    str  = "diy_scalp_config.json"
    diy_swing_config:    str  = "diy_swing_config.json"

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
        # Gold only — always force XAUUSD
        config.assets = ASSETS_XAUUSD

        self.config   = config
        self._running = False
        self._lock    = threading.Lock()
        self._last_daily_reset = date.today()

        # ── Core components ───────────────────────────────────────────
        self.mt5      = MT5Connector()
        self.scalping = ScalpingStrategy()
        self.swing    = SwingStrategy()
        self.news     = NewsReader(newsapi_key=os.getenv("NEWS_API_KEY", ""))
        self.risk_mgr = RiskManager(
            max_risk_pct           = config.risk_pct,
            max_daily_loss_pct     = 5.0,
            max_consecutive_losses = 3,
        )
        self.smc_engine    = SMCEngine()
        self.funded_engine: FundedModeEngine = None  # type: ignore
        self.demo_account:  DemoMode         = None  # type: ignore
        self.alerts = AlertManager(
            telegram_token   = os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", ""),
            gmail_user       = os.getenv("GMAIL_USER", ""),
            gmail_password   = os.getenv("GMAIL_APP_PASSWORD", ""),
        )
        self.history      = HistoryStore()
        self._correlation = CorrelationEngine(self.history)
        self._smc         = SMCEngine()
        self.journal      = TradeJournal()

        # ── Gold components ───────────────────────────────────────────
        self.gold_scalp  = GoldScalpStrategy()
        self.gold_swing  = GoldSwingStrategy()
        self.gold_risk   = GoldRiskManager(config)
        self.gold_alerts = GoldAlerts(self.alerts)
        self.gold_ml     = SignalClassifier(symbol="XAUUSD")

        # ── DIY Strategy (always on) ────────────────────────────────
        self.diy_scalp = DIYCustomStrategy(config_path=config.diy_scalp_config)
        self.diy_swing = DIYCustomStrategy(config_path=config.diy_swing_config)
        logger.info("[Core] DIY Custom Strategy Builder ACTIVE (ZP v1)")

        # ── World Monitor Intelligence (Defensive Shield) ──
        self.world_monitor = WorldMonitorAPI()

        # ── Dashboard State ──
        self.dashboard_state = {
            "symbol": ASSETS_XAUUSD,
            "balance": 0.0,
            "equity": 0.0,
            "positions": 0,
            "metrics": {
                "trend": "Sideways",
                "momentum": "Neutral",
                "volume": "Neutral",
                "rsi": 50.0,
                "adx": 20.0,
                "vwap": 0.0,
                "regime": "Low Volatility",
                "signal": "HOLD",
                "strength": 0.0
            }
        }

        # Open positions tracking
        self._real_positions: dict = {}   # ticket → {sl, tp, entry, direction, journal_id}

        logger.info(
            f"[Core] Agni-V Gold Bot initialised | Mode={config.mode} "
            f"| Strategy={config.strategy}"
        )

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
        
        # Pre-warm historical candle cache in the background so strategies
        # have real data available before the first loop tick.
        self.history.refresh_all_background(symbols=[ASSETS_XAUUSD])

        # ── Start News Sentiment Background Refresh ───────────────────────
        # Performs an immediate first fetch then refreshes every 15 minutes.
        # Without this call _articles stays empty and sentiment is always NEUTRAL.
        self.news.start_background_refresh()
        logger.info("[Core] News sentiment engine started.")

        self._running = True

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("[Core] Stopped by user.")
        finally:
            self.stop()

    def stop(self):
        """Stops the bot and disconnects all components."""
        self._running = False
        
        # ── Safe Disconnect for All Components ───────────────────
        # Use getattr(self, 'attr', None) to avoid AttributeErrors if startup failed 
        # or if specific assets (BTC/Gold) weren't initialized.
        components_to_stop = [
            'btc_binance', 'btc_bybit', 'ccxt', 'mt5', 'news'
        ]
        
        for comp_name in components_to_stop:
            comp = getattr(self, comp_name, None)
            if comp:
                try:
                    if hasattr(comp, 'stop'):
                        comp.stop()
                    elif hasattr(comp, 'disconnect'):
                        comp.disconnect()
                except Exception as e:
                    logger.debug(f"[Core] Error stopping {comp_name}: {e}")

        # 2. Notify user bot is OFF (Low Priority, don't block)
        def _silent_notify():
            try:
                # Use timeout to prevent hanging on shutdown if network is unstable
                self.alerts.send_telegram("🔴 <b>Agni-V Bot OFFLINE</b>\nBot shutting down safely. 👋", timeout=3)
            except:
                pass
        
        threading.Thread(target=_silent_notify, daemon=True).start()
        logger.info("[Core] Bot stopped.")

    # ── Mode Setup ────────────────────────────────────────────

    def _setup_mode(self, config: BotConfig):
        cfg = config
        # Connect MT5 for real/funded modes
        if cfg.mode in (MODE_REAL, MODE_FUNDED):
            ok = self.mt5.connect(cfg.mt5_account, cfg.mt5_password, cfg.mt5_server)
            if not ok:
                raise RuntimeError("MT5 connection failed. Check credentials.")

        if cfg.mode == MODE_DEMO:
            self.demo_account = DemoMode(starting_balance=cfg.firm_balance)

        if cfg.mode == MODE_FUNDED:
            self.funded_engine = FundedModeEngine(
                firm=cfg.firm,
                phase=cfg.firm_phase,
                starting_balance=cfg.firm_balance,
            )
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

    # ── Dashboard Layout ──────────────────────────────────────
    def _make_layout(self) -> Layout:
        """Create the dashboard layout with table and logs."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=12)
        )

        # Header
        mode_color = "green" if self.config.mode == MODE_REAL else ("yellow" if self.config.mode == MODE_FUNDED else "cyan")
        header_text = f"[bold white on blue] AGNI-V GOLD BOT [/]  MODE: [bold {mode_color}]{self.config.mode}[/]  SYMBOL: [bold white]{ASSETS_XAUUSD}[/]  BAL: [bold green]${self.dashboard_state['balance']:.2f}[/]"
        layout["header"].update(Panel(Text.from_markup(header_text), style="blue", box=box.ROUNDED))

        # Main Table (Ultimate Scalping Tool)
        m = self.dashboard_state["metrics"]
        table = Table(box=box.DOUBLE_EDGE, expand=True, show_header=True, header_style="bold white")
        table.add_column("ULTIMATE SCALPING TOOL", justify="center", style="bold white")
        table.add_column("Status / Value", justify="center", style="bold")

        # Row styling logic
        trend_val = m.get("trend", "Sideways")
        trend_style = "bold green" if trend_val == "UP" else ("bold red" if trend_val == "DOWN" else "dim white")
        
        momentum_val = m.get("momentum", "Neutral")
        mom_style = "bold green" if momentum_val == "Bullish" else ("bold red" if momentum_val == "Bearish" else "dim white")

        sig_val = m.get("signal", "HOLD")
        sig_color = "bold white on green" if sig_val == "BUY" else ("bold white on red" if sig_val == "SELL" else "dim white")
        
        # Advanced Signal (Logic: based on strength or final_signal)
        adv_sig = sig_val if m.get("strength", 0) > 0.8 else "No Trade"
        adv_color = "bold white on green" if adv_sig == "BUY" else ("bold white on red" if adv_sig == "SELL" else "black on bright_yellow")

        rsi_val = m.get("rsi", 50.0)
        rsi_style = "bold white on green" if rsi_val < 30 else ("bold white on red" if rsi_val > 70 else "black on bright_yellow")

        # HTF Filter (Simple H1 check)
        htf_val = "Bullish" if m.get("trend") == "UP" else ("Bearish" if m.get("trend") == "DOWN" else "Neutral")
        htf_style = "bold white on green" if htf_val == "Bullish" else ("bold white on red" if htf_val == "Bearish" else "dim white")

        # Building the table rows exactly like the image
        table.add_row("Trend (TF)", f"[{trend_style}]{trend_val}[/]")
        table.add_row("Momentum (TF)", f"[{mom_style}]{momentum_val}[/]")
        table.add_row("Volume (CMF)", "[dim white]Neutral[/]") # Placeholder for CMF
        table.add_row("Basic Signal", f"[{sig_color}]{'No Trade' if sig_val == 'HOLD' else sig_val}[/]")
        table.add_row("Advanced Signal", f"[{adv_color}] {adv_sig if adv_sig != 'No Trade' else 'No Trade'} [/]")
        table.add_row("RSI", f"[{rsi_style}] {rsi_val} [/]")
        table.add_row("HTF Filter", f"[{htf_style}] {htf_val} [/]")
        table.add_row("VWAP", f"[bold white on green] {m.get('vwap', 0.0)} [/]")
        table.add_row("ADX", f"[bold white on red] {m.get('adx', 0.0)} [/]")
        table.add_row("Mode", "[bold white on green] Custom [/]")
        table.add_row("Regime", f"[bold white on green] {m.get('regime', 'Low Volatility')} [/]")

        layout["body"].update(table)

        # Logs Panel
        logs_text = "\n".join(dashboard_logs.logs)
        layout["footer"].update(Panel(Text(logs_text), title="[bold white]Recent Intelligence Logs[/]", border_style="bright_black"))

        return layout

    def _main_loop(self):
        """Tick-level main loop. Checks each symbol every cycle."""
        SYMBOLS = [ASSETS_XAUUSD]

        logger.info(f"[Core] Main loop started | Symbols: {SYMBOLS}")

        while self._running:
            # Check for manual resume flag
            if os.path.exists("resume.flag"):
                self.risk_mgr.resume()
                try: os.remove("resume.flag") 
                except: pass
                self.alerts.risk_alert("Trading resumed manually via flag file.")
            self._run_daily_reset()

            # Setup Dashboard
            with Live(self._make_layout(), refresh_per_second=2, screen=True) as live:
                while self._running:
                    # Run Daily Reset Check
                    self._run_daily_reset()

                    for symbol in SYMBOLS:
                        try:
                            self._process_symbol(symbol)
                        except Exception as e:
                            logger.error(f"[Core] Error processing {symbol}: {e}")

                    # Trailing Stop Loss checks for real positions
                    if self.config.mode in (MODE_REAL, MODE_FUNDED):
                        self._check_trailing_sl()

                    # Weekend: close all funded positions by Friday 21:00 UTC
                    if self.funded_engine and self._is_weekend_close_time():
                        self._close_all_positions("Weekend close — prop firm rule")

                    # Update Dashboard
                    live.update(self._make_layout())
                    time.sleep(1) # Refresh every second

    # ── Symbol Processing ─────────────────────────────────────

    def _process_symbol(self, symbol: str):
        """Full signal pipeline for one XAUUSD tick."""
        # 1. News sentiment
        sentiment       = self.news.get_sentiment(symbol)
        upcoming_events = sentiment.get("high_impact_events", [])
        sent_label      = sentiment.get("label", "NEUTRAL")

        # 2. Price tick
        price_data = self._get_tick(symbol)
        if not price_data:
            return

        # 2.5 World Monitor Defensive Shield (Block trades during Global Shocks)
        crisis_level = self.world_monitor.get_crisis_level()
        if crisis_level == "CRITICAL":
            logger.warning(f"[Core] 🛡️ WORLD MONITOR SHIELD: Blocking ALL {symbol} trades due to Global Macro Crisis!")
            return

        # 3. Balance & account tier
        balance_data = self._get_balance()
        current_bal  = float(balance_data.get("balance", 100))
        self.risk_mgr.set_dynamic_safety(current_bal)
        self.gold_risk.set_dynamic_safety(current_bal)

        # 4. Strategy selection
        strategy_mode = self._select_strategy(symbol)
        is_nano       = current_bal < 50
        if is_nano:
            strategy_mode = STRATEGY_SCALP
            logger.info(f"[Core] Nano Account (${current_bal:.2f}) — forced SCALP mode")

        # 5. Generate signal
        signal_data  = self._generate_signal(symbol, strategy_mode,
                                              is_nano=is_nano,
                                              micro_scalp=self.config.micro_scalp)
        signal       = signal_data.get("signal", "HOLD")
        atr          = float(signal_data.get("atr", 0) or 0)
        strength     = float(signal_data.get("strength", 0))

        # 6. Sentiment blend
        final_signal = self._blend_signal(signal, sent_label)

        logger.info(
            f"[Core] XAUUSD | Strategy={strategy_mode} | Signal={signal} "
            f"| Sentiment={sent_label} | Final={final_signal} "
            f"| Strength={strength:.0%} | {signal_data.get('reason', '')}"
        )

        # Update Dashboard State
        self.dashboard_state["balance"] = current_bal
        self.dashboard_state["positions"] = len(self._get_open_positions())
        if self.config.use_diy_strategy:
            status = self.diy_scalp.get_status() if strategy_mode == STRATEGY_SCALP else self.diy_swing.get_status()
            m = status.get("metrics", {})
            self.dashboard_state["metrics"].update(m)
            self.dashboard_state["metrics"]["signal"] = signal
            self.dashboard_state["metrics"]["strength"] = strength

        if final_signal == "HOLD" or strength < 0.5:
            return

        # 7. ML filter
        ml_input = {
            "rsi":            signal_data.get("rsi", 50.0),
            "ema_distance":   signal_data.get("ema_distance", 0.0),
            "atr":            atr,
            "volume_ratio":   signal_data.get("volume_ratio", 1.0),
            "session_id":     get_current_gold_session()["session_id"],
            "news_score":     sentiment.get("score", 0.0),
            "spread":         price_data.get("ask", 0) - price_data.get("bid", 0),
            "mtf_confluence": signal_data.get("mtf_confluence", 3),
        }
        ai_approved = False
        if not self.config.use_ai_confirmation:
            ai_approved = True
            logger.info(f"[Core] XAUUSD | AI confirmation BYPASSED")
        else:
            threshold = 0.75 if current_bal < 500 else 0.70
            ml_res = self.gold_ml.predict_signal(ml_input)
            if strength >= 0.95:
                logger.info(f"[Core] XAUUSD | 🔥 PERFECT SETUP ({strength:.0%}) — bypassing AI")
                ai_approved = True
            elif ml_res["confidence"] < threshold:
                logger.warning(
                    f"[Core] XAUUSD | Trade REJECTED by AI "
                    f"(conf={ml_res['confidence']:.1%} < {threshold:.0%})"
                )
                return
            else:
                logger.info(f"[Core] XAUUSD | ✅ AI APPROVED (conf={ml_res['confidence']:.1%})")
                ai_approved = True

        # 8. Spread guard
        spread = price_data.get("ask", 0) - price_data.get("bid", 0)
        if strategy_mode == STRATEGY_SCALP and atr > 0 and spread > atr * 0.25:
            logger.warning(f"[Core] XAUUSD | Scalp REJECTED: spread too wide ({spread:.5f} vs {atr:.5f})")
            return

        # 9. Pre-trade guards
        can_trade, reason = self._check_all_guards(
            upcoming_events, symbol, final_signal, strategy=strategy_mode, ai_approved=ai_approved
        )
        if not can_trade:
            logger.warning(f"[Core] Trade blocked: {reason}")
            return

        # 10. Compute entry, SL, TP
        entry = price_data.get("ask") if final_signal == "BUY" else price_data.get("bid")
        if entry is None:
            return
        entry = float(entry)
        sl_pts = max(atr * 1.5, 1.0)   # minimum 1.0 XAU point SL
        sl = entry - sl_pts if final_signal == "BUY" else entry + sl_pts
        tp = entry + (sl_pts * 2.5) if final_signal == "BUY" else entry - (sl_pts * 2.5)  # 2.5R TP

        # 11. Gold risk rules + pyramid lots
        open_pos     = self._get_open_positions()
        open_gold    = [p for p in open_pos if ASSETS_XAUUSD in str(p.get("symbol", ""))
                        or self.mt5.map_symbol(ASSETS_XAUUSD) in str(p.get("symbol", ""))]
        risk_res = self.gold_risk.check_all_rules(
            balance         = current_bal,
            signal          = final_signal,
            atr             = atr,
            open_gold_pos   = len(open_gold),
            spread_points   = spread,
            news_pause      = bool(upcoming_events),
            signal_strength = strength,
        )
        if not risk_res["can_trade"]:
            logger.warning(f"[Core] Gold risk blocked: {risk_res['reason']}")
            return

        lots      = risk_res["lots"]          # e.g. [0.01, 0.01, 0.01]
        n_orders  = len(lots)
        sl_val    = risk_res["sl_value"]

        # Recompute SL/TP with gold risk manager's SL
        sl = entry - sl_val if final_signal == "BUY" else entry + sl_val
        tp = entry + (sl_val * 2.5) if final_signal == "BUY" else entry - (sl_val * 2.5)

        # 12. Signal alert (broadcast to Telegram/email)
        self.gold_alerts.signal_alert(
            symbol, final_signal, strategy_mode, signal_data.get("reason", ""),
            entry=entry, sl=sl, tp=tp
        )

        # 13. News boost — widen TP on aligned sentiment
        news_boosted = (sent_label == "BULLISH" and final_signal == "BUY") or \
                       (sent_label == "BEARISH" and final_signal == "SELL")
        if news_boosted:
            logger.info(f"[Core] 🚀 News momentum aligned — boosting TP by 20%")
            tp = entry + (sl_val * 3.0) if final_signal == "BUY" else entry - (sl_val * 3.0)

        # 14. Execute pyramid orders
        self._execute_pyramid_orders(
            symbol=symbol, direction=final_signal, lots=lots,
            entry=entry, sl=sl, tp=tp,
            strategy=strategy_mode, sentiment=sent_label,
            news_boosted=news_boosted, strength=strength,
        )

    def _execute_pyramid_orders(
        self, symbol: str, direction: str, lots: List[float],
        entry: float, sl: float, tp: float,
        strategy: str, sentiment: str,
        news_boosted: bool, strength: float,
    ):
        """
        Place 1–5 pyramid orders based on signal strength.
        Order 1 = anchor (largest), subsequent orders = scaled down.
        """
        n = len(lots)
        logger.info(
            f"[Core] 📊 PYRAMID EXECUTE: {n} orders | {direction} XAUUSD "
            f"| lots={lots} | strength={strength:.0%}"
        )
        self._play_sound("entry")

        placed = 0
        for i, lot in enumerate(lots):
            trade = self._place_trade(
                symbol, direction, lot, entry, sl, tp,
                strategy=strategy, sentiment=sentiment,
                news_boosted=news_boosted
            )
            if trade and "error" not in trade:
                placed += 1
                logger.info(
                    f"[Core] Order {i+1}/{n} PLACED ✅ | lot={lot} | ticket={trade.get('ticket', '?')}"
                )
                if i == 0:  # Only broadcast the alert for the first (anchor) order
                    try:
                        self.alerts.trade_opened(
                            {**trade, "strategy": strategy, "mode": self.config.mode},
                            sentiment=sentiment
                        )
                    except Exception:
                        pass
            else:
                err = trade.get("error", "unknown") if trade else "rejected"
                logger.warning(f"[Core] Order {i+1}/{n} FAILED: {err}")

        if placed > 0:
            logger.info(f"[Core] 🎯 {placed}/{n} orders placed | XAUUSD {direction}")
        else:
            logger.error(f"[Core] ❌ ALL {n} orders failed for XAUUSD {direction}")
            self.alerts.risk_alert(f"❌ All pyramid orders failed for XAUUSD {direction}")


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
 
    def _generate_signal(self, symbol: str, strategy_mode: str,
                         is_nano: bool = False, micro_scalp: bool = False) -> dict:

        if symbol == ASSETS_XAUUSD:
            tf = "M1" if micro_scalp else ("M5" if strategy_mode == STRATEGY_SCALP else "H1")
            if self.config.mode == MODE_DEMO and not self.mt5.connected:
                df = self.history.get_candles(symbol, tf, 300)
            else:
                df = self.mt5.get_ohlcv(symbol, tf, 300)

            if df.empty:
                return {"signal": "HOLD", "atr": 0, "strength": 0}

            # Gold Micro-Scalp: prefer London Open window (07:00-09:00 UTC)
            if micro_scalp:
                now_hour = datetime.utcnow().hour
                is_london_open = 7 <= now_hour <= 9
                is_ny_open     = 13 <= now_hour <= 15
                if not (is_london_open or is_ny_open):
                    return {
                        "signal": "HOLD", "atr": 0, "strength": 0,
                        "reason": f"Micro-Scalp Gold: waiting for London (07-09) or NY Open (13-15) UTC | now={now_hour}h"
                    }
                logger.info(f"[Core] Gold Micro-Scalp Mode: M1 candles | {'London' if is_london_open else 'NY'} Open window")

            ignore_sess = self.config.sniper_mode or (self.config.strategy == STRATEGY_SCALP) or micro_scalp
            is_sniper   = self.config.sniper_mode

            # ── DIY Strategy override (if enabled) ──────────────
            if self.config.use_diy_strategy and self.diy_scalp is not None:
                diy_strat  = self.diy_scalp if strategy_mode == STRATEGY_SCALP else self.diy_swing
                diy_signal = diy_strat.generate_signal(df)
                status     = diy_strat.get_status()
                logger.info(
                    f"[Core] {symbol} | DIY Strategy signal={diy_signal} "
                    f"| pending={status['pending_direction']} bars={status['pending_bars']}/{status['signal_expiry']}"
                )
                real_atr = float((df["high"].tail(14) - df["low"].tail(14)).mean())
                res = {"signal": diy_signal, "atr": real_atr, "strength": 0.95 if diy_signal != "HOLD" else 0,
                       "reason": f"DIY[{status['leading_indicator']}]+{status['active_filters']}"}
            else:
                # Default gold strategies
                if strategy_mode == STRATEGY_SCALP:
                    res = self.gold_scalp.generate_signal(df, is_nano=is_nano,
                                                          ignore_sessions=ignore_sess, is_sniper=is_sniper)
                else:
                    res = self.gold_swing.generate_signal(df)

            session_info = get_current_gold_session()
            if session_info["is_killzone"]:
                self.gold_alerts.session_alert(session_info["active_kz"], True)

        # Signal broadcasting moved to _process_symbol for final verified signals
        return res

    # ── Sentiment Blending ────────────────────────────────────

    def _blend_signal(self, tech_signal: str, sentiment: str) -> str:
        """
        Combine technical signal and news sentiment.
        USER EXPLICIT OVERRIDE: Blindly trust technical limits.
        """
        if tech_signal != "HOLD":
            logger.info(f"[Core] 🔥 SENTIMENT OVERRIDE: Forcing '{tech_signal}' setup over '{sentiment}' news")
        return tech_signal

    # ── Pre-Trade Guards ──────────────────────────────────────

    def _check_all_guards(self, upcoming_events: list, symbol: str, direction: str,
                          strategy: str = "SCALP", ai_approved: bool = False) -> tuple:
        open_pos  = self._get_open_positions()
        gold_pos  = [p for p in open_pos
                     if p.get("symbol") in (ASSETS_XAUUSD, self.mt5.map_symbol(ASSETS_XAUUSD))]
        gold_limit = 10 if strategy == STRATEGY_SCALP else 3

        if len(gold_pos) >= gold_limit:
            return False, f"Gold limit ({gold_limit}) reached — {len(gold_pos)} positions open."

        # Global concurrency cap
        global_limit = 10 if strategy == STRATEGY_SCALP else (5 if ai_approved else 3)
        if len(open_pos) >= global_limit:
            return False, f"Max global positions ({global_limit}) reached"

        # Asian session guard for scalp (22–07 UTC low volume)
        hour = datetime.utcnow().hour
        if strategy != STRATEGY_SWING and not ai_approved:
            if hour >= 22 or hour < 7:
                logger.warning(f"[Core] Asian session block | hour={hour}")
                return False, "Low-volume Asian session — waiting for London/NY open."

        # Correlation engine
        correlation_check = self._correlation.check_correlation_guard(symbol, direction)
        if not correlation_check["safe"]:
            return False, correlation_check["reason"]

        # Base risk manager
        balance = self._get_balance().get("balance", 0)
        ok, reason = self.risk_mgr.check_can_trade(balance)
        if not ok:
            return False, reason

        # Funded mode
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
                     strategy="", sentiment="NEUTRAL", news_boosted=False) -> dict:
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
            # ── Micro-Scalp: use 0.5R TP for fast exits ─────────────────────────
            if self.config.micro_scalp:
                risk_dist = abs(entry - sl)
                tp = (entry + 0.5 * risk_dist) if direction == "BUY" else (entry - 0.5 * risk_dist)
                logger.info(f"[Core] ⚡ Micro-Scalp TP set to 0.5R | TP={tp:.5f}")

            # Apply News Boost to TP if flagged (only if not already in micro mode)
            final_tp = tp
            if news_boosted and not self.config.micro_scalp:
                risk_dist = abs(entry - sl)
                final_tp = entry + (3.0 * risk_dist) if direction == "BUY" else entry - (3.0 * risk_dist)
            
            result = self.mt5.place_market_order(symbol, direction, volume, sl, final_tp)
            if "ticket" in result:
                # Log to trade journal
                j_id = self.journal.log_open(
                    symbol=symbol, direction=direction,
                    reason=trade_meta.get("reason", ""),
                    strength=trade_meta.get("strength", 0.0),
                    entry=entry, sl=sl, tp=final_tp,
                    strategy=strategy,
                )
                self._real_positions[result["ticket"]] = {
                    "sl": sl, "initial_sl": sl, "tp": final_tp, "entry": entry,
                    "direction": direction, "symbol": symbol, "strategy": strategy,
                    "entry_time": time.time(),
                    "partial_closed_1r": False,   # ← 3-Level TP Ladder flags
                    "partial_closed_2r": False,
                    "journal_id": j_id,
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
                # Position was closed (TP/SL hit or manually closed)
                info = self.mt5.get_closed_trade_info(ticket)
                pnl  = info.get("profit", 0.0)

                # Trade Journal — record close
                j_id = meta.get("journal_id", "")
                if j_id:
                    self.journal.log_close(j_id, pnl, info.get("reason", "Exit"))

                # Send Alert via Centralised Manager
                self.alerts.trade_closed({
                    "ticket":      ticket,
                    "symbol":      meta["symbol"],
                    "pnl":         pnl,
                    "exit_reason": info.get("reason", "Exit"),
                    "strategy":    meta.get("strategy", "N/A")
                })

                # Update Risk Manager + handle cooldown Telegram alert
                self.risk_mgr.update_after_trade(pnl)
                stats = self.risk_mgr.stats()
                if stats.get("cooldown_remaining_min", 0) > 0:
                    self.alerts.risk_alert(
                        f"⚡ Cooldown activated after {self.risk_mgr.cooldown_losses} losses! "
                        f"Trading paused for {stats['cooldown_remaining_min']} min. "
                        f"Will auto-resume."
                    )

                del self._real_positions[ticket]  # type: ignore
                self._play_sound("exit")
                continue

            tick = self.mt5.get_tick(meta["symbol"])
            if not tick:
                continue

            current = tick.get("bid") if meta["direction"] == "BUY" else tick.get("ask")
            if current is None:
                continue

            entry_time   = meta.get("entry_time", time.time())
            time_elapsed = time.time() - entry_time
            is_scalp     = meta.get("strategy") == STRATEGY_SCALP

            # ── 1. Time-Based Exit (TBE): scalp open > 20 mins → close ─────
            if is_scalp and time_elapsed > 1200:
                logger.info(f"[Core] ⏱️ TBE: Scalp #{ticket} open >20 min. Closing.")
                self.mt5.close_position(ticket)
                continue

            # ── 2. 3-Level TP Ladder (Zignaly / 3Commas style) ───────────
            initial_risk = abs(meta["entry"] - meta.get("initial_sl", meta["sl"]))
            profit_r     = abs(current - meta["entry"]) / initial_risk if initial_risk > 0 else 0

            pos = self.mt5.positions_get(ticket=ticket) if is_scalp else None
            cur_vol = pos[0].volume if pos and len(pos) > 0 else 0

            # Level 1: At 1R → close 30%, move SL to Break-Even
            if is_scalp and profit_r >= 1.0 and not meta.get("partial_closed_1r") and cur_vol > 0:
                close_vol = round(cur_vol * 0.30, 2)
                if close_vol >= 0.01:
                    ok = self.mt5.close_position(ticket, volume=close_vol)
                    if ok:
                        self.mt5.modify_sl_tp(ticket, meta["entry"], meta["tp"])  # → Break-Even
                        meta["partial_closed_1r"] = True
                        meta["sl"] = meta["entry"]
                        logger.info(f"[Core] 🎯 TP-L1 (1R): Closed 30% of #{ticket} | SL → B/E")
                continue

            # Level 2: At 2R → close another 30%, trail SL to 1R profit
            if is_scalp and profit_r >= 2.0 and meta.get("partial_closed_1r") \
                    and not meta.get("partial_closed_2r") and cur_vol > 0:
                close_vol = round(cur_vol * 0.30, 2)
                one_r_profit_sl = (
                    meta["entry"] + initial_risk if meta["direction"] == "BUY"
                    else meta["entry"] - initial_risk
                )
                if close_vol >= 0.01:
                    ok = self.mt5.close_position(ticket, volume=close_vol)
                    if ok:
                        self.mt5.modify_sl_tp(ticket, one_r_profit_sl, meta["tp"])  # → Lock 1R
                        meta["partial_closed_2r"] = True
                        meta["sl"] = one_r_profit_sl
                        logger.info(f"[Core] 🎯 TP-L2 (2R): Closed 30% of #{ticket} | SL → +1R locked")
                continue

            # Level 3: At 3R → let remaining 40% run with tight 0.5R trail
            # (handled by trailing SL below with tighter threshold)

            # ── 3. Trailing SL for the remaining position ────────────────
            # Use tighter trail (0.5R) at L3 to lock in max profit
            be_threshold  = 0.5 if (is_scalp and meta.get("partial_closed_2r")) else \
                            (0.5 if is_scalp else None)
            should_move, new_sl = self.risk_mgr.should_update_sl(
                meta["entry"], current, meta["sl"],
                meta.get("initial_sl", meta["sl"]), meta["direction"],
                override_breakeven_r=be_threshold
            )
            if should_move and new_sl != meta["sl"]:
                ok = self.mt5.modify_sl_tp(ticket, new_sl, meta["tp"])
                if ok:
                    self._real_positions[ticket]["sl"] = new_sl
                    logger.info(f"[Core] 🔒 Trail SL moved | #{ticket} | NewSL={new_sl:.5f} | Profit={profit_r:.1f}R")

    # ── Helpers ───────────────────────────────────────────────

    def _get_balance(self) -> dict:
        if self.config.mode == MODE_DEMO and self.demo_account:
            info = self.demo_account.get_account_info()
            return {"balance": info["balance"], "equity": info["equity"]}
        if self.mt5.connected:
            return self.mt5.get_account_info()
        return {"balance": self.config.firm_balance, "equity": self.config.firm_balance}

    def _get_tick(self, symbol: str) -> dict:


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
        target_syms = [ASSETS_XAUUSD]
        
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
    
    cfg = BotConfig(
        mode         = os.getenv("BOT_MODE", MODE_DEMO),
        assets       = ASSETS_XAUUSD,
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
