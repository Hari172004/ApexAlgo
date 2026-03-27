"""
rl/trading_env.py — Custom Gymnasium Trading Environment
=========================================================
Simulates a 1-step-per-bar scalp trading loop for PPO training.

Observation (10 features):
    rsi, ema_diff_pct, atr_norm, rvol, macd_hist_norm,
    bb_pct, ha_bull, h1_trend, session_id, close_norm

Actions:
    0 = BUY, 1 = SELL, 2 = HOLD

Reward:
    +PnL ratio when trade closes (next bar)
    -0.5 * drawdown fraction per bar in losing trade
    -0.01 per HOLD (tiny penalty to discourage over-sitting)
"""

import numpy as np
import pandas as pd  # type: ignore
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore
import ta  # type: ignore
import logging

logger = logging.getLogger("agniv.rl.env")

OBS_FEATURES = [
    "rsi", "ema_diff_pct", "atr_norm", "rvol",
    "macd_hist_norm", "bb_pct", "ha_bull", "h1_trend",
    "session_id", "close_norm",
]

# Pip value assumptions (normalised returns are used, not raw price)
TRADE_COST = 0.0002   # 0.02% spread + commission per trade
ATR_SL_MULT = 1.5     # SL = 1.5 × ATR
ATR_TP_MULT = 3.0     # TP = 3.0 × ATR  →  RRR = 2:1


class ScalpTradingEnv(gym.Env):
    """
    One episode = one pass through the entire DataFrame.
    Each step = one new OHLCV bar.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, symbol: str = "XAUUSD"):
        super().__init__()
        self.symbol = symbol
        self.df = self._build_features(df.copy())
        self.n_bars = len(self.df)

        # ── Spaces ───────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(len(OBS_FEATURES),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)   # 0=BUY 1=SELL 2=HOLD

        # State
        self._ptr: int = 0
        self._position: int = 0          # 0=flat, 1=long, -1=short
        self._entry_price: float = 0.0
        self._entry_atr: float = 0.0
        self._peak_pnl: float = 0.0

    # ─────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ptr = 50          # skip warm-up bars required by indicators
        self._position = 0
        self._entry_price = 0.0
        self._entry_atr = 0.0
        self._peak_pnl = 0.0
        return self._obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self._ptr]
        close = float(row["close"])
        atr   = float(row["atr_raw"])

        reward = 0.0
        terminated = False

        # ── Manage open position ──────────────────────────────────
        if self._position != 0:
            sl = ATR_SL_MULT * self._entry_atr
            tp = ATR_TP_MULT * self._entry_atr

            raw_pnl = (close - self._entry_price) * self._position
            pnl_pct = raw_pnl / (self._entry_price + 1e-9)

            if raw_pnl <= -sl:                       # stop-loss hit
                reward = pnl_pct - TRADE_COST
                self._position = 0
            elif raw_pnl >= tp:                      # take-profit hit
                reward = pnl_pct - TRADE_COST
                self._position = 0
            else:
                # Drawdown penalty: if we are pulling back from peak
                self._peak_pnl = max(self._peak_pnl, pnl_pct)
                dd = self._peak_pnl - pnl_pct
                reward = -0.5 * dd

        # ── Take new action (only when flat) ─────────────────────
        if self._position == 0:
            if action == 0:        # BUY
                self._position = 1
                self._entry_price = close
                self._entry_atr = atr
                self._peak_pnl = 0.0
                reward += -TRADE_COST      # entry cost
            elif action == 1:      # SELL
                self._position = -1
                self._entry_price = close
                self._entry_atr = atr
                self._peak_pnl = 0.0
                reward += -TRADE_COST
            else:                  # HOLD
                reward += -0.01    # tiny penalty to discourage over-sitting

        self._ptr += 1
        if self._ptr >= self.n_bars - 1:
            terminated = True
            # Close any open position at end of episode
            if self._position != 0:
                last_close = float(self.df.iloc[self._ptr]["close"])
                raw_pnl = (last_close - self._entry_price) * self._position
                reward += raw_pnl / (self._entry_price + 1e-9) - TRADE_COST
                self._position = 0

        obs = self._obs()
        return obs, float(reward), terminated, False, {}

    def render(self):
        pass

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self._ptr]
        obs = np.array([row[f] for f in OBS_FEATURES], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all 10 normalised observation features to the DataFrame."""
        df.columns = [c.lower() for c in df.columns]

        # Require OHLCV
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"ScalpTradingEnv: missing required column '{col}'")

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        # Raw ATR (kept in a separate col for SL/TP calc)
        df["atr_raw"] = ta.volatility.average_true_range(high, low, close, window=14)  # type: ignore

        # RSI (normalised 0–1)
        df["rsi"] = ta.momentum.rsi(close, window=14) / 100.0  # type: ignore

        # EMA diff %
        ema9  = ta.trend.ema_indicator(close, window=9)   # type: ignore
        ema21 = ta.trend.ema_indicator(close, window=21)  # type: ignore
        df["ema_diff_pct"] = (ema9 - ema21) / (close + 1e-9)

        # ATR normalised
        df["atr_norm"] = df["atr_raw"] / (close + 1e-9)

        # Relative volume
        vol_ma = df["volume"].rolling(20).mean()
        df["rvol"] = df["volume"] / (vol_ma + 1e-9)
        df["rvol"] = df["rvol"].clip(0, 5)

        # MACD histogram (normalised by ATR)
        macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)  # type: ignore
        df["macd_hist_norm"] = macd.macd_diff() / (df["atr_raw"] + 1e-9)

        # Bollinger band % position
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2.0)  # type: ignore
        bbu = bb.bollinger_hband()
        bbl = bb.bollinger_lband()
        df["bb_pct"] = (close - bbl) / (bbu - bbl + 1e-9)

        # Heiken Ashi bull flag (1 or 0)
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open  = ((df["open"] + df["close"]) / 2).shift(1)
        df["ha_bull"] = (ha_close > ha_open).astype(float)

        # H1 trend proxy: is EMA50 > EMA100 on current bar?
        ema50  = ta.trend.ema_indicator(close, window=50)   # type: ignore
        ema100 = ta.trend.ema_indicator(close, window=100)  # type: ignore
        df["h1_trend"] = (ema50 > ema100).astype(float)

        # Session ID (0=Asia 1=London 2=NY) — rough hour-based
        if "time" in df.columns:
            try:
                hours = pd.to_datetime(df["time"]).dt.hour
                df["session_id"] = pd.cut(
                    hours,
                    bins=[-1, 6, 12, 20, 24],
                    labels=[0, 1, 2, 0]
                ).astype(float) / 2.0   # normalise to 0–1
            except Exception:
                df["session_id"] = 0.5
        else:
            df["session_id"] = 0.5

        # Close normalised (z-score over rolling 100 bars)
        roll_mean = close.rolling(100).mean()
        roll_std  = close.rolling(100).std()
        df["close_norm"] = (close - roll_mean) / (roll_std + 1e-9)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
