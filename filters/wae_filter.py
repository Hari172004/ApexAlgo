"""
wae_filter.py — Waddah Attar Explosion (WAE) Filter
=====================================================
Detects momentum explosions where MACD momentum EXCEEDS
the Bollinger Band width AND the deadzone threshold.

Only allows trades when there is REAL directional energy —
filters out fakeouts and low-conviction breakouts.

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("agniv.filters.wae")


class WAEFilter:
    """
    Waddah Attar Explosion — Momentum Explosion Gate.

    Signal logic:
      - trendUp   = MACD delta * sensitivity  (positive = bullish momentum)
      - trendDown = MACD delta * sensitivity  (negative = bearish momentum)
      - explosion = BB width (upper - lower)
      - deadzone  = RMA(TR, 100) * 3.7        (noise floor)

    BUY  allowed only when: trendUp   > explosion AND explosion > deadzone AND trendUp   > deadzone
    SELL allowed only when: trendDown > explosion AND explosion > deadzone AND trendDown > deadzone
    """

    def __init__(
        self,
        sensitivity:    int   = 150,
        fast_ema:       int   = 20,
        slow_ema:       int   = 40,
        bb_length:      int   = 20,
        bb_mult:        float = 2.0,
        deadzone_rma:   int   = 100,
        deadzone_mult:  float = 3.7,
    ):
        self.sensitivity   = sensitivity
        self.fast_ema      = fast_ema
        self.slow_ema      = slow_ema
        self.bb_length     = bb_length
        self.bb_mult       = bb_mult
        self.deadzone_rma  = deadzone_rma
        self.deadzone_mult = deadzone_mult

    def _ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _rma(self, series: pd.Series, period: int) -> pd.Series:
        """Wilder's RMA (same as Pine Script ta.rma)."""
        return series.ewm(alpha=1.0 / period, adjust=False).mean()

    def _macd_delta(self, close: pd.Series) -> pd.Series:
        """MACD line change: (fastEMA - slowEMA) delta between bars."""
        macd = self._ema(close, self.fast_ema) - self._ema(close, self.slow_ema)
        return (macd - macd.shift(1)) * self.sensitivity

    def _bb_width(self, close: pd.Series) -> pd.Series:
        """Bollinger Band width: upper - lower."""
        basis = close.rolling(self.bb_length).mean()
        std   = close.rolling(self.bb_length).std(ddof=0)
        upper = basis + self.bb_mult * std
        lower = basis - self.bb_mult * std
        return upper - lower

    def _deadzone(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Deadzone = RMA of True Range * multiplier."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return self._rma(tr, self.deadzone_rma) * self.deadzone_mult

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate WAE on the full DataFrame.

        Returns:
          {
            "safe_buy"   : bool — buy explosion confirmed,
            "safe_sell"  : bool — sell explosion confirmed,
            "trend_up"   : float,
            "trend_down" : float,
            "explosion"  : float,
            "deadzone"   : float,
          }
        """
        min_bars = max(self.slow_ema, self.bb_length, self.deadzone_rma) + 10
        if len(df) < min_bars:
            return {
                "safe_buy"   : False,
                "safe_sell"  : False,
                "trend_up"   : 0.0,
                "trend_down" : 0.0,
                "explosion"  : 0.0,
                "deadzone"   : 0.0,
            }

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        t1        = self._macd_delta(close)
        explosion = self._bb_width(close)
        dz        = self._deadzone(high, low, close)

        trend_up   = float(t1.iloc[-1]) if float(t1.iloc[-1]) >= 0 else 0.0
        trend_down = abs(float(t1.iloc[-1])) if float(t1.iloc[-1]) < 0 else 0.0
        exp_val    = float(explosion.iloc[-1])
        dz_val     = float(dz.iloc[-1])

        safe_buy  = (trend_up   > exp_val and exp_val > dz_val and trend_up   > dz_val)
        safe_sell = (trend_down > exp_val and exp_val > dz_val and trend_down > dz_val)

        logger.debug(
            f"[WAE] trendUp={trend_up:.4f} | trendDn={trend_down:.4f} | "
            f"explosion={exp_val:.4f} | deadzone={dz_val:.4f} | "
            f"safe_buy={safe_buy} | safe_sell={safe_sell}"
        )

        return {
            "safe_buy"   : safe_buy,
            "safe_sell"  : safe_sell,
            "trend_up"   : trend_up,
            "trend_down" : trend_down,
            "explosion"  : exp_val,
            "deadzone"   : dz_val,
        }


# Gold Singleton
gold_wae_filter = WAEFilter(sensitivity=150, fast_ema=20, slow_ema=40)
