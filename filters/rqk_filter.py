"""
rqk_filter.py — Rational Quadratic Kernel (RQK) Trend Filter
=============================================================
Nadaraya-Watson Rational Quadratic Kernel smoother.
Gives a real-time, non-repainting trend bias — better than
a simple EMA because it weights recent bars more intelligently.

Original logic from the TradingView 'Nadaraya-Watson: RQK' indicator
(used in ZPayab DIY Strategy Builder) — translated to Python/NumPy.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("agniv.filters.rqk")


class RQKFilter:
    """
    Rational Quadratic Kernel Trend Filter.

    - yhat1: The kernel regression line (main trend smoother).
    - yhat2: A faster version (with lag offset) used to detect crossovers.
    - Output: UP | DOWN | NEUTRAL

    Parameters
    ----------
    lookback   : int   — window of bars used for estimation (default 8).
                          Higher = smoother but slower to react.
    rel_weight : float — relative weighting across timeframes (default 8.0).
                          As this → 0, longer frames dominate.
                          As this → ∞, acts like a Gaussian kernel.
    start_bar  : int   — omit the first N bars (very volatile, skews fit).
    lag        : int   — lag for crossover detection (1 = fast, 2 = smooth).
    """

    def __init__(
        self,
        lookback:   int   = 8,
        rel_weight: float = 8.0,
        start_bar:  int   = 25,
        lag:        int   = 2,
    ):
        self.h  = lookback
        self.r  = rel_weight
        self.x0 = start_bar
        self.lag = lag

    def _kernel_regression(self, src: np.ndarray, h: float) -> float:
        """Compute one kernel regression value for the latest bar."""
        size = len(src)
        current_weight    = 0.0
        cumulative_weight = 0.0

        for i in range(min(size + self.x0, size)):
            y = src[-(i + 1)]           # most recent first
            w = (1.0 + (i ** 2) / (h ** 2 * 2.0 * self.r)) ** (-self.r)
            current_weight    += y * w
            cumulative_weight += w

        if cumulative_weight == 0:
            return float(src[-1])
        return current_weight / cumulative_weight

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Analyse the DataFrame and return:
          {
            "trend"    : "UP" | "DOWN" | "NEUTRAL",
            "safe"     : bool,
            "yhat1"    : float  — main kernel regression value,
            "yhat2"    : float  — lagged kernel value,
            "bullish"  : bool,
            "bearish"  : bool,
          }
        """
        min_bars = self.x0 + self.h + 10
        if len(df) < min_bars:
            return {
                "trend": "NEUTRAL", "safe": False,
                "yhat1": 0.0, "yhat2": 0.0,
                "bullish": False, "bearish": False,
            }

        src = df["close"].values.astype(float)

        # Main kernel regression (full lookback)
        yhat1_now  = self._kernel_regression(src,         self.h)
        yhat1_prev = self._kernel_regression(src[:-1],    self.h)
        yhat1_prev2= self._kernel_regression(src[:-2],    self.h)

        # Lagged kernel (for crossover detection) — uses h - lag
        h_lag = max(3.0, self.h - self.lag)
        yhat2_now  = self._kernel_regression(src,      h_lag)

        # Rate of change
        is_bullish = yhat1_now > yhat1_prev   # rising kernel
        is_bearish = yhat1_now < yhat1_prev   # falling kernel

        was_bullish = yhat1_prev > yhat1_prev2
        was_bearish = yhat1_prev < yhat1_prev2

        # Crossover detection (smooth mode)
        bullish_cross = yhat2_now > yhat1_now
        bearish_cross = yhat2_now < yhat1_now

        if is_bullish:
            trend = "UP"
        elif is_bearish:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        safe = trend != "NEUTRAL"

        logger.debug(
            f"[RQK] trend={trend} | yhat1={yhat1_now:.4f} | "
            f"yhat2={yhat2_now:.4f} | bullish={is_bullish}"
        )

        return {
            "trend"   : trend,
            "safe"    : safe,
            "yhat1"   : float(yhat1_now),
            "yhat2"   : float(yhat2_now),
            "bullish" : is_bullish,
            "bearish" : is_bearish,
        }


# Singleton instance for Gold
gold_rqk_filter = RQKFilter(lookback=10, rel_weight=12.0, start_bar=20, lag=2)
