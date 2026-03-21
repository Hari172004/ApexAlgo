"""
gold_indicators.py -- Full professional indicator suite for Gold (XAUUSD)
20+ indicators: EMA, SMA, RSI, MACD, BB, ATR, VWAP, Fibonacci, Pivot Points,
Ichimoku, Supertrend, Parabolic SAR, Williams %R, CCI, Donchian, Heiken Ashi
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("apexalgo.gold_indicators")


# ── Core Indicators ────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Main calculate function ────────────────────────────────────────────────

def calculate_gold_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full institutional Gold indicator suite.
    Input df must have columns: open, high, low, close, volume
    """
    if df.empty or len(df) < 200:
        return df

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # 1. EMAs — trend direction and dynamic S/R
    for p in [9, 21, 50, 100, 200]:
        df[f"ema_{p}"] = _ema(c, p)

    # 2. SMAs — golden cross / death cross
    df["sma_50"]  = _sma(c, 50)
    df["sma_200"] = _sma(c, 200)
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # 3. RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9)
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # 5. Bollinger Bands (20, 2)
    mid = _sma(c, 20)
    std = c.rolling(20).std()
    df["bb_mid"]   = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0, np.nan)
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.8)

    # 6. ATR 14 — stop loss sizing
    df["atr"] = _atr(df, 14)
    df["atr_avg"] = _sma(df["atr"], 20)
    df["atr_spike"] = df["atr"] > df["atr_avg"] * 3.0  # ATR spike flag

    # 7. Stochastic RSI (14, 14, 3, 3)
    rsi14 = df["rsi"]
    min_rsi = rsi14.rolling(14).min()
    max_rsi = rsi14.rolling(14).max()
    stoch_rsi = (rsi14 - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    df["stoch_k"] = stoch_rsi.rolling(3).mean() * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # 8. VWAP (rolling intraday approximation)
    tpv = (h + l + c) / 3 * v
    df["vwap"] = tpv.cumsum() / v.cumsum()

    # 9. Williams %R (14)
    highest_h = h.rolling(14).max()
    lowest_l  = l.rolling(14).min()
    df["williams_r"] = -100 * (highest_h - c) / (highest_h - lowest_l).replace(0, np.nan)

    # 10. CCI — Commodity Channel Index (20)
    tp = (h + l + c) / 3
    df["cci"] = (tp - _sma(tp, 20)) / (0.015 * tp.rolling(20).std())

    # 11. Donchian Channel (20)
    df["donchian_high"] = h.rolling(20).max()
    df["donchian_low"]  = l.rolling(20).min()
    df["donchian_mid"]  = (df["donchian_high"] + df["donchian_low"]) / 2

    # 12. Parabolic SAR
    df["psar"] = _parabolic_sar(df)

    # 13. Supertrend (10 periods, multiplier 3)
    df["supertrend"], df["supertrend_dir"] = _supertrend(df, period=10, multiplier=3)

    # 14. Ichimoku Cloud
    df = _ichimoku(df)

    # 15. Heiken Ashi
    ha = _heiken_ashi(df.copy())
    df["ha_open"]  = ha["ha_open"]
    df["ha_close"] = ha["ha_close"]
    df["ha_high"]  = ha["ha_high"]
    df["ha_low"]   = ha["ha_low"]
    df["ha_bull"]  = (df["ha_close"] > df["ha_open"]).astype(int)

    # 16. Fibonacci Levels (last 30-bar swing)
    df = _add_fibonacci(df, lookback=30)

    # 17. Pivot Points (last day's OHLC)
    df = _add_pivots(df)

    return df


# ── Parabolic SAR ─────────────────────────────────────────────────────────

def _parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)
    psar  = close.copy()
    bull  = True
    af    = af_start
    ep    = low[0]
    hp    = high[0]
    lp    = low[0]

    for i in range(2, n):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], low[i - 1], low[i - 2])
            if low[i] < psar[i]:
                bull  = False
                psar[i] = hp
                lp    = low[i]
                af    = af_start
                ep    = low[i]
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_start, af_max)
                ep = hp
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], high[i - 1], high[i - 2])
            if high[i] > psar[i]:
                bull  = True
                psar[i] = lp
                hp    = high[i]
                af    = af_start
                ep    = high[i]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_start, af_max)
                ep = lp

    return pd.Series(psar, index=df.index)


# ── Supertrend ────────────────────────────────────────────────────────────

def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr   = _atr(df, period)
    hl2   = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 = bull, -1 = bear

    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        fu_prev = final_upper.iloc[i - 1]
        fl_prev = final_lower.iloc[i - 1]
        c_prev  = df["close"].iloc[i - 1]
        c_curr  = df["close"].iloc[i]

        final_upper.iloc[i] = upper.iloc[i] if upper.iloc[i] < fu_prev or c_prev > fu_prev else fu_prev
        final_lower.iloc[i] = lower.iloc[i] if lower.iloc[i] > fl_prev or c_prev < fl_prev else fl_prev

        if direction.iloc[i - 1] == -1 and c_curr > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[i - 1] == 1 and c_curr < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        trend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return trend, direction


# ── Ichimoku Cloud ────────────────────────────────────────────────────────

def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"]
    l = df["low"]

    # Tenkan-sen (Conversion Line) — 9 periods
    df["tenkan"]  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    # Kijun-sen (Base Line) — 26 periods
    df["kijun"]   = (h.rolling(26).max() + l.rolling(26).min()) / 2
    # Senkou Span A — avg of Tenkan/Kijun shifted 26 forward
    df["span_a"]  = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    # Senkou Span B — 52-period range shifted 26 forward
    df["span_b"]  = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    # Chikou Span — close shifted 26 backward
    df["chikou"]  = df["close"].shift(-26)

    df["ichimoku_bull"] = (
        (df["close"] > df["span_a"]) &
        (df["close"] > df["span_b"]) &
        (df["span_a"] > df["span_b"])
    ).astype(int)

    return df


# ── Heiken Ashi ───────────────────────────────────────────────────────────

def _heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["ha_open"]  = (df["open"].shift(1) + df["close"].shift(1)) / 2
    df["ha_open"]  = df["ha_open"].fillna((df["open"] + df["close"]) / 2)
    df["ha_high"]  = df[["high", "ha_open", "ha_close"]].max(axis=1)
    df["ha_low"]   = df[["low",  "ha_open", "ha_close"]].min(axis=1)
    return df


# ── Fibonacci Levels ──────────────────────────────────────────────────────

FIBO_RETRACE = [0.236, 0.382, 0.5, 0.618, 0.786]
FIBO_EXTEND  = [1.272, 1.618]

def _add_fibonacci(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()
    diff = swing_high - swing_low

    for level in FIBO_RETRACE:
        df[f"fib_{int(level*1000)}"] = swing_high - level * diff
    for level in FIBO_EXTEND:
        df[f"fib_ext_{int(level*1000)}"] = swing_high + level * diff

    df["fib_swing_high"] = swing_high
    df["fib_swing_low"]  = swing_low
    return df


# ── Pivot Points ──────────────────────────────────────────────────────────

def _add_pivots(df: pd.DataFrame) -> pd.DataFrame:
    ph = df["high"].iloc[-2]
    pl = df["low"].iloc[-2]
    pc = df["close"].iloc[-2]

    pivot = (ph + pl + pc) / 3
    df["pivot"]  = pivot
    df["pivot_r1"] = 2 * pivot - pl
    df["pivot_r2"] = pivot + (ph - pl)
    df["pivot_r3"] = ph + 2 * (pivot - pl)
    df["pivot_s1"] = 2 * pivot - ph
    df["pivot_s2"] = pivot - (ph - pl)
    df["pivot_s3"] = pl - 2 * (ph - pivot)
    return df
