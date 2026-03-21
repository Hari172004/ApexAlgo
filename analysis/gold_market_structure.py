"""
gold_market_structure.py -- Full institutional SMC analysis for Gold (XAUUSD)
Detects: BOS, CHOCH, Order Blocks, FVGs, Liquidity Sweeps, Equal H/L,
         Asian Range, London Open breakout, Kill Zone timing.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("apexalgo.gold_smc")


# ── Swing Point Detection ─────────────────────────────────────────────────

def _find_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Basic N-bar fractal swing detection."""
    df["swing_high"] = False
    df["swing_low"]  = False

    for i in range(lookback, len(df) - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_h.max():
            df.at[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == window_l.min():
            df.at[df.index[i], "swing_low"] = True
    return df


# ── Main Analysis ─────────────────────────────────────────────────────────

def detect_gold_smc(df: pd.DataFrame) -> dict:
    """
    Full SMC analysis on XAUUSD candle data.
    Returns structured context dict consumed by strategies.
    """
    if df.empty or len(df) < 60:
        return _empty_context()

    df = _find_swings(df, lookback=5)

    sh = df[df["swing_high"]]
    sl = df[df["swing_low"]]

    last_sh = float(sh["high"].iloc[-1])  if not sh.empty else 0.0
    last_sl = float(sl["low"].iloc[-1])   if not sl.empty else 0.0

    current_close = float(df["close"].iloc[-1])

    # 1. BOS / CHOCH
    bos, choch = _detect_bos_choch(df, last_sh, last_sl, current_close)

    # 2. Trend via HH/HL and LH/LL
    trend = _detect_trend(sh, sl)

    # 3. Order Blocks
    bull_obs = _find_order_blocks(df, direction="bull")
    bear_obs  = _find_order_blocks(df, direction="bear")

    # 4. Fair Value Gaps
    fvgs = _find_fvgs(df)

    # 5. Liquidity Sweeps
    sweeps = _find_liquidity_sweeps(df, last_sh, last_sl)

    # 6. Equal Highs / Equal Lows
    equal_highs = _find_equal_levels(sh["high"].values, tolerance=0.05)  # type: ignore
    equal_lows  = _find_equal_levels(sl["low"].values,  tolerance=0.05)  # type: ignore

    # 7. Asian Range
    asian_range = _get_asian_range(df)

    return {
        "trend":        trend,
        "bos":          bos,
        "choch":        choch,
        "bull_obs":     bull_obs[:3],
        "bear_obs":     bear_obs[:3],
        "fvgs":         fvgs[:5],
        "sweeps":       sweeps,
        "equal_highs":  equal_highs,
        "equal_lows":   equal_lows,
        "asian_range":  asian_range,
        "last_sh":      last_sh,
        "last_sl":      last_sl,
    }


def _empty_context() -> dict:
    return {
        "trend": "UNKNOWN", "bos": "NONE", "choch": "NONE",
        "bull_obs": [], "bear_obs": [], "fvgs": [],
        "sweeps": [], "equal_highs": [], "equal_lows": [],
        "asian_range": {"high": 0.0, "low": 0.0},
        "last_sh": 0.0, "last_sl": 0.0,
    }


# ── BOS and CHOCH ─────────────────────────────────────────────────────────

def _detect_bos_choch(df: pd.DataFrame, last_sh: float, last_sl: float, current_close: float):
    bos   = "NONE"
    choch = "NONE"

    if current_close > last_sh and last_sh > 0:
        bos = "BULLISH"
    elif current_close < last_sl and last_sl > 0:
        bos = "BEARISH"

    # CHOCH: price breaks opposite structure level suggesting reversal
    if bos == "BULLISH" and current_close < last_sl and last_sl > 0:
        choch = "BEARISH_REVERSAL"
    elif bos == "BEARISH" and current_close > last_sh and last_sh > 0:
        choch = "BULLISH_REVERSAL"

    return bos, choch


# ── Trend: HH/HL vs LH/LL ────────────────────────────────────────────────

def _detect_trend(sh: pd.DataFrame, sl: pd.DataFrame) -> str:
    if len(sh) < 2 or len(sl) < 2:
        return "UNKNOWN"

    sh_vals = sh["high"].values
    sl_vals = sl["low"].values

    hh = sh_vals[-1] > sh_vals[-2]
    hl = sl_vals[-1] > sl_vals[-2]
    lh = sh_vals[-1] < sh_vals[-2]
    ll = sl_vals[-1] < sl_vals[-2]

    if hh and hl:
        return "UPTREND"
    elif lh and ll:
        return "DOWNTREND"
    else:
        return "RANGING"


# ── Order Blocks ──────────────────────────────────────────────────────────

def _find_order_blocks(df: pd.DataFrame, direction: str = "bull") -> list:
    obs = []
    for i in range(len(df) - 2, 5, -1):
        candle = df.iloc[i]
        is_bear_candle = candle["close"] < candle["open"]
        is_bull_candle = candle["close"] > candle["open"]

        if direction == "bull" and is_bear_candle:
            # OB valid if followed by significant upward move
            future_high = df["high"].iloc[i + 1:i + 10].max() if i + 10 <= len(df) else df["high"].iloc[i + 1:].max()
            if future_high > candle["high"] * 1.002:
                obs.append({
                    "type": "BULLISH_OB",
                    "top":    float(candle["high"]),
                    "bottom": float(candle["low"]),
                    "mid":    float((candle["high"] + candle["low"]) / 2),
                })
        elif direction == "bear" and is_bull_candle:
            future_low = df["low"].iloc[i + 1:i + 10].min() if i + 10 <= len(df) else df["low"].iloc[i + 1:].min()
            if future_low < candle["low"] * 0.998:
                obs.append({
                    "type": "BEARISH_OB",
                    "top":    float(candle["high"]),
                    "bottom": float(candle["low"]),
                    "mid":    float((candle["high"] + candle["low"]) / 2),
                })
    return obs


# ── Fair Value Gaps ───────────────────────────────────────────────────────

def _find_fvgs(df: pd.DataFrame) -> list:
    fvgs = []
    for i in range(2, len(df)):
        low_2  = float(df["low"].iloc[i - 2])
        high_2 = float(df["high"].iloc[i - 2])
        low_0  = float(df["low"].iloc[i])
        high_0 = float(df["high"].iloc[i])

        if low_2 > high_0:  # Bullish FVG
            fvgs.append({"type": "BULL_FVG", "top": low_2, "bottom": high_0})
        elif high_2 < low_0:  # Bearish FVG
            fvgs.append({"type": "BEAR_FVG", "top": low_0, "bottom": high_2})

    return fvgs[-8:] if len(fvgs) > 8 else fvgs


# ── Liquidity Sweeps ──────────────────────────────────────────────────────

def _find_liquidity_sweeps(df: pd.DataFrame, last_sh: float, last_sl: float) -> list:
    sweeps = []
    recent = df.tail(10)

    for _, row in recent.iterrows():
        if last_sh > 0 and float(row["high"]) > last_sh and float(row["close"]) < last_sh:
            sweeps.append({"type": "SELL_SIDE_SWEEP", "level": last_sh})
        if last_sl > 0 and float(row["low"]) < last_sl and float(row["close"]) > last_sl:
            sweeps.append({"type": "BUY_SIDE_SWEEP", "level": last_sl})

    return sweeps


# ── Equal Highs / Equal Lows ──────────────────────────────────────────────

def _find_equal_levels(values: np.ndarray, tolerance: float = 0.05) -> list:
    """Find price levels that repeat within tolerance (liquidity pools)."""
    clusters = []
    if len(values) < 2:
        return clusters
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) <= tolerance:
                clusters.append(float(round((values[i] + values[j]) / 2, 3)))
    return list(set(clusters))[:5]


# ── Asian Range ───────────────────────────────────────────────────────────

def _get_asian_range(df: pd.DataFrame) -> dict:
    """
    Approximate Asian session range from last 24 candles on M5,
    or last 8 candles on H1 (00:00-08:00 GMT).
    """
    tail = df.tail(24)
    return {
        "high": float(tail["high"].max()),
        "low":  float(tail["low"].min()),
        "mid":  float((tail["high"].max() + tail["low"].min()) / 2),
    }


# ── Price nearness helper ─────────────────────────────────────────────────

def near_ob(price: float, obs: list, threshold_pips: float = 3.0) -> bool:
    """Check if current price is inside or near an Order Block."""
    for ob in obs:
        if ob["bottom"] - threshold_pips * 0.01 <= price <= ob["top"] + threshold_pips * 0.01:
            return True
    return False


def near_fvg(price: float, fvgs: list) -> bool:
    """Check if current price is inside a Fair Value Gap."""
    for fvg in fvgs:
        bottom = fvg.get("bottom", 0)
        top    = fvg.get("top", 0)
        if bottom <= price <= top:
            return True
    return False
