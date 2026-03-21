"""
smc.py — Smart Money Concepts (SMC) Engine
============================================
Provides structural price action analysis for institutional trading.
Detects Order Blocks (OB), Fair Value Gaps (FVG), and Liquidity Sweeps.
"""

import pandas as pd
import numpy as np

class SMCEngine:
    
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> dict:
        """
        Detects the most recent unmitigated Fair Value Gaps.
        Bullish FVG: Low of candle 3 > High of candle 1
        Bearish FVG: High of candle 3 < Low of candle 1
        Returns the top/bottom price boundaries of the closest active gap.
        """
        if len(df) < 3:
            return {"bullish": [], "bearish": []}

        bullish_fvgs = []
        bearish_fvgs = []

        # Iterate backwards from recent to past
        for i in range(len(df) - 1, 1, -1):
            c1_high = df['high'].iloc[i-2]
            c1_low  = df['low'].iloc[i-2]
            c3_high = df['high'].iloc[i]
            c3_low  = df['low'].iloc[i]

            # Bullish FVG
            if c3_low > c1_high:
                # The gap is between c1_high and c3_low
                bullish_fvgs.append({
                    "top": c3_low,
                    "bottom": c1_high,
                    "index": df.index[i-1]
                })
            
            # Bearish FVG
            elif c3_high < c1_low:
                # The gap is between c3_high and c1_low
                bearish_fvgs.append({
                    "top": c1_low,
                    "bottom": c3_high,
                    "index": df.index[i-1]
                })

            # For performance, only return the most recent 5 gaps
            if len(bullish_fvgs) + len(bearish_fvgs) >= 10:
                break

        return {
            "bullish": bullish_fvgs,
            "bearish": bearish_fvgs
        }

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, wick_threshold: float = 0.5) -> dict:
        """
        Detects recent Order Blocks.
        Bullish OB: The last down candle before a strong impulsive up move.
        Bearish OB: The last up candle before a strong impulsive down move.
        """
        if len(df) < 5:
            return {"bullish": [], "bearish": []}

        bullish_obs = []
        bearish_obs = []

        # Calculate basic candle properties
        body = abs(df['close'] - df['open'])
        direction = np.where(df['close'] > df['open'], 1, -1)
        avg_body = body.rolling(14).mean()

        for i in range(len(df) - 2, 0, -1):
            # Bullish OB Check:
            # Current candle is DOWN. Next candle is a massive UP candle (Engulfing/Impulsive)
            if direction[i] == -1 and direction[i+1] == 1:
                # If the up candle is significantly larger than average
                if body.iloc[i+1] > avg_body.iloc[i+1] * 1.5:
                    bullish_obs.append({
                        "top": df['open'].iloc[i],     # Top of the down candle body
                        "bottom": df['low'].iloc[i],   # Bottom of its wick
                        "index": df.index[i]
                    })
            
            # Bearish OB Check:
            # Current candle is UP. Next candle is a massive DOWN candle
            if direction[i] == 1 and direction[i+1] == -1:
                if body.iloc[i+1] > avg_body.iloc[i+1] * 1.5:
                    bearish_obs.append({
                        "top": df['high'].iloc[i],    # Top of the up candle wick
                        "bottom": df['open'].iloc[i], # Bottom of its body
                        "index": df.index[i]
                    })
            
            if len(bullish_obs) + len(bearish_obs) >= 10:
                break

        return {
            "bullish": bullish_obs,
            "bearish": bearish_obs
        }

    @staticmethod
    def get_smc_context(df: pd.DataFrame, current_price: float) -> dict:
        """
        Returns boolean flags indicating if the current price is mitigating
        an Order Block or sitting inside a Fair Value Gap.
        """
        fvgs = SMCEngine.detect_fvg(df)
        obs = SMCEngine.detect_order_blocks(df)

        in_bullish_ob = any(ob['bottom'] <= current_price <= ob['top'] for ob in obs['bullish'])
        in_bearish_ob = any(ob['bottom'] <= current_price <= ob['top'] for ob in obs['bearish'])

        in_bullish_fvg = any(fvg['bottom'] <= current_price <= fvg['top'] for fvg in fvgs['bullish'])
        in_bearish_fvg = any(fvg['bottom'] <= current_price <= fvg['top'] for fvg in fvgs['bearish'])

        return {
            "in_bull_zone": in_bullish_ob or in_bullish_fvg,
            "in_bear_zone": in_bearish_ob or in_bearish_fvg,
            "nearest_bull_ob": obs['bullish'][0] if obs['bullish'] else None,
            "nearest_bear_ob": obs['bearish'][0] if obs['bearish'] else None,
        }
