"""
rl/train_ppo.py — Standalone PPO Training Script
==================================================
Trains separate PPO models for Gold (XAUUSD) and BTC (BTCUSD).

Usage (from project root):
    python rl/train_ppo.py --symbol XAUUSD
    python rl/train_ppo.py --symbol BTCUSD
    python rl/train_ppo.py               # trains both

Output:
    rl/XAUUSD_ppo.zip
    rl/BTCUSD_ppo.zip
"""

import argparse
import logging
import os
import sys

# Allow running from repo root
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import pandas as pd  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agniv.rl.train")

SYMBOL_CSV_MAP = {
    "XAUUSD": os.path.join(ROOT, "data", "XAUUSD_D1_history.csv"),
    "BTCUSD": os.path.join(ROOT, "data", "BTCUSD_D1_history.csv"),
}

DEFAULT_TIMESTEPS = {
    "XAUUSD": 300_000,    # Gold: longer training for more nuanced regime detection
    "BTCUSD": 200_000,    # BTC: faster price action, fewer timesteps needed
}


def load_data(symbol: str) -> pd.DataFrame:
    csv_path = SYMBOL_CSV_MAP.get(symbol)
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No CSV found for {symbol}. Expected: {csv_path}\n"
            "Run download_history.py first, or place a CSV with columns "
            "[open, high, low, close, volume] in data/."
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    logger.info(f"[{symbol}] Loaded {len(df):,} bars from {csv_path}")
    return df


def train_symbol(symbol: str, timesteps: int):
    logger.info(f"{'='*60}")
    logger.info(f"  Training PPO for {symbol}  ({timesteps:,} timesteps)")
    logger.info(f"{'='*60}")

    df = load_data(symbol)

    from rl.ppo_agent import PPOAgent  # type: ignore
    agent = PPOAgent(symbol)
    final_reward = agent.train(df, total_timesteps=timesteps)

    logger.info(f"[{symbol}] ✅ Training complete. Final eval reward: {final_reward:.4f}")
    model_path = os.path.join(os.path.dirname(__file__), f"{symbol}_ppo.zip")
    logger.info(f"[{symbol}] Model saved to: {model_path}")
    return final_reward


def main():
    parser = argparse.ArgumentParser(description="Train PPO agents for Agni-V Scalp Sniper")
    parser.add_argument(
        "--symbol",
        choices=["XAUUSD", "BTCUSD"],
        default=None,
        help="Symbol to train (default: trains both XAUUSD and BTCUSD)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override number of training timesteps"
    )
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["XAUUSD", "BTCUSD"]

    results = {}
    for sym in symbols:
        ts = args.timesteps or DEFAULT_TIMESTEPS[sym]
        try:
            reward = train_symbol(sym, ts)
            results[sym] = {"status": "OK", "reward": reward}
        except Exception as e:
            logger.error(f"[{sym}] Training failed: {e}")
            results[sym] = {"status": "FAILED", "error": str(e)}

    logger.info("\n" + "="*60)
    logger.info("  Training Summary")
    logger.info("="*60)
    for sym, res in results.items():
        if res["status"] == "OK":
            logger.info(f"  {sym}: ✅ OK  |  Eval Reward = {res['reward']:.4f}")
        else:
            logger.info(f"  {sym}: ❌ FAILED — {res['error']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
