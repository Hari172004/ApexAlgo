import os
import sys
import logging
import pandas as pd
import numpy as np

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.signal_classifier import SignalClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

def generate_training_data(symbol: str, samples: int = 1000) -> pd.DataFrame:
    """Generate realistic synthetic past trading history for the ML bot."""
    np.random.seed(42 if symbol == "XAUUSD" else 99)
    
    # "rsi", "ema_distance", "atr", "volume_ratio", "session_id", "news_score", "spread", "mtf_confluence"
    data = {
        "rsi": np.random.uniform(20, 80, samples),
        "ema_distance": np.random.normal(0, 1.5, samples),
        "atr": np.random.uniform(0.1, 3.5, samples),
        "volume_ratio": np.random.uniform(0.5, 3.0, samples),
        "session_id": np.random.choice([1, 2, 3], samples), # 1=Asia, 2=London, 3=NY
        "news_score": np.random.uniform(-1, 1, samples),
        "spread": np.random.uniform(8, 25, samples) if symbol == "XAUUSD" else np.random.uniform(0.5, 3.0, samples),
        "mtf_confluence": np.random.choice([0, 1, 2, 3, 4, 5], samples)
    }
    
    df = pd.DataFrame(data)
    
    # Establish a "winning" pattern for the Random Forest to discover
    # (High MTF + Good Volume + Low Spread + News alignment = Win)
    win_prob = np.zeros(samples)
    
    # Baseline 45% win rate
    win_prob += 0.45
    
    # Significant Bonuses
    win_prob += df["mtf_confluence"] * 0.12  # up to +72% total influence
    win_prob += np.where(df["volume_ratio"] > 1.8, 0.20, -0.15)
    win_prob += np.where((df["rsi"] > 35) & (df["rsi"] < 65), 0.15, -0.10)
    win_prob += np.where(df["news_score"] > 0.5, 0.10, -0.05)
    
    # Significant Penalties
    win_prob -= np.where(df["spread"] > (16 if symbol == "XAUUSD" else 1.8), 0.30, 0.0)
    if symbol == "XAUUSD":
        win_prob -= np.where(df["session_id"] == 1, 0.20, 0.0) # Asian session penalty for Gold
        
    win_prob = np.clip(win_prob, 0.01, 0.99)
    
    # Roll the dice to determine win/loss
    random_rolls = np.random.random(samples)
    df["result"] = (random_rolls < win_prob).astype(int)
    
    print(f"Generated {samples} training rows for {symbol}. Label dist: {df['result'].mean():.1%} Wins.")
    return df

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("Initiating Background Autonomous Machine Learning...")
    print(f"{'='*50}\n")
    
    # 1. Train Gold
    print("--- 🥇 TRAINING XAUUSD MODEL ---")
    df_gold = generate_training_data("XAUUSD", 2500)
    gold_ml = SignalClassifier("XAUUSD")
    success, acc = gold_ml.train_model(df_gold)
    print(f"Gold Training Outcome: {'SUCCESS' if success else 'FAILED'} (Accuracy: {acc:.1%})\n")
    
    # 2. Train BTC
    print("--- 🪙 TRAINING BTCUSD MODEL ---")
    df_btc = generate_training_data("BTCUSD", 3000)
    btc_ml = SignalClassifier("BTCUSD")
    success2, acc2 = btc_ml.train_model(df_btc)
    print(f"BTC Training Outcome: {'SUCCESS' if success2 else 'FAILED'} (Accuracy: {acc2:.1%})\n")
    
    print("Optimization Loop Complete. AI Modules Ready.")
