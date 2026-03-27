"""
rl/ppo_agent.py — PPO Agent Wrapper
=====================================
Wraps stable-baselines3 PPO for the Agni-V Scalp Sniper.

Usage:
    from rl.ppo_agent import PPOAgent
    agent = PPOAgent("XAUUSD")
    result = agent.predict(obs_dict)   # {"action": "BUY", "confidence": 0.87}
"""

import os
import logging
import numpy as np
import pandas as pd   # type: ignore
from typing import Dict, Any, Optional

logger = logging.getLogger("agniv.rl.ppo")

MODEL_DIR = os.path.join(os.path.dirname(__file__))
OBS_FEATURES = [
    "rsi", "ema_diff_pct", "atr_norm", "rvol",
    "macd_hist_norm", "bb_pct", "ha_bull", "h1_trend",
    "session_id", "close_norm",
]
ACTION_MAP = {0: "BUY", 1: "SELL", 2: "HOLD"}
REVERSE_MAP = {"BUY": 0, "SELL": 1, "HOLD": 2}


class PPOAgent:
    """
    Loads a pre-trained PPO model and provides a predict() interface
    compatible with the Scalp Sniper signal pipeline.

    Falls back to pass-through (HOLD, confidence=1.0) if no model file exists.
    """

    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol      = symbol
        self.model_path  = os.path.join(MODEL_DIR, f"{symbol}_ppo.zip")
        self._model      = None
        self._available  = False
        self._load()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def predict(self, obs_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Given a dict of observation features (matching OBS_FEATURES keys),
        return {"action": "BUY"|"SELL"|"HOLD", "confidence": float}.

        If no model is loaded, returns {"action": "HOLD", "confidence": 1.0}
        to act as a transparent pass-through so the bot still works.
        """
        if not self._available or self._model is None:
            return {"action": "HOLD", "confidence": 1.0, "ppo_active": False}

        try:
            obs = self._dict_to_obs(obs_dict)
            action, _ = self._model.predict(obs, deterministic=True)
            action_int = int(action)

            # Get probability distribution for confidence score
            obs_tensor = self._to_tensor(obs)
            dist = self._model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            confidence = float(probs[action_int])

            return {
                "action":     ACTION_MAP.get(action_int, "HOLD"),
                "confidence": round(confidence, 3),
                "ppo_active": True,
                "probs":      {ACTION_MAP[i]: round(float(p), 3) for i, p in enumerate(probs)},
            }
        except Exception as e:
            logger.warning(f"[PPO:{self.symbol}] Predict error: {e}. Falling back to pass-through.")
            return {"action": "HOLD", "confidence": 1.0, "ppo_active": False}

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 200_000,
        n_envs: int = 4,
    ) -> float:
        """
        Train a PPO model on the provided DataFrame and save to disk.
        Returns the mean episode reward from the final evaluation.
        """
        try:
            from stable_baselines3 import PPO                           # type: ignore
            from stable_baselines3.common.env_util import make_vec_env  # type: ignore
            from stable_baselines3.common.callbacks import EvalCallback # type: ignore
            from rl.trading_env import ScalpTradingEnv                  # type: ignore
        except ImportError as e:
            logger.error(f"[PPO] stable-baselines3 not installed: {e}")
            return 0.0

        logger.info(f"[PPO:{self.symbol}] Starting PPO training — {total_timesteps:,} timesteps")

        def _make_env():
            return ScalpTradingEnv(df.copy(), symbol=self.symbol)

        vec_env = make_vec_env(_make_env, n_envs=n_envs)

        model = PPO(
            policy           = "MlpPolicy",
            env              = vec_env,
            n_steps          = 2048,
            batch_size       = 64,
            n_epochs         = 10,
            gamma            = 0.99,
            gae_lambda       = 0.95,
            clip_range       = 0.2,
            ent_coef         = 0.01,      # encourage exploration
            learning_rate    = 3e-4,
            verbose          = 1,
        )

        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        logger.info(f"[PPO:{self.symbol}] Model saved → {self.model_path}")

        # Reload so predict() is immediately available
        self._model     = model
        self._available = True

        # Quick evaluation
        eval_env = ScalpTradingEnv(df.copy(), symbol=self.symbol)
        obs, _ = eval_env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        logger.info(f"[PPO:{self.symbol}] Eval reward over {steps} bars: {total_reward:.4f}")
        return total_reward

    def is_available(self) -> bool:
        return self._available

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self.model_path):
            logger.info(
                f"[PPO:{self.symbol}] No model file at {self.model_path}. "
                "Pass-through mode active. Run rl/train_ppo.py to train."
            )
            return
        try:
            from stable_baselines3 import PPO  # type: ignore
            self._model     = PPO.load(self.model_path)
            self._available = True
            logger.info(f"[PPO:{self.symbol}] Model loaded from {self.model_path}")
        except ImportError:
            logger.warning("[PPO] stable-baselines3 not installed — pass-through mode.")
        except Exception as e:
            logger.error(f"[PPO:{self.symbol}] Failed to load model: {e}")

    def _dict_to_obs(self, obs_dict: Dict[str, float]) -> np.ndarray:
        obs = np.array(
            [obs_dict.get(f, 0.0) for f in OBS_FEATURES],
            dtype=np.float32,
        )
        return np.clip(obs, -5.0, 5.0)

    def _to_tensor(self, obs: np.ndarray):
        """Convert obs array to the tensor format expected by SB3 policy."""
        import torch  # type: ignore
        return torch.tensor(obs[np.newaxis, :], dtype=torch.float32)
