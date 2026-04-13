"""
models/rl_agent.py — PPO trading agent via stable-baselines3.
Trains on the BTCTradingEnv. GPU-accelerated via PyTorch backend.
"""

import numpy as np
from utils.logger import get_logger
from models.rl_env import BTCTradingEnv
from config import (
    RL_ALGO, RL_TIMESTEPS, RL_LEARNING_RATE,
    RL_BATCH_SIZE, RL_N_STEPS, MODELS_DIR, DEVICE,
)

logger = get_logger()

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    SB3_OK = True
except ImportError:
    SB3_OK = False
    logger.warning("stable-baselines3 not found.")


class RLAgent:
    def __init__(self, algo: str = RL_ALGO):
        if not SB3_OK:
            raise ImportError("stable-baselines3 required.")
        self.algo  = algo
        self.model = None

    def _make_env(
        self,
        features: np.ndarray,
        prices: np.ndarray,
    ) -> Monitor:
        env = BTCTradingEnv(features, prices)
        return Monitor(env)

    def fit(
        self,
        train_features: np.ndarray,
        train_prices: np.ndarray,
        val_features: np.ndarray   = None,
        val_prices: np.ndarray     = None,
    ) -> None:
        train_env = self._make_env(train_features, train_prices)

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
        )

        algo_cls = PPO if self.algo == "PPO" else SAC

        self.model = algo_cls(
            "MlpPolicy",
            train_env,
            learning_rate   = RL_LEARNING_RATE,
            batch_size      = RL_BATCH_SIZE,
            n_steps         = RL_N_STEPS,      # PPO only
            policy_kwargs   = policy_kwargs,
            device          = DEVICE,
            verbose         = 1,
            tensorboard_log = str(MODELS_DIR / "rl_logs"),
        ) if self.algo == "PPO" else algo_cls(
            "MlpPolicy",
            train_env,
            learning_rate = RL_LEARNING_RATE,
            batch_size    = RL_BATCH_SIZE,
            policy_kwargs = policy_kwargs,
            device        = DEVICE,
            verbose       = 1,
        )

        callbacks = []
        if val_features is not None and val_prices is not None:
            val_env  = self._make_env(val_features, val_prices)
            eval_cb  = EvalCallback(
                val_env,
                best_model_save_path = str(MODELS_DIR),
                log_path             = str(MODELS_DIR / "rl_eval"),
                eval_freq            = 10_000,
                n_eval_episodes      = 3,
                verbose              = 0,
            )
            callbacks.append(eval_cb)

        logger.info(f"Training {self.algo} agent for {RL_TIMESTEPS:,} steps...")
        self.model.learn(total_timesteps=RL_TIMESTEPS, callback=callbacks or None)
        self.save()
        logger.info("RL agent training complete.")

    def predict(
        self,
        features: np.ndarray,
        prices: np.ndarray,
    ) -> dict:
        """
        Run the trained agent on a feature set.
        Returns dict with actions, portfolio values, total PnL.
        """
        if self.model is None:
            raise RuntimeError("Agent not trained. Call fit() first.")

        env = BTCTradingEnv(features, prices)
        obs, _ = env.reset()
        actions, portfolio_values = [], []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            actions.append(int(action))
            portfolio_values.append(info["portfolio_value"])

        return {
            "actions":          actions,
            "portfolio_values": portfolio_values,
            "final_pnl":        info["episode_pnl"],
            "total_trades":     info["total_trades"],
        }

    def save(self, filename: str = "rl_agent") -> None:
        path = MODELS_DIR / filename
        self.model.save(str(path))
        logger.info(f"RL agent saved → {path}")

    def load(self, filename: str = "rl_agent") -> None:
        path = MODELS_DIR / f"{filename}.zip"
        if not path.exists():
            logger.warning(f"RL agent not found at {path}")
            return
        algo_cls   = PPO if self.algo == "PPO" else SAC
        self.model = algo_cls.load(str(MODELS_DIR / filename), device=DEVICE)
        logger.info(f"RL agent loaded ← {path}")