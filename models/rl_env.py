"""
models/rl_env.py — Custom Gymnasium trading environment for BTC.
State: feature window + portfolio state.
Action: 0=hold, 1=buy, 2=sell.
Reward: risk-adjusted PnL with drawdown penalty.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.logger import get_logger
from config import RL_INITIAL_CAPITAL, RL_TRADE_FEE, LSTM_SEQ_LEN

logger = get_logger()


class BTCTradingEnv(gym.Env):
    """
    Single-asset BTC trading environment.

    Observation space : (seq_len, n_features) window of features
                        + 3 portfolio state values (position, cash_ratio, unrealised_pnl)
    Action space      : Discrete(3) — 0=hold, 1=long, 2=close/short
    Reward            : step PnL - drawdown_penalty - trade_cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        seq_len: int   = LSTM_SEQ_LEN,
        initial_capital: float = RL_INITIAL_CAPITAL,
        fee: float     = RL_TRADE_FEE,
    ):
        super().__init__()

        self.features   = features.astype(np.float32)
        self.prices     = prices.astype(np.float32)
        self.seq_len    = seq_len
        self.capital    = initial_capital
        self.fee        = fee
        self.n_features = features.shape[1]

        # Observation: flattened feature window + portfolio state
        obs_dim = seq_len * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=hold, 1=buy, 2=sell

        self._reset_state()

    def _reset_state(self):
        self.t          = self.seq_len
        self.cash       = self.capital
        self.position   = 0.0       # BTC held
        self.entry_price = 0.0
        self.peak_value = self.capital
        self.total_trades = 0
        self.episode_pnl  = 0.0

    def _portfolio_value(self) -> float:
        return self.cash + self.position * self.prices[self.t]

    def _obs(self) -> np.ndarray:
        window = self.features[self.t - self.seq_len : self.t].flatten()
        pv     = self._portfolio_value()
        port   = np.array([
            self.position,
            self.cash / self.capital,
            (pv - self.capital) / self.capital,
        ], dtype=np.float32)
        return np.concatenate([window, port])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        price = self.prices[self.t]
        pv_before = self._portfolio_value()

        # Execute action
        if action == 1 and self.position == 0 and self.cash > 0:
            # Buy — go long with all cash
            btc_bought      = (self.cash * (1 - self.fee)) / price
            self.position   = btc_bought
            self.entry_price = price
            self.cash       = 0.0
            self.total_trades += 1

        elif action == 2 and self.position > 0:
            # Sell — close position
            proceeds    = self.position * price * (1 - self.fee)
            self.cash   = proceeds
            self.position = 0.0
            self.total_trades += 1

        # Advance time
        self.t += 1
        pv_after = self._portfolio_value()

        # Reward: step return
        step_return = (pv_after - pv_before) / (pv_before + 1e-8)

        # Drawdown penalty
        self.peak_value = max(self.peak_value, pv_after)
        drawdown = (self.peak_value - pv_after) / (self.peak_value + 1e-8)
        reward   = step_return - 0.1 * drawdown

        self.episode_pnl = (pv_after - self.capital) / self.capital

        done = self.t >= len(self.prices) - 1
        info = {
            "portfolio_value": pv_after,
            "episode_pnl":     self.episode_pnl,
            "total_trades":    self.total_trades,
            "drawdown":        drawdown,
        }
        return self._obs(), float(reward), done, False, info

    def render(self):
        pv = self._portfolio_value()
        logger.debug(
            f"t={self.t} | price={self.prices[self.t]:.2f} | "
            f"PV={pv:.2f} | pos={self.position:.4f} | cash={self.cash:.2f}"
        )