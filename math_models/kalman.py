"""
kalman.py — Kalman Filter & Smoother
Covers: linear state-space models, trend extraction, latent factor estimation,
        adaptive Kalman for time-varying parameters.
Maps to: Kalman Filter block (state-space, latent factors).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Base Linear Kalman Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Standard linear Gaussian Kalman filter.

    State-space model:
        x_t = F x_{t-1} + B u_t + q_t,    q_t ~ N(0, Q)    [transition]
        y_t = H x_t     + r_t,             r_t ~ N(0, R)    [observation]

    Parameters
    ----------
    F : (n,n)  state transition matrix
    H : (m,n)  observation matrix
    Q : (n,n)  process noise covariance
    R : (m,m)  observation noise covariance
    x0: (n,)   initial state estimate
    P0: (n,n)  initial state covariance
    B : (n,k)  control matrix (optional)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        B: Optional[np.ndarray] = None,
    ):
        self.F = np.atleast_2d(F).astype(float)
        self.H = np.atleast_2d(H).astype(float)
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)
        self.x = np.atleast_1d(x0).astype(float)
        self.P = np.atleast_2d(P0).astype(float)
        self.B = B

        self.n = self.F.shape[0]
        self.m = self.H.shape[0]

        # Storage for smoother
        self._xs = []
        self._Ps = []
        self._log_likelihoods = []

    def predict(self, u: Optional[np.ndarray] = None):
        """Time update (predict step)."""
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x += self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y: np.ndarray) -> float:
        """
        Measurement update.

        Returns
        -------
        log-likelihood contribution of this observation.
        """
        y = np.atleast_1d(y).astype(float)
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        innov = y - self.H @ self.x                       # innovation

        self.x = self.x + K @ innov
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form (stable)

        # Log-likelihood
        sign, logdet = np.linalg.slogdet(S)
        log_lik = -0.5 * (self.m * np.log(2 * np.pi) + logdet + innov @ np.linalg.inv(S) @ innov)

        self._xs.append(self.x.copy())
        self._Ps.append(self.P.copy())
        self._log_likelihoods.append(float(log_lik))

        return float(log_lik)

    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run filter over a sequence of observations.

        Parameters
        ----------
        observations : (T, m) or (T,) array

        Returns
        -------
        (filtered_states, filtered_covs, total_log_lik)
            filtered_states : (T, n)
            filtered_covs   : (T, n, n)
        """
        obs = np.atleast_2d(observations)
        if obs.shape[0] == 1:
            obs = obs.T  # (T, 1)

        self._xs, self._Ps, self._log_likelihoods = [], [], []

        for y in obs:
            self.predict()
            self.update(y)

        states = np.array(self._xs)      # (T, n)
        covs = np.array(self._Ps)        # (T, n, n)
        total_ll = float(np.sum(self._log_likelihoods))

        logger.info(f"Kalman filter complete. T={len(obs)}, log-lik={total_ll:.2f}")
        return states, covs, total_ll

    def smooth(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rauch-Tung-Striebel (RTS) smoother.
        Call after filter().

        Returns
        -------
        (smoothed_states, smoothed_covs) each (T, n) and (T, n, n)
        """
        states, covs, _ = self.filter(observations)
        T = len(states)

        xs_smooth = states.copy()
        Ps_smooth = covs.copy()

        for t in range(T - 2, -1, -1):
            P_pred = self.F @ Ps_smooth[t] @ self.F.T + self.Q
            G = covs[t] @ self.F.T @ np.linalg.inv(P_pred)   # smoother gain

            xs_smooth[t] = states[t] + G @ (xs_smooth[t + 1] - self.F @ states[t])
            Ps_smooth[t] = covs[t] + G @ (Ps_smooth[t + 1] - P_pred) @ G.T

        return xs_smooth, Ps_smooth


# ---------------------------------------------------------------------------
# 2. Local Level Model (random walk + noise) — trend extraction
# ---------------------------------------------------------------------------

def local_level_model(
    sigma_level: float = 0.01,
    sigma_obs: float = 0.05,
    x0: float = 0.0,
) -> KalmanFilter:
    """
    Construct a Local Level (random walk + measurement noise) Kalman filter.
    Useful for extracting a smooth price trend.

    State: x_t = x_{t-1} + η_t,   η_t ~ N(0, σ_level²)
    Obs:   y_t = x_t + ε_t,       ε_t ~ N(0, σ_obs²)
    """
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[sigma_level ** 2]])
    R = np.array([[sigma_obs ** 2]])
    x0_vec = np.array([x0])
    P0 = np.array([[sigma_obs ** 2]])

    return KalmanFilter(F, H, Q, R, x0_vec, P0)


# ---------------------------------------------------------------------------
# 3. Local Linear Trend — level + slope
# ---------------------------------------------------------------------------

def local_linear_trend(
    sigma_level: float = 0.01,
    sigma_slope: float = 0.001,
    sigma_obs: float = 0.05,
    x0: float = 0.0,
    slope0: float = 0.0,
) -> KalmanFilter:
    """
    Local linear trend model:
        level_t  = level_{t-1} + slope_{t-1} + η_t
        slope_t  = slope_{t-1} + ζ_t
        y_t      = level_t + ε_t

    Captures both trend and drift in a price series.
    """
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([sigma_level ** 2, sigma_slope ** 2])
    R = np.array([[sigma_obs ** 2]])
    x0_vec = np.array([x0, slope0])
    P0 = np.diag([sigma_obs ** 2, sigma_slope ** 2])

    return KalmanFilter(F, H, Q, R, x0_vec, P0)


# ---------------------------------------------------------------------------
# 4. Adaptive Kalman — dynamic beta estimation (for factor models)
# ---------------------------------------------------------------------------

class AdaptiveKalmanRegression:
    """
    Kalman filter for time-varying regression: y_t = β_t · x_t + ε_t
    β_t evolves as a random walk: β_t = β_{t-1} + η_t

    Used to estimate a time-varying hedge ratio in cointegration / pair trading.
    """

    def __init__(
        self,
        n_features: int = 1,
        delta: float = 1e-4,       # state noise scaling (higher = faster adaptation)
        R_init: float = 1.0,
        P_init: float = 1.0,
    ):
        self.n = n_features
        self.delta = delta
        self.Vw = delta / (1 - delta) * np.eye(n_features)   # process noise
        self.Ve = R_init                                       # obs noise (scalar)

        self.beta = np.zeros(n_features)
        self.P = P_init * np.eye(n_features)

        self.betas = []
        self.errors = []

    def update(self, y: float, x: np.ndarray) -> np.ndarray:
        """
        Update the time-varying beta with one observation (y, x).

        Returns current beta estimate.
        """
        x = np.atleast_1d(x).astype(float)

        # Predict
        P_pred = self.P + self.Vw

        # Innovation
        y_hat = float(x @ self.beta)
        innov = float(y) - y_hat
        self.errors.append(innov)

        # Update observation noise adaptively (from innovation variance)
        # (optional: static Ve also works)
        S = float(x @ P_pred @ x) + self.Ve
        K = P_pred @ x / S

        self.beta = self.beta + K * innov
        self.P = P_pred - np.outer(K, x) @ P_pred

        self.betas.append(self.beta.copy())
        return self.beta.copy()

    def run(self, y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
        """
        Run over full series.

        Returns DataFrame of beta estimates (columns = feature names).
        """
        betas = []
        for yt, (_, row) in zip(y.values, X.iterrows()):
            b = self.update(float(yt), row.values)
            betas.append(b)

        return pd.DataFrame(betas, index=y.index, columns=X.columns)


# ---------------------------------------------------------------------------
# 5. High-level convenience: extract price trend
# ---------------------------------------------------------------------------

def extract_trend(
    prices: pd.Series,
    model: str = "local_linear",
    sigma_level: float = 0.005,
    sigma_slope: float = 0.001,
    sigma_obs: float = 0.02,
) -> pd.DataFrame:
    """
    Extract smooth trend from a price series using Kalman smoother.

    Parameters
    ----------
    prices : pd.Series of prices
    model  : 'local_level' | 'local_linear'

    Returns
    -------
    DataFrame with columns: price, trend, (slope if local_linear)
    """
    log_p = np.log(prices.values)

    if model == "local_level":
        kf = local_level_model(sigma_level, sigma_obs, x0=log_p[0])
        smooth_states, _ = kf.smooth(log_p)
        trend = np.exp(smooth_states[:, 0])
        out = pd.DataFrame({"price": prices.values, "trend": trend}, index=prices.index)
    else:
        kf = local_linear_trend(sigma_level, sigma_slope, sigma_obs, x0=log_p[0])
        smooth_states, _ = kf.smooth(log_p)
        trend = np.exp(smooth_states[:, 0])
        slope = smooth_states[:, 1]
        out = pd.DataFrame(
            {"price": prices.values, "trend": trend, "slope": slope},
            index=prices.index,
        )

    logger.info(f"Kalman trend extracted: model={model}, T={len(prices)}")
    return out