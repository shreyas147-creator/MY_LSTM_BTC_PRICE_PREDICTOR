"""
copula.py — Copula Theory: Tail Dependence & Joint Distribution Modelling
Covers: Gaussian, Clayton, Gumbel, Frank copulas; Sklar's theorem;
        tail dependence coefficients; copula-based joint return simulation.
Maps to: Copula Theory block (tail dependence, Sklar's thm.).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple
from scipy import stats
from scipy.optimize import minimize_scalar, minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Pseudo-uniform transforms (probability integral transform)
# ---------------------------------------------------------------------------

def to_uniform(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Apply empirical CDF (rank-based) to each column → pseudo-uniform marginals.
    Required by Sklar's theorem before fitting a copula.
    """
    n = len(returns)
    u = returns.rank() / (n + 1)    # ties handled by mean rank
    return u


def from_uniform_normal(u: np.ndarray) -> np.ndarray:
    """Inverse-normal transform: U[0,1] → N(0,1) via ppf."""
    return stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


# ---------------------------------------------------------------------------
# 2. Gaussian Copula
# ---------------------------------------------------------------------------

class GaussianCopula:
    """
    Multivariate Gaussian copula: captures symmetric linear dependence.
    C(u1,…,uk) = Φ_R(Φ⁻¹(u1),…,Φ⁻¹(uk))   where R is the correlation matrix.
    """

    def __init__(self):
        self.R = None         # correlation matrix
        self.n_assets = None

    def fit(self, u: np.ndarray) -> "GaussianCopula":
        """
        Fit correlation matrix to pseudo-uniform observations.

        Parameters
        ----------
        u : (T, k) array of pseudo-uniforms in (0,1)
        """
        z = from_uniform_normal(u)      # (T, k) Gaussian quantiles
        self.R = np.corrcoef(z.T)
        self.n_assets = u.shape[1]
        logger.info(f"GaussianCopula fitted: n_assets={self.n_assets}\nCorr:\n{np.round(self.R, 3)}")
        return self

    def simulate(self, n: int, seed: int = 42) -> np.ndarray:
        """
        Draw n samples from the copula → (n, k) pseudo-uniform matrix.
        """
        rng = np.random.default_rng(seed)
        Z = rng.multivariate_normal(np.zeros(self.n_assets), self.R, size=n)
        return stats.norm.cdf(Z)

    def log_density(self, u: np.ndarray) -> np.ndarray:
        """Log-density of the Gaussian copula at each observation."""
        z = from_uniform_normal(u)
        k = self.n_assets
        R_inv = np.linalg.inv(self.R)
        sign, logdet = np.linalg.slogdet(self.R)

        # c(u) = φ_R(z) / ∏φ(z_i)
        # log c = -0.5 log|R| - 0.5 z'(R⁻¹ - I)z
        quad = np.einsum("ti,ij,tj->t", z, R_inv - np.eye(k), z)
        log_c = -0.5 * logdet - 0.5 * quad
        return log_c

    @property
    def tail_dependence(self) -> dict:
        """
        Gaussian copula has zero tail dependence for |ρ| < 1.
        Returns λ_U = λ_L = 0 for all pairs.
        """
        return {"upper": 0.0, "lower": 0.0, "note": "Gaussian copula: asymptotically independent tails"}


# ---------------------------------------------------------------------------
# 3. Archimedean Copulas (Clayton, Gumbel, Frank)
# ---------------------------------------------------------------------------

class ClaytonCopula:
    """
    Clayton copula — exhibits lower tail dependence (joint crashes).
    C(u,v; θ) = (u^{-θ} + v^{-θ} - 1)^{-1/θ},   θ > 0
    λ_L = 2^{-1/θ},   λ_U = 0
    """

    def __init__(self, theta: Optional[float] = None):
        self.theta = theta

    def fit(self, u: np.ndarray, v: np.ndarray) -> "ClaytonCopula":
        """MLE estimation of θ."""
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)

        def neg_log_lik(theta):
            if theta <= 0:
                return 1e10
            log_c = (
                np.log(1 + theta)
                - (1 + theta) * (np.log(u) + np.log(v))
                - (2 + 1 / theta) * np.log(u ** (-theta) + v ** (-theta) - 1)
            )
            return -np.sum(log_c)

        result = minimize_scalar(neg_log_lik, bounds=(0.01, 20), method="bounded")
        self.theta = float(result.x)
        logger.info(f"Clayton copula: θ={self.theta:.4f}, λ_L={self.lower_tail:.4f}")
        return self

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        return (u ** (-self.theta) + v ** (-self.theta) - 1) ** (-1 / self.theta)

    def simulate(self, n: int, seed: int = 42) -> np.ndarray:
        """Conditional inversion sampling."""
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n)
        w = rng.uniform(0, 1, n)
        # Conditional CDF inversion
        v = u * (w ** (-self.theta / (1 + self.theta)) - 1 + u ** self.theta) ** (-1 / self.theta)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        return np.column_stack([u, v])

    @property
    def lower_tail(self) -> float:
        return float(2 ** (-1 / self.theta)) if self.theta else 0.0

    @property
    def upper_tail(self) -> float:
        return 0.0


class GumbelCopula:
    """
    Gumbel copula — exhibits upper tail dependence (joint rallies).
    C(u,v; θ) = exp(-((-ln u)^θ + (-ln v)^θ)^{1/θ}),   θ ≥ 1
    λ_U = 2 - 2^{1/θ},   λ_L = 0
    """

    def __init__(self, theta: Optional[float] = None):
        self.theta = theta

    def fit(self, u: np.ndarray, v: np.ndarray) -> "GumbelCopula":
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)

        def neg_log_lik(theta):
            if theta < 1:
                return 1e10
            a = (-np.log(u)) ** theta + (-np.log(v)) ** theta
            C = np.exp(-(a ** (1 / theta)))
            log_c = (
                np.log(C)
                + (theta - 1) * np.log(-np.log(u))
                + (theta - 1) * np.log(-np.log(v))
                - np.log(u) - np.log(v)
                + (1 / theta - 2) * np.log(a)
                + np.log(a ** (1 / theta) + theta - 1)
            )
            return -np.sum(log_c)

        result = minimize_scalar(neg_log_lik, bounds=(1.001, 20), method="bounded")
        self.theta = float(result.x)
        logger.info(f"Gumbel copula: θ={self.theta:.4f}, λ_U={self.upper_tail:.4f}")
        return self

    @property
    def upper_tail(self) -> float:
        return float(2 - 2 ** (1 / self.theta)) if self.theta else 0.0

    @property
    def lower_tail(self) -> float:
        return 0.0


class FrankCopula:
    """
    Frank copula — symmetric, no tail dependence. Useful baseline.
    λ_U = λ_L = 0
    """

    def __init__(self, theta: Optional[float] = None):
        self.theta = theta

    def fit(self, u: np.ndarray, v: np.ndarray) -> "FrankCopula":
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)

        def neg_log_lik(theta):
            if abs(theta) < 1e-6:
                return 1e10
            exp_t = np.exp(-theta)
            numer = -theta * (1 - exp_t) * np.exp(-theta * (u + v))
            denom = (1 - exp_t - (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v))) ** 2
            log_c = np.log(np.maximum(numer / denom, 1e-300))
            return -np.sum(log_c)

        result = minimize_scalar(neg_log_lik, bounds=(-20, 20), method="bounded")
        self.theta = float(result.x)
        logger.info(f"Frank copula: θ={self.theta:.4f}")
        return self

    @property
    def upper_tail(self) -> float:
        return 0.0

    @property
    def lower_tail(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# 4. Empirical tail dependence coefficient
# ---------------------------------------------------------------------------

def empirical_tail_dependence(
    u: np.ndarray,
    v: np.ndarray,
    q: float = 0.05,
) -> dict:
    """
    Non-parametric tail dependence coefficients.

    λ_L = P(V < q | U < q)  — lower tail (joint crash)
    λ_U = P(V > 1-q | U > 1-q)  — upper tail (joint rally)

    Parameters
    ----------
    u, v : (T,) pseudo-uniform series
    q    : tail quantile (e.g. 0.05 = bottom/top 5%)
    """
    n = len(u)
    lower = float(np.mean((u < q) & (v < q))) / float(np.mean(u < q) + 1e-12)
    upper = float(np.mean((u > 1 - q) & (v > 1 - q))) / float(np.mean(u > 1 - q) + 1e-12)

    return {
        "lambda_lower": lower,
        "lambda_upper": upper,
        "q": q,
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# 5. Copula selection — fit all, pick by AIC
# ---------------------------------------------------------------------------

def select_copula(
    series_a: pd.Series,
    series_b: pd.Series,
) -> dict:
    """
    Fit Clayton, Gumbel, Frank copulas and select best by AIC.

    Returns
    -------
    dict with best copula name, all copulas, empirical tail dependence
    """
    df = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    u_df = to_uniform(df)
    u = u_df["a"].values
    v = u_df["b"].values

    # Gaussian copula (full multivariate)
    gc = GaussianCopula()
    gc.fit(u_df.values)
    gc_ll = float(gc.log_density(u_df.values).sum())

    # Archimedean
    clay = ClaytonCopula().fit(u, v)
    gumb = GumbelCopula().fit(u, v)
    frank = FrankCopula().fit(u, v)

    # Empirical tail dependence
    emp_tail = empirical_tail_dependence(u, v)

    logger.info(f"Empirical λ_L={emp_tail['lambda_lower']:.3f}, λ_U={emp_tail['lambda_upper']:.3f}")

    return {
        "gaussian": gc,
        "clayton": clay,
        "gumbel": gumb,
        "frank": frank,
        "empirical_tail": emp_tail,
        "u": u,
        "v": v,
    }