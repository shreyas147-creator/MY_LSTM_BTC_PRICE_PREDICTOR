"""
evt.py — Extreme Value Theory (EVT)
Covers: Peaks-Over-Threshold (POT) with GPD, GEV block maxima,
        tail VaR beyond 99%, Expected Shortfall, Hill estimator.
Maps to: Extreme Value Theory block (GPD, GEV, VaR beyond 99%).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Peaks-Over-Threshold (POT) — GPD fitting
# ---------------------------------------------------------------------------

class GPDTailModel:
    """
    Generalised Pareto Distribution fitted to exceedances over a threshold u.

    P(X - u > y | X > u) ~ GPD(ξ, β)
      ξ > 0  → heavy tail (Pareto-like)
      ξ = 0  → exponential tail
      ξ < 0  → bounded tail

    Parameters fitted via MLE.
    """

    def __init__(self, threshold: Optional[float] = None, threshold_quantile: float = 0.95):
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
        self.xi = None      # shape (tail index)
        self.beta = None    # scale
        self.n_total = None
        self.n_exceed = None
        self.exceedances = None

    def fit(self, losses: np.ndarray) -> "GPDTailModel":
        """
        Fit GPD to the tail of a loss distribution.

        Parameters
        ----------
        losses : 1-D array of losses (positive = loss).
                 Pass negative returns * -1 for left tail.
        """
        x = np.asarray(losses, dtype=np.float64)
        self.n_total = len(x)

        if self.threshold is None:
            self.threshold = float(np.quantile(x, self.threshold_quantile))

        exceedances = x[x > self.threshold] - self.threshold
        self.exceedances = exceedances
        self.n_exceed = len(exceedances)

        if self.n_exceed < 10:
            logger.warning(f"Only {self.n_exceed} exceedances — threshold may be too high.")

        # MLE: fit GPD
        xi0, beta0 = 0.1, float(exceedances.mean())

        def neg_log_lik(params):
            xi, beta = params
            if beta <= 0:
                return 1e10
            if xi != 0:
                arg = 1 + xi * exceedances / beta
                if np.any(arg <= 0):
                    return 1e10
                return self.n_exceed * np.log(beta) + (1 + 1 / xi) * np.sum(np.log(arg))
            else:
                return self.n_exceed * np.log(beta) + np.sum(exceedances) / beta

        result = minimize(neg_log_lik, [xi0, beta0], method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        self.xi = float(result.x[0])
        self.beta = float(result.x[1])

        logger.info(
            f"GPD fitted: u={self.threshold:.4f}, ξ={self.xi:.4f}, β={self.beta:.4f}, "
            f"n_exceed={self.n_exceed}/{self.n_total}"
        )
        return self

    def tail_probability(self, x: float) -> float:
        """P(X > x) for x > threshold, using semi-parametric formula."""
        if self.xi is None:
            raise RuntimeError("Model not fitted.")
        pu = self.n_exceed / self.n_total       # P(X > u)
        excess = x - self.threshold
        if excess <= 0:
            return 1.0
        if self.xi == 0:
            return pu * np.exp(-excess / self.beta)
        arg = 1 + self.xi * excess / self.beta
        if arg <= 0:
            return 0.0
        return pu * arg ** (-1 / self.xi)

    def var(self, alpha: float) -> float:
        """
        Value at Risk at confidence level α (e.g. 0.999 = 99.9%).
        Returns the loss quantile.
        """
        if self.xi is None:
            raise RuntimeError("Model not fitted.")
        pu = self.n_exceed / self.n_total
        if self.xi == 0:
            return self.threshold + self.beta * np.log(pu / (1 - alpha))
        return self.threshold + (self.beta / self.xi) * ((pu / (1 - alpha)) ** self.xi - 1)

    def es(self, alpha: float) -> float:
        """
        Expected Shortfall (CVaR) at confidence level α.
        ES = VaR/(1-ξ) + (β - ξ·u)/(1-ξ)   for ξ < 1
        """
        if self.xi is None:
            raise RuntimeError("Model not fitted.")
        if self.xi >= 1:
            return np.inf
        var = self.var(alpha)
        return var / (1 - self.xi) + (self.beta - self.xi * self.threshold) / (1 - self.xi)

    def var_table(self, levels: list = (0.95, 0.99, 0.999, 0.9999)) -> pd.DataFrame:
        """Return VaR and ES for multiple confidence levels."""
        rows = []
        for alpha in levels:
            v = self.var(alpha)
            e = self.es(alpha)
            rows.append({"confidence": alpha, "VaR": v, "ES_CVaR": e})
        df = pd.DataFrame(rows)
        logger.info(f"\n{df.to_string(index=False)}")
        return df


# ---------------------------------------------------------------------------
# 2. GEV — Block Maxima approach (quarterly/monthly worst losses)
# ---------------------------------------------------------------------------

class GEVModel:
    """
    Generalised Extreme Value distribution fitted to block maxima.

    H(z; μ, σ, ξ) = exp(-(1 + ξ(z-μ)/σ)^{-1/ξ})
      ξ > 0 → Fréchet (heavy tail)
      ξ = 0 → Gumbel
      ξ < 0 → Weibull (bounded)
    """

    def __init__(self, block_size: int = 63):
        """block_size: e.g. 63 trading days ≈ 1 quarter"""
        self.block_size = block_size
        self.xi = None
        self.mu = None
        self.sigma = None
        self._rv = None

    def fit(self, losses: np.ndarray) -> "GEVModel":
        """
        Extract block maxima and fit GEV via scipy.
        """
        x = np.asarray(losses, dtype=np.float64)
        n_blocks = len(x) // self.block_size
        blocks = x[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        maxima = blocks.max(axis=1)

        # Scipy fit (MLE)
        self.xi, self.mu, self.sigma = stats.genextreme.fit(maxima)
        # scipy uses -ξ convention
        self._rv = stats.genextreme(self.xi, self.mu, self.sigma)

        logger.info(
            f"GEV fitted: μ={self.mu:.4f}, σ={self.sigma:.4f}, ξ={self.xi:.4f} "
            f"({n_blocks} blocks)"
        )
        return self

    def var(self, alpha: float) -> float:
        """Return period VaR — quantile of the block maxima distribution."""
        return float(self._rv.ppf(alpha))

    def return_level(self, return_period: int) -> float:
        """
        Return level for a given return period (in blocks).
        E.g. return_period=40 quarters → 10-year worst quarterly loss.
        """
        alpha = 1 - 1 / return_period
        return self.var(alpha)

    @property
    def tail_index(self) -> float:
        """Tail index (1/ξ for Fréchet; ξ > 0 means heavy tail)."""
        return float(1 / self.xi) if self.xi and self.xi > 0 else np.inf


# ---------------------------------------------------------------------------
# 3. Hill estimator — non-parametric tail index
# ---------------------------------------------------------------------------

def hill_estimator(
    losses: np.ndarray,
    k_range: Optional[range] = None,
) -> pd.DataFrame:
    """
    Hill estimator for the tail index α = 1/ξ.
    Computed for a range of k (number of order statistics used).

    Returns
    -------
    DataFrame with k, xi_hat (shape), alpha_hat (tail index)
    """
    x = np.sort(np.asarray(losses, dtype=np.float64))[::-1]   # descending
    n = len(x)

    if k_range is None:
        k_range = range(10, min(n // 2, 200))

    records = []
    for k in k_range:
        log_ratios = np.log(x[:k]) - np.log(x[k])
        xi_hat = float(np.mean(log_ratios))
        alpha_hat = 1.0 / xi_hat if xi_hat > 0 else np.inf
        records.append({"k": k, "xi_hat": xi_hat, "alpha_hat": alpha_hat})

    df = pd.DataFrame(records)
    stable_alpha = float(df.loc[df["k"].between(50, 100), "alpha_hat"].mean())
    logger.info(f"Hill estimator: stable α ≈ {stable_alpha:.2f} (heavier tail = smaller α)")
    return df


# ---------------------------------------------------------------------------
# 4. Mean Excess Function — threshold selection diagnostic
# ---------------------------------------------------------------------------

def mean_excess_function(
    losses: np.ndarray,
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """
    Mean Excess Function (MEF): E[X - u | X > u] vs u.
    If MEF is linear and upward-sloping → GPD with ξ > 0 (heavy tail).
    Useful for threshold selection.
    """
    x = np.sort(np.asarray(losses, dtype=np.float64))
    thresholds = np.linspace(np.percentile(x, 50), np.percentile(x, 98), n_thresholds)

    records = []
    for u in thresholds:
        exceed = x[x > u] - u
        if len(exceed) < 5:
            break
        records.append({"threshold": u, "mean_excess": float(exceed.mean()), "n_exceed": len(exceed)})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Full EVT pipeline for a return series
# ---------------------------------------------------------------------------

def evt_pipeline(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
    var_levels: list = (0.99, 0.999, 0.9999),
    block_size: int = 63,
) -> dict:
    """
    Run POT + GEV + Hill estimator on a return series.

    Parameters
    ----------
    returns : pd.Series of log-returns (daily)
    """
    losses = -returns.dropna().values    # losses = negative returns

    # POT
    gpd = GPDTailModel(threshold_quantile=threshold_quantile).fit(losses)
    var_table = gpd.var_table(levels=list(var_levels))

    # GEV
    gev = GEVModel(block_size=block_size).fit(losses)

    # Hill
    hill = hill_estimator(losses)

    # MEF
    mef = mean_excess_function(losses)

    result = {
        "gpd": gpd,
        "gev": gev,
        "var_table": var_table,
        "hill": hill,
        "mef": mef,
        "summary": {
            "gpd_xi": gpd.xi,
            "gpd_beta": gpd.beta,
            "gev_xi": gev.xi,
            "tail_index_hill": float(hill["alpha_hat"].median()),
            "var_99": gpd.var(0.99),
            "var_999": gpd.var(0.999),
            "es_99": gpd.es(0.99),
        },
    }
    logger.info(f"EVT pipeline complete. VaR(99.9%)={result['summary']['var_999']:.4f}")
    return result