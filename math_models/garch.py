"""
garch.py — GARCH / EGARCH / GJR-GARCH Volatility Modelling
Wraps the `arch` library for fitting and forecasting.
Maps to: Volatility Modeling block (GARCH, Heston, SABR) in the diagram.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not installed. Run: pip install arch")


# ---------------------------------------------------------------------------
# 1. GARCH(p,q) — standard conditional heteroskedasticity
# ---------------------------------------------------------------------------

class GARCHModel:
    """
    Wrapper around arch.arch_model for GARCH/EGARCH/GJR variants.

    Parameters
    ----------
    vol   : 'Garch' | 'EGarch' | 'Gjr-Garch'
    p, q  : lag orders for variance equation
    dist  : 'normal' | 't' | 'skewt'
    """

    def __init__(
        self,
        vol: str = "Garch",
        p: int = 1,
        q: int = 1,
        dist: str = "t",
        mean: str = "AR",
        lags: int = 1,
    ):
        if not ARCH_AVAILABLE:
            raise ImportError("Install arch: pip install arch")
        self.vol = vol
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.lags = lags
        self.result = None
        self.model = None

    def fit(self, returns: pd.Series, verbose: bool = False) -> "GARCHModel":
        """
        Fit the GARCH model to a returns series (in percent, e.g. 100*log-returns).

        Parameters
        ----------
        returns : pd.Series of log-returns (already multiplied by 100 recommended)
        """
        self.model = arch_model(
            returns,
            mean=self.mean,
            lags=self.lags,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
        )
        disp = "off" if not verbose else "final"
        self.result = self.model.fit(disp=disp, show_warning=False)
        logger.info(f"{self.vol}({self.p},{self.q}) fitted. AIC={self.result.aic:.2f}")
        return self

    def forecast_variance(self, horizon: int = 10) -> np.ndarray:
        """
        Multi-step variance forecast (annualised vol).

        Returns
        -------
        np.ndarray of shape (horizon,) — annualised volatility
        """
        if self.result is None:
            raise RuntimeError("Model not fitted yet.")
        fc = self.result.forecast(horizon=horizon, reindex=False)
        # variance is in (pct return)^2 space → convert to annualised vol
        var_forecast = fc.variance.values[-1]          # (horizon,)
        ann_vol = np.sqrt(var_forecast * 252) / 100.0  # assuming daily, pct input
        return ann_vol

    def conditional_volatility(self) -> np.ndarray:
        """Return in-sample conditional volatility series."""
        if self.result is None:
            raise RuntimeError("Model not fitted yet.")
        return self.result.conditional_volatility.values / 100.0   # back to decimal

    def summary(self) -> str:
        if self.result is None:
            return "Not fitted."
        return str(self.result.summary())

    @property
    def params(self) -> dict:
        if self.result is None:
            return {}
        return dict(self.result.params)


# ---------------------------------------------------------------------------
# 2. Realised volatility (non-parametric) — for comparison / features
# ---------------------------------------------------------------------------

def realised_volatility(returns: pd.Series, window: int = 20, annualise: bool = True) -> pd.Series:
    """
    Rolling realised volatility.

    Parameters
    ----------
    returns : daily log-returns (decimal)
    window  : rolling window in days
    """
    rv = returns.rolling(window).std()
    if annualise:
        rv = rv * np.sqrt(252)
    rv.name = f"rv_{window}d"
    return rv


# ---------------------------------------------------------------------------
# 3. SABR-inspired vol surface approximation (Hagan formula)
# ---------------------------------------------------------------------------

def sabr_implied_vol(
    F: float,        # forward price
    K: float,        # strike
    T: float,        # expiry (years)
    alpha: float,    # initial vol
    beta: float,     # CEV exponent (0=normal, 1=lognormal)
    rho: float,      # correlation
    nu: float,       # vol-of-vol
) -> float:
    """
    Hagan et al. (2002) SABR implied volatility approximation.
    Valid for F ≠ K (ATM formula is a limit; handled separately).
    """
    if abs(F - K) < 1e-8:
        # ATM formula
        FK_mid = F
        term1 = alpha / (FK_mid ** (1 - beta))
        term2 = 1 + (
            ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_mid ** (2 - 2 * beta)
            + rho * beta * nu * alpha / (4 * FK_mid ** (1 - beta))
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        ) * T
        return term1 * term2

    log_FK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)

    # z and x(z)
    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    numer = alpha
    denom_1 = FK_beta * (
        1
        + (1 - beta) ** 2 / 24 * log_FK ** 2
        + (1 - beta) ** 4 / 1920 * log_FK ** 4
    )
    correction = (
        (1 - beta) ** 2 / 24 * alpha ** 2 / FK_beta ** 2
        + rho * beta * nu * alpha / (4 * FK_beta)
        + (2 - 3 * rho ** 2) / 24 * nu ** 2
    )

    iv = (numer / denom_1) * (z / x_z) * (1 + correction * T)
    return float(iv)


def sabr_vol_surface(
    F: float,
    strikes: np.ndarray,
    T: float,
    alpha: float = 0.2,
    beta: float = 0.5,
    rho: float = -0.3,
    nu: float = 0.4,
) -> np.ndarray:
    """Compute SABR vol for an array of strikes."""
    return np.array([sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])


# ---------------------------------------------------------------------------
# 4. Convenience: fit best GARCH variant by AIC
# ---------------------------------------------------------------------------

def fit_best_garch(
    returns: pd.Series,
    candidates: Optional[list] = None,
) -> GARCHModel:
    """
    Try multiple GARCH specs and return the one with lowest AIC.

    Parameters
    ----------
    candidates : list of (vol, p, q, dist) tuples
    """
    if candidates is None:
        candidates = [
            ("Garch", 1, 1, "normal"),
            ("Garch", 1, 1, "t"),
            ("Garch", 2, 1, "t"),
            ("EGarch", 1, 1, "t"),
            ("Gjr-Garch", 1, 1, "t"),
        ]

    best_aic = np.inf
    best_model = None

    for (vol, p, q, dist) in candidates:
        try:
            m = GARCHModel(vol=vol, p=p, q=q, dist=dist)
            m.fit(returns)
            aic = m.result.aic
            logger.info(f"  {vol}({p},{q},{dist}) AIC={aic:.2f}")
            if aic < best_aic:
                best_aic = aic
                best_model = m
        except Exception as e:
            logger.debug(f"  {vol}({p},{q},{dist}) failed: {e}")

    logger.info(f"Best model: {best_model.vol}({best_model.p},{best_model.q}) AIC={best_aic:.2f}")
    return best_model