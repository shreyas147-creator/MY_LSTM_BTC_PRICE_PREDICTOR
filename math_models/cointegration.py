"""
cointegration.py — Cointegration & Vector Error Correction Model (VEC/VECM)
Covers: Johansen cointegration test, VECM fitting, long-run equilibrium,
        Engle-Granger two-step, half-life estimation, spread z-score.
Maps to: Cointegration / VEC block (Long-run equilibrium, ECM).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    logger.warning("statsmodels missing. Run: pip install statsmodels")


# ---------------------------------------------------------------------------
# 1. Engle-Granger two-step cointegration test
# ---------------------------------------------------------------------------

def engle_granger_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Tests whether a linear combination of y and x is stationary.
    Returns OLS residuals, ADF p-value, and the hedge ratio β.
    """
    if not STATSMODELS_OK:
        raise ImportError("statsmodels required.")

    # Step 1: OLS  y = α + β x + ε
    x_c = np.column_stack([np.ones(len(x)), x.values])
    beta_ols, _, _, _ = np.linalg.lstsq(x_c, y.values, rcond=None)
    alpha, beta = beta_ols

    residuals = y.values - alpha - beta * x.values
    resid_series = pd.Series(residuals, index=y.index, name="spread")

    # Step 2: ADF on residuals
    adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(residuals, maxlag=1)

    # Cointegration test (statsmodels)
    coint_stat, coint_pval, crit_vals = coint(y, x)

    result = {
        "hedge_ratio": float(beta),
        "intercept": float(alpha),
        "residuals": resid_series,
        "adf_stat": float(adf_stat),
        "adf_pval": float(adf_pval),
        "coint_stat": float(coint_stat),
        "coint_pval": float(coint_pval),
        "coint_crit_1pct": float(crit_vals[0]),
        "is_cointegrated_5pct": adf_pval < 0.05,
    }
    logger.info(f"EG test: β={beta:.4f}, ADF p={adf_pval:.4f}, cointegrated={result['is_cointegrated_5pct']}")
    return result


# ---------------------------------------------------------------------------
# 2. Johansen test — for k>2 series or more rigorous inference
# ---------------------------------------------------------------------------

def johansen_test(
    df: pd.DataFrame,
    det_order: int = 0,   # -1=no constant, 0=constant, 1=constant+trend
    k_ar_diff: int = 1,
) -> dict:
    """
    Johansen cointegration test for a system of price series.

    Parameters
    ----------
    df         : DataFrame with price series as columns
    det_order  : deterministic term order
    k_ar_diff  : lags in differences

    Returns
    -------
    dict with trace statistics, eigenvalues, cointegrating vectors
    """
    if not STATSMODELS_OK:
        raise ImportError("statsmodels required.")

    result = coint_johansen(df.values, det_order, k_ar_diff)

    n_cols = df.shape[1]
    trace_stats = result.lr1          # trace statistics
    crit_90 = result.cvt[:, 0]       # 90% critical values
    crit_95 = result.cvt[:, 1]
    crit_99 = result.cvt[:, 2]

    n_coint = int(np.sum(trace_stats > crit_95))
    logger.info(f"Johansen: {n_coint} cointegrating vector(s) at 5% level")

    return {
        "trace_stats": trace_stats,
        "crit_95": crit_95,
        "crit_99": crit_99,
        "eigenvalues": result.eig,
        "cointegrating_vectors": result.evec,   # columns are CI vectors
        "n_cointegrating_vectors": n_coint,
        "series": list(df.columns),
    }


# ---------------------------------------------------------------------------
# 3. VECM (Vector Error Correction Model)
# ---------------------------------------------------------------------------

class VECModel:
    """
    VECM wrapper: fits a Vector Error Correction Model for cointegrated series.
    Captures short-run dynamics + long-run equilibrium (ECM term).
    """

    def __init__(self, k_ar_diff: int = 1, coint_rank: int = 1, det_order: str = "co"):
        if not STATSMODELS_OK:
            raise ImportError("statsmodels required.")
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.det_order = det_order
        self.model = None
        self.result = None

    def fit(self, df: pd.DataFrame) -> "VECModel":
        """
        Fit VECM to price DataFrame (each column is a series).
        """
        self.model = VECM(
            df,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.det_order,
        )
        self.result = self.model.fit()
        logger.info(f"VECM fitted: rank={self.coint_rank}, k_ar_diff={self.k_ar_diff}")
        logger.info(f"  Alpha (adj speed):\n{self.result.alpha}")
        logger.info(f"  Beta  (CI vector):\n{self.result.beta}")
        return self

    def forecast(self, steps: int = 10) -> pd.DataFrame:
        """Multi-step ahead forecast (levels)."""
        if self.result is None:
            raise RuntimeError("Model not fitted.")
        fc = self.result.predict(steps=steps)
        return pd.DataFrame(fc)

    @property
    def alpha(self) -> np.ndarray:
        """Speed of adjustment coefficients."""
        return self.result.alpha if self.result else None

    @property
    def beta(self) -> np.ndarray:
        """Cointegrating vectors (normalised)."""
        return self.result.beta if self.result else None

    @property
    def error_correction_term(self) -> np.ndarray:
        """Long-run equilibrium deviation (ECM term)."""
        return self.result.resid if self.result else None


# ---------------------------------------------------------------------------
# 4. Spread analysis utilities
# ---------------------------------------------------------------------------

def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: Optional[float] = None,
) -> pd.Series:
    """
    Compute the mean-reverting spread: spread = y - β·x.
    If hedge_ratio is None, estimates β via OLS.
    """
    if hedge_ratio is None:
        x_c = np.column_stack([np.ones(len(x)), x.values])
        coef, _, _, _ = np.linalg.lstsq(x_c, y.values, rcond=None)
        hedge_ratio = coef[1]

    spread = y - hedge_ratio * x
    spread.name = "spread"
    return spread


def spread_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread — used for entry/exit signals."""
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    z = (spread - mu) / sigma
    z.name = "zscore"
    return z


def half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life via OLS on AR(1):
        Δspread_t = λ·spread_{t-1} + ε

    half_life = -ln(2) / λ  (in same units as the time index)
    """
    delta = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    spread_lag, delta = spread_lag.align(delta, join="inner")

    X = np.column_stack([np.ones(len(spread_lag)), spread_lag.values])
    coef, _, _, _ = np.linalg.lstsq(X, delta.values, rcond=None)
    lam = coef[1]

    if lam >= 0:
        logger.warning("AR(1) coefficient ≥ 0 — spread may not be mean-reverting.")
        return np.inf

    hl = -np.log(2) / lam
    logger.info(f"Mean-reversion half-life: {hl:.1f} periods")
    return float(hl)


# ---------------------------------------------------------------------------
# 5. Convenience pipeline: test, fit, signal
# ---------------------------------------------------------------------------

def cointegration_pipeline(
    df: pd.DataFrame,
    signal_col: str,
    hedge_col: str,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> dict:
    """
    Full cointegration pipeline:
      1. EG test
      2. Compute spread + z-score
      3. Generate long/short signals

    Parameters
    ----------
    df          : DataFrame containing signal_col and hedge_col
    signal_col  : name of the target asset column
    hedge_col   : name of the hedge asset column

    Returns
    -------
    dict with test results, spread, z-score, signals
    """
    y = df[signal_col]
    x = df[hedge_col]

    eg = engle_granger_test(y, x)
    spread = compute_spread(y, x, hedge_ratio=eg["hedge_ratio"])
    z = spread_zscore(spread, window=zscore_window)
    hl = half_life(spread)

    signal = pd.Series(0, index=df.index, name="coint_signal")
    signal[z < -entry_z] = 1     # spread low → long signal_col
    signal[z > entry_z] = -1     # spread high → short signal_col
    signal[(z > -exit_z) & (z < exit_z)] = 0  # exit zone

    return {
        "eg_test": eg,
        "spread": spread,
        "zscore": z,
        "half_life": hl,
        "signal": signal,
    }