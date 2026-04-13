"""
bayesian.py — Bayesian Inference via PyMC MCMC + Variational Bayes
Covers: posterior estimation of GBM/GARCH parameters, regime priors,
        variational inference for fast approximate posteriors.
Maps to: Bayesian Inference block (MCMC, variational Bayes).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pymc as pm
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.warning("PyMC not found. Run: pip install pymc")

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Bayesian GBM parameter estimation  (μ, σ from log-returns)
# ---------------------------------------------------------------------------

def bayesian_gbm_params(
    log_returns: np.ndarray,
    dt: float = 1 / 252,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
) -> dict:
    """
    Estimate μ (drift) and σ (diffusion) of a GBM from observed log-returns.

    Model:
        r_t ~ Normal(μ*dt, σ²*dt)
        μ ~ Normal(0, 0.5)       [diffuse, annualised]
        σ ~ HalfNormal(0.5)      [positive]

    Returns
    -------
    dict with posterior samples, summary stats, and HPDI intervals.
    """
    if not PYMC_AVAILABLE:
        raise ImportError("Install PyMC: pip install pymc")

    r = np.asarray(log_returns, dtype=np.float64)

    with pm.Model() as gbm_model:
        # Priors
        mu_ann = pm.Normal("mu_ann", mu=0.0, sigma=0.5)       # annualised drift
        sigma_ann = pm.HalfNormal("sigma_ann", sigma=0.5)      # annualised vol

        # Daily parameters
        mu_daily = mu_ann * dt
        sigma_daily = sigma_ann * np.sqrt(dt)

        # Likelihood
        obs = pm.Normal("obs", mu=mu_daily, sigma=sigma_daily, observed=r)

        # Inference
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=False,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["mu_ann", "sigma_ann"]) if ARVIZ_AVAILABLE else {}
    hpdi = {}
    for var in ["mu_ann", "sigma_ann"]:
        samples = trace.posterior[var].values.flatten()
        hpdi[var] = {
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "hdi_3%": float(np.percentile(samples, 3)),
            "hdi_97%": float(np.percentile(samples, 97)),
        }
        logger.info(f"  {var}: mean={hpdi[var]['mean']:.4f}, std={hpdi[var]['std']:.4f}")

    return {"trace": trace, "hpdi": hpdi, "summary": summary}


# ---------------------------------------------------------------------------
# 2. Bayesian GARCH(1,1) — joint posterior of ω, α, β
# ---------------------------------------------------------------------------

def bayesian_garch(
    returns: np.ndarray,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
) -> dict:
    """
    Bayesian GARCH(1,1) estimation via MCMC.
    h_t = ω + α ε²_{t-1} + β h_{t-1}

    Priors:
        ω ~ HalfNormal(0.01)
        α ~ Beta(1, 5)          — small persistence
        β ~ Beta(5, 1)          — high persistence
        ν ~ Exponential(0.1)+2  — DoF for Student-t errors
    """
    if not PYMC_AVAILABLE:
        raise ImportError("Install PyMC: pip install pymc")

    r = np.asarray(returns, dtype=np.float64)
    T = len(r)

    with pm.Model() as garch_model:
        omega = pm.HalfNormal("omega", sigma=0.01)
        alpha = pm.Beta("alpha", alpha=1, beta=5)
        beta = pm.Beta("beta", alpha=5, beta=1)
        nu = pm.Deterministic("nu", 2 + pm.Exponential("nu_raw", lam=0.1))

        # GARCH variance recursion (pytensor scan)
        h0 = pm.HalfNormal("h0", sigma=0.1)

        def garch_step(r_t, h_prev, omega, alpha, beta):
            h_t = omega + alpha * r_t ** 2 + beta * h_prev
            return h_t

        h_seq, _ = pt.scan(
            fn=garch_step,
            sequences=r[:-1],
            outputs_info=h0,
            non_sequences=[omega, alpha, beta],
        )
        h = pt.concatenate([[h0], h_seq])

        # Student-t likelihood
        obs = pm.StudentT("obs", nu=nu, mu=0, sigma=pt.sqrt(h), observed=r)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.85,
            progressbar=False,
            return_inferencedata=True,
        )

    result = {}
    for var in ["omega", "alpha", "beta", "nu"]:
        s = trace.posterior[var].values.flatten()
        result[var] = {"mean": float(s.mean()), "std": float(s.std())}
        logger.info(f"  GARCH {var}: {result[var]}")

    return {"trace": trace, "params": result}


# ---------------------------------------------------------------------------
# 3. Variational Bayes (ADVI) — fast approximate posterior
# ---------------------------------------------------------------------------

def variational_gbm_params(
    log_returns: np.ndarray,
    dt: float = 1 / 252,
    n_iterations: int = 50_000,
) -> dict:
    """
    Mean-field ADVI approximation to GBM parameter posterior.
    Much faster than MCMC — suitable for live re-estimation.

    Returns
    -------
    dict with approximate mean and std of μ_ann, σ_ann
    """
    if not PYMC_AVAILABLE:
        raise ImportError("Install PyMC: pip install pymc")

    r = np.asarray(log_returns, dtype=np.float64)

    with pm.Model() as model:
        mu_ann = pm.Normal("mu_ann", mu=0.0, sigma=0.5)
        sigma_ann = pm.HalfNormal("sigma_ann", sigma=0.5)
        obs = pm.Normal(
            "obs",
            mu=mu_ann * dt,
            sigma=sigma_ann * np.sqrt(dt),
            observed=r,
        )
        approx = pm.fit(n=n_iterations, method="advi", progressbar=False)

    means = approx.bij.rval(approx.mean.eval())
    stds = approx.bij.rval(approx.std.eval())

    result = {}
    for k in ["mu_ann", "sigma_ann"]:
        result[k] = {
            "approx_mean": float(means[k]),
            "approx_std": float(stds[k]),
        }
    return result


# ---------------------------------------------------------------------------
# 4. Bayesian updating — online posterior via conjugate Normal-Normal
# ---------------------------------------------------------------------------

class BayesianOnlineEstimator:
    """
    Conjugate Normal-Normal online estimator for the mean of log-returns.
    Updates incrementally as new data arrives (no MCMC needed).

    Prior: μ ~ Normal(mu_0, tau_0^2)
    Likelihood: r_t ~ Normal(μ, sigma^2)  [sigma known]
    """

    def __init__(self, mu_0: float = 0.0, tau_0: float = 0.5, sigma: float = 0.02):
        self.mu_n = mu_0         # posterior mean
        self.tau_n2 = tau_0 ** 2  # posterior variance
        self.sigma2 = sigma ** 2
        self.n = 0

    def update(self, r: float) -> dict:
        """Update posterior with a single new observation."""
        denom = self.tau_n2 + self.sigma2
        self.mu_n = (self.sigma2 * self.mu_n + self.tau_n2 * r) / denom
        self.tau_n2 = (self.tau_n2 * self.sigma2) / denom
        self.n += 1
        return self.posterior

    def update_batch(self, returns: np.ndarray) -> dict:
        for r in returns:
            self.update(float(r))
        return self.posterior

    @property
    def posterior(self) -> dict:
        return {
            "mu_n": self.mu_n,
            "tau_n": np.sqrt(self.tau_n2),
            "n_obs": self.n,
            "credible_95": (
                self.mu_n - 1.96 * np.sqrt(self.tau_n2),
                self.mu_n + 1.96 * np.sqrt(self.tau_n2),
            ),
        }