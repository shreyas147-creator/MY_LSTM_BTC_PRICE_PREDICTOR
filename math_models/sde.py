"""
sde.py — Stochastic Differential Equations
Covers: Geometric Brownian Motion, Itô SDE, Merton Jump-Diffusion, Lévy processes
GPU-accelerated via CuPy with NumPy fallback.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy available — SDE running on GPU")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    logger.warning("CuPy not found — SDE falling back to NumPy (CPU)")

xp = cp if GPU_AVAILABLE else np


# ---------------------------------------------------------------------------
# 1. Geometric Brownian Motion  (classic Black-Scholes SDE)
#    dS = μ S dt + σ S dW_t
# ---------------------------------------------------------------------------

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths using the exact log-normal solution:
        S(t) = S0 * exp((μ - σ²/2)t + σ W_t)

    Returns
    -------
    np.ndarray of shape (n_steps+1, n_paths)
    """
    rng = xp.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = xp.sqrt(xp.array(dt))

    # Brownian increments
    dW = rng.standard_normal((n_steps, n_paths)) * sqrt_dt          # (steps, paths)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * dW         # Itô correction

    log_S = xp.concatenate(
        [xp.full((1, n_paths), xp.log(xp.array(S0))),
         xp.cumsum(log_returns, axis=0)],
        axis=0,
    )
    paths = xp.exp(log_S)

    if GPU_AVAILABLE:
        return cp.asnumpy(paths)
    return paths


# ---------------------------------------------------------------------------
# 2. General Itô SDE — Euler-Maruyama scheme
#    dX = a(X,t) dt + b(X,t) dW_t
# ---------------------------------------------------------------------------

def euler_maruyama(
    drift_fn,
    diffusion_fn,
    X0: float,
    T: float,
    dt: float,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generic Euler-Maruyama integrator for a user-supplied Itô SDE.

    Parameters
    ----------
    drift_fn     : callable(X, t) -> array   — drift  a(X,t)
    diffusion_fn : callable(X, t) -> array   — diffusion b(X,t)

    Returns
    -------
    np.ndarray of shape (n_steps+1, n_paths)
    """
    rng = xp.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = xp.sqrt(xp.array(dt))

    X = xp.full((n_paths,), X0, dtype=xp.float64)
    paths = [X.copy()]

    for i in range(n_steps):
        t = i * dt
        dW = rng.standard_normal(n_paths) * sqrt_dt
        X = X + drift_fn(X, t) * dt + diffusion_fn(X, t) * dW
        paths.append(X.copy())

    out = xp.stack(paths, axis=0)   # (n_steps+1, n_paths)
    if GPU_AVAILABLE:
        return cp.asnumpy(out)
    return out


# ---------------------------------------------------------------------------
# 3. Merton Jump-Diffusion
#    dS = (μ - λk̄) S dt + σ S dW_t + S dJ_t
#    J_t = compound Poisson with log-normal jump sizes
# ---------------------------------------------------------------------------

def simulate_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,          # jump intensity (expected jumps per year)
    mu_j: float,         # mean log-jump size
    sigma_j: float,      # std  log-jump size
    T: float,
    dt: float,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Merton (1976) jump-diffusion.  Fat tails via compound Poisson jumps.

    Returns
    -------
    np.ndarray of shape (n_steps+1, n_paths)
    """
    rng = xp.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = float(xp.sqrt(xp.array(dt)))

    # compensator so drift is still μ
    k_bar = float(xp.exp(xp.array(mu_j + 0.5 * sigma_j ** 2))) - 1.0
    drift_adj = mu - lam * k_bar - 0.5 * sigma ** 2

    log_S = xp.full((n_paths,), float(xp.log(xp.array(S0))), dtype=xp.float64)
    paths = [log_S.copy()]

    for _ in range(n_steps):
        # diffusion part
        dW = rng.standard_normal(n_paths) * sqrt_dt
        # Poisson jump counts
        N = rng.poisson(lam * dt, n_paths)
        # log-jump sizes (sum of N_i log-normal jumps)
        jump_log = xp.array([
            float(xp.sum(rng.normal(mu_j, sigma_j, int(n)))) if n > 0 else 0.0
            for n in (cp.asnumpy(N) if GPU_AVAILABLE else N)
        ])
        log_S = log_S + drift_adj * dt + sigma * dW + jump_log
        paths.append(log_S.copy())

    out = xp.exp(xp.stack(paths, axis=0))
    if GPU_AVAILABLE:
        return cp.asnumpy(out)
    return out


# ---------------------------------------------------------------------------
# 4. Heston Stochastic Volatility (bonus — used by vol models)
#    dS = μ S dt + √v S dW_S
#    dv = κ(θ - v) dt + ξ √v dW_v     corr(dW_S, dW_v) = ρ
# ---------------------------------------------------------------------------

def simulate_heston(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,   # mean-reversion speed
    theta: float,   # long-run variance
    xi: float,      # vol-of-vol
    rho: float,     # correlation
    T: float,
    dt: float,
    n_paths: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Full-truncation Euler scheme for the Heston model.

    Returns
    -------
    (S_paths, v_paths) each of shape (n_steps+1, n_paths)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    S = np.full(n_paths, S0, dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)

    S_paths = [S.copy()]
    v_paths = [v.copy()]

    chol = np.array([[1.0, 0.0],
                     [rho, np.sqrt(max(1 - rho ** 2, 1e-12))]])

    for _ in range(n_steps):
        Z = rng.standard_normal((2, n_paths))
        dW = chol @ Z * sqrt_dt          # correlated Brownians

        v_pos = np.maximum(v, 0.0)       # full truncation
        S = S * np.exp((mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW[0])
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW[1]

        S_paths.append(S.copy())
        v_paths.append(v.copy())

    return np.stack(S_paths), np.stack(v_paths)


# ---------------------------------------------------------------------------
# 5. Summary statistics across paths
# ---------------------------------------------------------------------------

def path_statistics(paths: np.ndarray) -> dict:
    """
    Compute mean, std, VaR 5%, CVaR 5% across terminal values.

    Parameters
    ----------
    paths : (n_steps+1, n_paths)
    """
    terminal = paths[-1]
    var_5 = float(np.percentile(terminal, 5))
    cvar_5 = float(terminal[terminal <= var_5].mean())
    return {
        "mean": float(terminal.mean()),
        "std": float(terminal.std()),
        "var_5pct": var_5,
        "cvar_5pct": cvar_5,
        "min": float(terminal.min()),
        "max": float(terminal.max()),
    }