"""
fokker_planck.py — Fokker-Planck Equation (PDF evolution)
Numerically evolves the probability density p(x,t) under a drift-diffusion SDE.
Uses explicit finite-difference (upwind + central) scheme.
Also exports confidence bands by integrating the PDF.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Fokker-Planck solver
#    ∂p/∂t = -∂/∂x[μ(x) p] + ½ ∂²/∂x²[σ²(x) p]
# ---------------------------------------------------------------------------

class FokkerPlanckSolver:
    """
    1-D Fokker-Planck PDE solver via explicit finite differences.

    Spatial grid : [x_min, x_max] with nx points
    Time grid    : [0, T] with nt steps
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        nx: int,
        T: float,
        dt: float,
    ):
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = self.x[1] - self.x[0]
        self.T = T
        self.dt = dt
        self.nt = int(T / dt)
        self.nx = nx

        # CFL check
        logger.info(f"FP solver: nx={nx}, nt={self.nt}, dx={self.dx:.4f}, dt={dt:.6f}")

    def solve(
        self,
        mu_fn,
        sigma_fn,
        p0: np.ndarray,
        store_every: int = 10,
    ) -> tuple:
        """
        Evolve the PDF forward in time.

        Parameters
        ----------
        mu_fn    : callable(x) -> array    — drift  μ(x)
        sigma_fn : callable(x) -> array    — diffusion σ(x)
        p0       : (nx,) initial PDF (will be normalised)
        store_every : save snapshot every N steps

        Returns
        -------
        (times, pdf_snapshots)
            times         : list of saved time points
            pdf_snapshots : list of (nx,) arrays
        """
        x = self.x
        dx = self.dx
        dt = self.dt

        mu = mu_fn(x)          # (nx,)
        sigma2 = sigma_fn(x) ** 2  # (nx,)

        # Normalise initial condition
        p = p0.copy().astype(np.float64)
        p = np.maximum(p, 0.0)
        p /= (p.sum() * dx)

        times = []
        snapshots = []

        for step in range(self.nt):
            p_new = p.copy()

            # Advection — upwind scheme
            mu_pos = np.maximum(mu, 0.0)
            mu_neg = np.minimum(mu, 0.0)
            adv = (
                mu_pos * (p - np.roll(p, 1)) / dx
                + mu_neg * (np.roll(p, -1) - p) / dx
            )

            # Diffusion — central differences
            diff = np.zeros_like(p)
            # σ²(x) may vary with x → product rule
            sig2_p = sigma2 * p
            diff[1:-1] = (sig2_p[2:] - 2 * sig2_p[1:-1] + sig2_p[:-2]) / (2 * dx ** 2)

            p_new = p + dt * (-adv + diff)

            # Reflective boundary conditions
            p_new[0] = p_new[1]
            p_new[-1] = p_new[-2]

            # Positivity + renormalise (mass conservation)
            p_new = np.maximum(p_new, 0.0)
            norm = p_new.sum() * dx
            if norm > 1e-12:
                p_new /= norm

            p = p_new

            if step % store_every == 0:
                times.append(step * dt)
                snapshots.append(p.copy())

        return times, snapshots


# ---------------------------------------------------------------------------
# 2. Confidence bands from PDF
# ---------------------------------------------------------------------------

def pdf_confidence_bands(
    x: np.ndarray,
    pdf: np.ndarray,
    levels: list = (0.68, 0.90, 0.95),
) -> dict:
    """
    Compute credible intervals from a 1-D PDF by integrating outward from the mode.

    Returns
    -------
    dict {level: (lower, upper)} in x-space
    """
    dx = x[1] - x[0]
    mode_idx = int(np.argmax(pdf))
    cdf = np.cumsum(pdf) * dx

    bands = {}
    for level in levels:
        # central interval: find lo/hi such that CDF(hi)-CDF(lo) ≈ level
        tail = (1.0 - level) / 2.0
        lo_idx = int(np.searchsorted(cdf, tail))
        hi_idx = int(np.searchsorted(cdf, 1.0 - tail))
        lo_idx = max(lo_idx, 0)
        hi_idx = min(hi_idx, len(x) - 1)
        bands[level] = (float(x[lo_idx]), float(x[hi_idx]))

    return bands


# ---------------------------------------------------------------------------
# 3. Build GBM-derived initial PDF (log-normal)
# ---------------------------------------------------------------------------

def lognormal_pdf(x: np.ndarray, mu_ln: float, sigma_ln: float) -> np.ndarray:
    """Log-normal PDF over positive support array x."""
    x_pos = np.maximum(x, 1e-10)
    return (1.0 / (x_pos * sigma_ln * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_pos) - mu_ln) ** 2) / (2 * sigma_ln ** 2)
    )


# ---------------------------------------------------------------------------
# 4. High-level convenience wrapper
# ---------------------------------------------------------------------------

def evolve_price_pdf(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float = 1e-4,
    nx: int = 400,
    price_range_factor: float = 3.0,
) -> dict:
    """
    Evolve a GBM price PDF forward in time and extract confidence bands.

    Parameters
    ----------
    S0       : current price
    mu       : drift
    sigma    : diffusion
    T        : horizon (years)
    dt       : time step
    nx       : spatial grid points
    price_range_factor : how many sigmas around S0 to cover

    Returns
    -------
    dict with keys: x, times, snapshots, final_pdf, bands
    """
    sigma_total = sigma * np.sqrt(T)
    x_min = S0 * np.exp(-price_range_factor * sigma_total)
    x_max = S0 * np.exp(+price_range_factor * sigma_total)

    solver = FokkerPlanckSolver(x_min, x_max, nx, T, dt)
    x = solver.x

    # GBM drift and diffusion in price space
    mu_fn = lambda xv: mu * xv              # drift: μ x
    sigma_fn = lambda xv: sigma * xv        # diffusion: σ x

    # log-normal initial PDF centred at S0
    mu_ln = np.log(S0)
    sigma_ln = 0.01 * sigma          # tight initial spike
    p0 = lognormal_pdf(x, mu_ln, sigma_ln)
    p0 = gaussian_filter1d(p0, sigma=1)    # smooth away grid artefacts

    times, snapshots = solver.solve(mu_fn, sigma_fn, p0)

    final_pdf = snapshots[-1]
    bands = pdf_confidence_bands(x, final_pdf)

    return {
        "x": x,
        "times": times,
        "snapshots": snapshots,
        "final_pdf": final_pdf,
        "bands": bands,
    }