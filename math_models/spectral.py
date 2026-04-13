"""
spectral.py — Spectral Analysis: FFT + Wavelet Cycle Detection
Covers: DFT/FFT power spectrum, dominant cycle extraction, CWT (Morlet),
        band-pass filtering, spectral entropy, Hilbert transform for instantaneous phase.
Maps to: Spectral Analysis block (FFT, wavelet cycle detection).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("PyWavelets not found. Run: pip install PyWavelets")

try:
    from scipy.signal import hilbert, butter, filtfilt
    from scipy.fft import fft, fftfreq, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    from numpy.fft import fft, fftfreq, ifft
    SCIPY_AVAILABLE = False
    logger.warning("scipy not found, using numpy FFT (limited filtering)")


# ---------------------------------------------------------------------------
# 1. FFT Power Spectrum
# ---------------------------------------------------------------------------

def fft_power_spectrum(
    series: np.ndarray,
    sampling_rate: float = 1.0,
    detrend: bool = True,
) -> dict:
    """
    Compute one-sided FFT power spectrum of a time series.

    Parameters
    ----------
    series        : 1-D price or return series
    sampling_rate : samples per unit time (e.g. 1 for daily, 24 for hourly)
    detrend       : remove linear trend before FFT

    Returns
    -------
    dict with freqs, periods, power, dominant_period
    """
    x = np.asarray(series, dtype=np.float64)
    N = len(x)

    if detrend:
        t = np.arange(N)
        coef = np.polyfit(t, x, 1)
        x = x - np.polyval(coef, t)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(N)
    x_windowed = x * window

    # FFT
    X = fft(x_windowed)
    freqs = fftfreq(N, d=1.0 / sampling_rate)

    # One-sided spectrum
    n_one_sided = N // 2
    freqs_pos = freqs[:n_one_sided]
    power = (2.0 / N) * np.abs(X[:n_one_sided]) ** 2   # two-sided → one-sided

    # Avoid DC component
    freqs_pos = freqs_pos[1:]
    power = power[1:]

    periods = 1.0 / freqs_pos
    dominant_idx = int(np.argmax(power))

    result = {
        "freqs": freqs_pos,
        "periods": periods,
        "power": power,
        "dominant_freq": float(freqs_pos[dominant_idx]),
        "dominant_period": float(periods[dominant_idx]),
        "spectral_entropy": float(_spectral_entropy(power)),
    }
    logger.info(f"FFT: dominant period = {result['dominant_period']:.1f} bars")
    return result


def _spectral_entropy(power: np.ndarray) -> float:
    """Normalised spectral entropy — 0=single freq, 1=white noise."""
    p = power / power.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    H_max = np.log2(len(power))
    return float(H / H_max) if H_max > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. Top-N dominant cycles from FFT
# ---------------------------------------------------------------------------

def dominant_cycles(
    series: np.ndarray,
    n_cycles: int = 5,
    sampling_rate: float = 1.0,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
) -> List[dict]:
    """
    Extract the top-N dominant cycles (frequency, period, amplitude, phase).

    Returns
    -------
    List of dicts sorted by amplitude descending.
    """
    x = np.asarray(series, dtype=np.float64)
    N = len(x)

    # Detrend
    t = np.arange(N)
    coef = np.polyfit(t, x, 1)
    x = x - np.polyval(coef, t)

    X = fft(x)
    freqs = fftfreq(N, d=1.0 / sampling_rate)
    n_pos = N // 2

    freqs_pos = freqs[1:n_pos]
    X_pos = X[1:n_pos]
    amplitudes = np.abs(X_pos) * 2 / N
    phases = np.angle(X_pos)
    periods = 1.0 / freqs_pos

    # Filter by period range
    mask = np.ones(len(freqs_pos), dtype=bool)
    if min_period is not None:
        mask &= periods >= min_period
    if max_period is not None:
        mask &= periods <= max_period

    freqs_f = freqs_pos[mask]
    amps_f = amplitudes[mask]
    phases_f = phases[mask]
    periods_f = periods[mask]

    # Top-N by amplitude
    top_idx = np.argsort(amps_f)[::-1][:n_cycles]
    cycles = []
    for i in top_idx:
        cycles.append({
            "freq": float(freqs_f[i]),
            "period": float(periods_f[i]),
            "amplitude": float(amps_f[i]),
            "phase": float(phases_f[i]),
        })

    return cycles


# ---------------------------------------------------------------------------
# 3. Bandpass filter (Butterworth)
# ---------------------------------------------------------------------------

def bandpass_filter(
    series: np.ndarray,
    low_period: float,
    high_period: float,
    sampling_rate: float = 1.0,
    order: int = 4,
) -> np.ndarray:
    """
    Isolate a frequency band via zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    low_period  : lower period bound (e.g. 10 bars)
    high_period : upper period bound (e.g. 40 bars)

    Returns
    -------
    Filtered signal of same length.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for bandpass filter.")

    nyquist = sampling_rate / 2.0
    lo_freq = sampling_rate / high_period   # high period → low frequency
    hi_freq = sampling_rate / low_period

    lo = lo_freq / nyquist
    hi = min(hi_freq / nyquist, 0.99)

    if lo >= hi:
        raise ValueError(f"Invalid band: lo={lo:.4f} >= hi={hi:.4f}")

    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, series)


# ---------------------------------------------------------------------------
# 4. Continuous Wavelet Transform (Morlet) — time-frequency analysis
# ---------------------------------------------------------------------------

def morlet_cwt(
    series: np.ndarray,
    scales: Optional[np.ndarray] = None,
    sampling_rate: float = 1.0,
) -> dict:
    """
    Morlet CWT for time-frequency localisation of cycles.
    Reveals how dominant frequencies evolve over time.

    Returns
    -------
    dict with: scales, periods, time, power (2D), ridge (dominant period per time)
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required: pip install PyWavelets")

    x = np.asarray(series, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)   # standardise

    if scales is None:
        scales = np.arange(2, min(len(x) // 2, 128))

    coeffs, freqs = pywt.cwt(x, scales, "morl", sampling_period=1.0 / sampling_rate)
    power = np.abs(coeffs) ** 2              # (n_scales, T)
    periods = 1.0 / (freqs + 1e-12)

    # Ridge: period with max power at each time step
    ridge_idx = np.argmax(power, axis=0)
    ridge_period = periods[ridge_idx]

    return {
        "scales": scales,
        "periods": periods,
        "time": np.arange(len(x)),
        "power": power,
        "ridge_period": ridge_period,
        "freqs": freqs,
    }


# ---------------------------------------------------------------------------
# 5. Hilbert Transform — instantaneous amplitude, frequency, phase
# ---------------------------------------------------------------------------

def hilbert_analysis(series: np.ndarray) -> dict:
    """
    Hilbert transform: extract analytic signal → instantaneous amplitude & phase.

    Useful for detecting cycle turning points and trend strength.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Hilbert transform.")

    x = np.asarray(series, dtype=np.float64)
    x_detrended = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))

    analytic = hilbert(x_detrended)
    inst_amplitude = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi)

    return {
        "amplitude": inst_amplitude,
        "phase": inst_phase,
        "inst_freq": np.append(inst_freq, inst_freq[-1]),  # pad to same length
        "in_phase": np.real(analytic),
        "quadrature": np.imag(analytic),
    }


# ---------------------------------------------------------------------------
# 6. Spectral feature extraction — for ML feature pipeline
# ---------------------------------------------------------------------------

def spectral_features(
    series: np.ndarray,
    sampling_rate: float = 1.0,
    n_top_cycles: int = 3,
) -> dict:
    """
    Extract a compact spectral feature set for the ML feature matrix.

    Returns
    -------
    Flat dict of scalar features: dominant period, amplitudes, spectral entropy, etc.
    """
    spec = fft_power_spectrum(series, sampling_rate)
    cycles = dominant_cycles(series, n_cycles=n_top_cycles, sampling_rate=sampling_rate)

    features = {
        "spectral_entropy": spec["spectral_entropy"],
        "dominant_period": spec["dominant_period"],
    }
    for i, c in enumerate(cycles):
        features[f"cycle_{i+1}_period"] = c["period"]
        features[f"cycle_{i+1}_amplitude"] = c["amplitude"]

    if SCIPY_AVAILABLE:
        hil = hilbert_analysis(series)
        features["mean_inst_amplitude"] = float(hil["amplitude"].mean())
        features["std_inst_freq"] = float(hil["inst_freq"].std())

    return features