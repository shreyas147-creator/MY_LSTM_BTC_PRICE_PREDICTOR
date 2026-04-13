"""
tda.py — Topological Data Analysis (TDA)
Covers: Persistent homology (H0, H1), Vietoris-Rips complex,
        persistence diagrams, Betti numbers, topological feature extraction,
        sliding-window embeddings for time-series TDA.
Maps to: Topological Data Analysis block (persistent homology, TDA).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    from ripser import ripser
    from persim import plot_diagrams, wasserstein, bottleneck
    RIPSER_OK = True
except ImportError:
    RIPSER_OK = False
    logger.warning("ripser/persim not found. Run: pip install ripser persim")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ---------------------------------------------------------------------------
# 1. Sliding-window embedding (Takens' theorem)
# ---------------------------------------------------------------------------

def sliding_window_embedding(
    series: np.ndarray,
    window: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Embed a 1-D time series as a point cloud via sliding window (Takens).
    Each point = [x_t, x_{t+1}, ..., x_{t+window-1}]

    Parameters
    ----------
    series : 1-D array
    window : embedding dimension (window size)
    stride : step between windows

    Returns
    -------
    (n_windows, window) point cloud
    """
    n = len(series)
    indices = np.arange(0, n - window + 1, stride)
    cloud = np.array([series[i:i + window] for i in indices])
    return cloud


# ---------------------------------------------------------------------------
# 2. Compute persistent homology via Ripser
# ---------------------------------------------------------------------------

def compute_persistence(
    point_cloud: np.ndarray,
    max_dim: int = 1,
    max_edge_length: float = np.inf,
    metric: str = "euclidean",
) -> dict:
    """
    Compute Vietoris-Rips persistent homology.

    Parameters
    ----------
    point_cloud    : (n_points, n_dims) array
    max_dim        : max homology dimension (1 = loops, 2 = voids)
    max_edge_length: threshold for Rips complex

    Returns
    -------
    dict with 'dgms' (list of persistence diagrams per dim), 'H0', 'H1', etc.
    """
    if not RIPSER_OK:
        raise ImportError("Install ripser: pip install ripser persim")

    if SKLEARN_OK:
        scaler = StandardScaler()
        X = scaler.fit_transform(point_cloud)
    else:
        X = point_cloud

    result = ripser(X, maxdim=max_dim, thresh=max_edge_length, metric=metric)
    dgms = result["dgms"]       # list: dgms[0]=H0, dgms[1]=H1, …

    out = {"dgms": dgms}
    for dim, dgm in enumerate(dgms):
        # Remove inf bars from H0 (connected component that never dies)
        finite = dgm[~np.isinf(dgm[:, 1])]
        out[f"H{dim}"] = finite
        lifetimes = finite[:, 1] - finite[:, 0]
        out[f"H{dim}_lifetimes"] = lifetimes
        logger.info(f"  H{dim}: {len(finite)} finite bars, max lifetime={lifetimes.max() if len(lifetimes) else 0:.4f}")

    return out


# ---------------------------------------------------------------------------
# 3. Persistence statistics / features
# ---------------------------------------------------------------------------

def persistence_features(
    dgms: List[np.ndarray],
    dims: List[int] = (0, 1),
) -> dict:
    """
    Extract scalar topological features from persistence diagrams.

    Features per dimension:
    - total_persistence  = Σ (death - birth)
    - max_lifetime       = max(death - birth)
    - n_bars             = number of bars
    - entropy            = persistence entropy
    - betti              = number of features alive at midlife

    Returns
    -------
    Flat dict of features suitable for ML feature matrix.
    """
    features = {}

    for dim in dims:
        if dim >= len(dgms):
            continue
        dgm = dgms[dim]
        finite = dgm[~np.isinf(dgm[:, 1])] if len(dgm) > 0 else np.zeros((0, 2))

        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([0.0])
        total = float(lifetimes.sum())
        max_life = float(lifetimes.max()) if len(lifetimes) > 0 else 0.0

        # Persistent entropy
        if total > 0:
            p = lifetimes / total
            p = p[p > 0]
            H = -float(np.sum(p * np.log(p)))
        else:
            H = 0.0

        # Midpoint Betti number (features alive at midpoint of their lifespan)
        midpoints = (finite[:, 0] + finite[:, 1]) / 2 if len(finite) > 0 else []
        betti = len(midpoints)

        features[f"H{dim}_n_bars"] = len(finite)
        features[f"H{dim}_total_persistence"] = total
        features[f"H{dim}_max_lifetime"] = max_life
        features[f"H{dim}_entropy"] = H
        features[f"H{dim}_betti"] = betti

    return features


# ---------------------------------------------------------------------------
# 4. Wasserstein / bottleneck distance between diagrams
# ---------------------------------------------------------------------------

def topological_distance(
    dgm_a: np.ndarray,
    dgm_b: np.ndarray,
    metric: str = "wasserstein",
) -> float:
    """
    Distance between two persistence diagrams.

    Parameters
    ----------
    metric : 'wasserstein' | 'bottleneck'
    """
    if not RIPSER_OK:
        raise ImportError("Install persim: pip install persim")

    # Remove infinities
    fa = dgm_a[~np.isinf(dgm_a[:, 1])] if len(dgm_a) > 0 else np.zeros((1, 2))
    fb = dgm_b[~np.isinf(dgm_b[:, 1])] if len(dgm_b) > 0 else np.zeros((1, 2))

    if len(fa) == 0:
        fa = np.zeros((1, 2))
    if len(fb) == 0:
        fb = np.zeros((1, 2))

    if metric == "wasserstein":
        return float(wasserstein(fa, fb))
    elif metric == "bottleneck":
        return float(bottleneck(fa, fb))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# 5. Rolling TDA features — for ML feature matrix
# ---------------------------------------------------------------------------

def rolling_tda_features(
    prices: pd.Series,
    embed_window: int = 20,
    embed_stride: int = 1,
    tda_stride: int = 5,
    max_dim: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling topological features over a price series.

    Uses an outer rolling window over the embedded point cloud.

    Parameters
    ----------
    prices       : pd.Series of prices
    embed_window : sliding window dimension for Takens embedding
    embed_stride : stride for embedding
    tda_stride   : how often to compute TDA (every N steps — expensive)
    max_dim      : max homology dimension

    Returns
    -------
    pd.DataFrame with topological features, indexed to price timestamps.
    """
    if not RIPSER_OK:
        raise ImportError("Install ripser: pip install ripser persim")

    log_returns = np.log(prices / prices.shift(1)).dropna().values
    n = len(log_returns)

    records = []
    indices = []

    outer_window = embed_window * 3   # outer rolling window

    for i in range(outer_window, n, tda_stride):
        segment = log_returns[i - outer_window:i]
        cloud = sliding_window_embedding(segment, embed_window, embed_stride)

        try:
            ph = compute_persistence(cloud, max_dim=max_dim)
            feats = persistence_features(ph["dgms"])
        except Exception as e:
            logger.debug(f"TDA failed at i={i}: {e}")
            feats = {}

        records.append(feats)
        indices.append(prices.index[i + 1])  # +1 offset: returns lag

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, index=indices)
    df = df.fillna(0.0)
    logger.info(f"Rolling TDA: {len(df)} rows, {len(df.columns)} features")
    return df


# ---------------------------------------------------------------------------
# 6. Change-point detection via TDA distances
# ---------------------------------------------------------------------------

def tda_change_detection(
    prices: pd.Series,
    window: int = 30,
    embed_dim: int = 10,
    stride: int = 5,
    metric: str = "wasserstein",
) -> pd.Series:
    """
    Detect regime changes by tracking Wasserstein distance between
    consecutive persistence diagrams. Large spikes = topological change.

    Returns
    -------
    pd.Series of distances (indexed to price timestamps)
    """
    if not RIPSER_OK:
        raise ImportError("Install ripser/persim.")

    log_r = np.log(prices / prices.shift(1)).dropna().values
    n = len(log_r)

    prev_dgm = None
    distances = []
    timestamps = []

    for i in range(window, n, stride):
        segment = log_r[i - window:i]
        cloud = sliding_window_embedding(segment, embed_dim)

        try:
            ph = compute_persistence(cloud, max_dim=1)
            dgm_h1 = ph.get("H1", np.zeros((1, 2)))

            if prev_dgm is not None and len(dgm_h1) > 0 and len(prev_dgm) > 0:
                d = topological_distance(prev_dgm, dgm_h1, metric)
                distances.append(d)
                timestamps.append(prices.index[i + 1])

            prev_dgm = dgm_h1

        except Exception as e:
            logger.debug(f"TDA change det. failed at i={i}: {e}")

    s = pd.Series(distances, index=timestamps, name=f"tda_{metric}_dist")
    logger.info(f"TDA change detection: {len(s)} distance values computed")
    return s