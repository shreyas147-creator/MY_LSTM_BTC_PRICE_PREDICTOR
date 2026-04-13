"""
information.py — Information Theory for Financial Signals
Covers: Shannon entropy, KL divergence, mutual information,
        transfer entropy (causal information flow), relative entropy,
        information-theoretic feature selection.
Maps to: Information Theory block (entropy, mutual info, KL div.).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.neighbors import KernelDensity
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn not found. Some features will be limited.")


# ---------------------------------------------------------------------------
# 1. Discrete entropy utilities
# ---------------------------------------------------------------------------

def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    """
    Shannon entropy H(X) = -Σ p_i log p_i.

    Parameters
    ----------
    p    : probability mass function (will be normalised)
    base : logarithm base (2=bits, e=nats)
    """
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p) / np.log(base)))


def empirical_entropy(series: np.ndarray, n_bins: int = 30, base: float = 2.0) -> float:
    """
    Estimate entropy from data via histogram binning.
    """
    counts, _ = np.histogram(series, bins=n_bins)
    probs = counts / counts.sum()
    return shannon_entropy(probs, base)


def differential_entropy_gaussian(sigma: float) -> float:
    """
    Differential entropy of a Gaussian: h(X) = 0.5 ln(2πe σ²).
    """
    return 0.5 * float(np.log(2 * np.pi * np.e * sigma ** 2))


# ---------------------------------------------------------------------------
# 2. Joint entropy & mutual information
# ---------------------------------------------------------------------------

def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
    base: float = 2.0,
) -> float:
    """
    Joint entropy H(X, Y) via 2-D histogram.
    """
    counts, _, _ = np.histogram2d(x, y, bins=n_bins)
    probs = counts / counts.sum()
    return shannon_entropy(probs.flatten(), base)


def mutual_information_discrete(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
    base: float = 2.0,
) -> float:
    """
    Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
    """
    hx = empirical_entropy(x, n_bins, base)
    hy = empirical_entropy(y, n_bins, base)
    hxy = joint_entropy(x, y, n_bins, base)
    mi = hx + hy - hxy
    return float(max(mi, 0.0))   # numerical floor at 0


def normalised_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    NMI = I(X;Y) / sqrt(H(X)·H(Y))  ∈ [0, 1]
    1 = perfect dependence, 0 = independence.
    """
    mi = mutual_information_discrete(x, y, n_bins)
    hx = empirical_entropy(x, n_bins)
    hy = empirical_entropy(y, n_bins)
    denom = np.sqrt(hx * hy)
    return float(mi / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. KL Divergence (relative entropy)
# ---------------------------------------------------------------------------

def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
    epsilon: float = 1e-10,
) -> float:
    """
    KL divergence D_KL(P || Q) = Σ p_i log(p_i / q_i).
    Asymmetric: measures how much Q differs from P.

    Parameters
    ----------
    p, q : probability distributions (will be normalised)
    """
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q) / np.log(base)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence — symmetric, bounded in [0,1] (for base-2 log).
    JS(P||Q) = 0.5 KL(P||M) + 0.5 KL(Q||M),  M = (P+Q)/2
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m, base=2) + 0.5 * kl_divergence(q, m, base=2))


def kl_returns_vs_gaussian(returns: np.ndarray, n_bins: int = 50) -> float:
    """
    KL divergence of empirical return distribution vs fitted Gaussian.
    High KL → fat tails / non-Gaussian regime.
    """
    counts, bin_edges = np.histogram(returns, bins=n_bins, density=False)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p = counts / counts.sum()

    mu, sigma = float(np.mean(returns)), float(np.std(returns))
    q = stats.norm.pdf(bin_centers, mu, sigma)
    q /= q.sum()

    kl = kl_divergence(p, q)
    logger.info(f"KL(empirical || Gaussian) = {kl:.4f}  (>0.1 suggests non-Gaussian)")
    return kl


# ---------------------------------------------------------------------------
# 4. Transfer Entropy — directional causal flow
# ---------------------------------------------------------------------------

def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 10,
    base: float = 2.0,
) -> float:
    """
    Transfer entropy T(S → T) = I(T_future ; S_past | T_past).
    Measures information flow from source to target (directional MI).

    T_{S→T} = H(T_{t+1} | T_t) - H(T_{t+1} | T_t, S_t)

    Parameters
    ----------
    source, target : 1-D time series
    lag            : history lag

    Returns
    -------
    Transfer entropy in bits.
    """
    n = min(len(source), len(target))
    S = np.asarray(source[:n], dtype=np.float64)
    T = np.asarray(target[:n], dtype=np.float64)

    # Discretise via equal-frequency binning
    def discretise(x, bins):
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(x, quantiles)
        edges[-1] += 1e-10
        return np.digitize(x, edges[1:-1])

    S_d = discretise(S, n_bins)
    T_d = discretise(T, n_bins)

    # Build lagged arrays
    T_future = T_d[lag:]
    T_past = T_d[:-lag]
    S_past = S_d[:-lag]

    # Trim to same length
    min_len = min(len(T_future), len(T_past), len(S_past))
    T_future = T_future[:min_len]
    T_past = T_past[:min_len]
    S_past = S_past[:min_len]

    # H(T_{t+1} | T_t) — conditional entropy
    h_tf_given_tp = _conditional_entropy_discrete(T_future, T_past, n_bins)

    # H(T_{t+1} | T_t, S_t) — conditional entropy given both
    joint_past = T_past * n_bins + S_past   # encode (T_t, S_t) as single variable
    h_tf_given_tp_sp = _conditional_entropy_discrete(T_future, joint_past, n_bins ** 2)

    te = h_tf_given_tp - h_tf_given_tp_sp
    return float(max(te, 0.0))


def _conditional_entropy_discrete(Y: np.ndarray, X: np.ndarray, n_states_x: int) -> float:
    """H(Y|X) = H(X,Y) - H(X)."""
    joint_counts = np.zeros((n_states_x, n_states_x), dtype=float)
    for xi, yi in zip(X, Y):
        xi = min(int(xi), n_states_x - 1)
        yi = min(int(yi), n_states_x - 1)
        joint_counts[xi, yi] += 1

    p_joint = joint_counts / joint_counts.sum()
    p_x = p_joint.sum(axis=1)

    h_xy = shannon_entropy(p_joint.flatten())
    h_x = shannon_entropy(p_x)
    return h_xy - h_x


def transfer_entropy_matrix(
    df: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute pairwise transfer entropy matrix.
    TE[i,j] = T(col_i → col_j)
    """
    cols = df.columns.tolist()
    n = len(cols)
    te_mat = np.zeros((n, n))

    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            if i != j:
                te_mat[i, j] = transfer_entropy(
                    df[src].dropna().values,
                    df[tgt].dropna().values,
                    lag=lag, n_bins=n_bins,
                )

    return pd.DataFrame(te_mat, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# 5. Information-theoretic feature selection
# ---------------------------------------------------------------------------

def information_feature_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "regression",
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Rank features by mutual information with the target.

    Parameters
    ----------
    X    : feature matrix
    y    : target variable
    task : 'regression' | 'classification'

    Returns
    -------
    DataFrame with columns: feature, mutual_info, nmi, kl_gaussian
    """
    records = []

    for col in X.columns:
        x = X[col].fillna(0).values
        mi = mutual_information_discrete(x, y.values, n_bins)
        nmi = normalised_mutual_information(x, y.values, n_bins)
        kl = kl_returns_vs_gaussian(x - x.mean()) if x.std() > 0 else 0.0
        records.append({"feature": col, "mutual_info": mi, "nmi": nmi, "kl_vs_gaussian": kl})

    df = pd.DataFrame(records).sort_values("mutual_info", ascending=False).reset_index(drop=True)
    logger.info(f"Top 5 features by MI:\n{df.head(5)[['feature', 'mutual_info']].to_string()}")
    return df


# ---------------------------------------------------------------------------
# 6. Rolling entropy features — for ML pipeline
# ---------------------------------------------------------------------------

def rolling_entropy_features(
    returns: pd.Series,
    window: int = 60,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Compute rolling entropy metrics as features for the ML model.

    Features:
    - rolling_entropy  : rolling Shannon entropy of returns
    - kl_vs_normal     : KL divergence vs fitted Gaussian
    - excess_kurtosis  : non-Gaussianity measure
    """
    r = returns.dropna()
    n = len(r)
    records = []

    for i in range(window, n):
        seg = r.iloc[i - window:i].values
        ent = empirical_entropy(seg, n_bins)
        kl = kl_returns_vs_gaussian(seg, n_bins)
        kurt = float(stats.kurtosis(seg))
        records.append({
            "rolling_entropy": ent,
            "kl_vs_gaussian": kl,
            "excess_kurtosis": kurt,
        })

    idx = r.index[window:]
    df = pd.DataFrame(records, index=idx)
    return df