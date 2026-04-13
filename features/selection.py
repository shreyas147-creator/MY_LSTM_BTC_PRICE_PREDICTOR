"""
features/selection.py — Feature selection via mutual information + RMT denoising.
Removes low-signal features before model training to reduce noise and overfitting.

Methods:
  1. Mutual information ranking (sklearn)
  2. Random Matrix Theory denoising (Marchenko-Pastur)
  3. Variance + correlation filtering (fast pre-filter)
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger
from config import MI_THRESHOLD, RMT_THRESHOLD

logger = get_logger()

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn not found.")


# ---------------------------------------------------------------------------
# 1. Variance filter — remove near-zero variance features
# ---------------------------------------------------------------------------

def variance_filter(
    features: pd.DataFrame,
    threshold: float = 1e-6,
) -> list[str]:
    """Remove columns with variance below threshold."""
    var = features.var()
    keep = var[var > threshold].index.tolist()
    dropped = len(features.columns) - len(keep)
    if dropped:
        logger.info(f"Variance filter: dropped {dropped} near-zero variance features")
    return keep


# ---------------------------------------------------------------------------
# 2. Correlation filter — remove redundant features
# ---------------------------------------------------------------------------

def correlation_filter(
    features: pd.DataFrame,
    threshold: float = 0.95,
) -> list[str]:
    """
    Remove features with pairwise Pearson correlation above threshold.
    Keeps the first of each correlated pair.
    """
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    keep = [c for c in features.columns if c not in to_drop]
    logger.info(f"Correlation filter: dropped {len(to_drop)} highly correlated features")
    return keep


# ---------------------------------------------------------------------------
# 3. Mutual information ranking
# ---------------------------------------------------------------------------

def mutual_info_ranking(
    features: pd.DataFrame,
    labels: pd.Series,
    task: str = "classification",
    threshold: float = MI_THRESHOLD,
    n_neighbors: int = 5,
) -> pd.Series:
    """
    Rank features by mutual information with the target label.

    Parameters
    ----------
    features  : feature DataFrame (no NaNs)
    labels    : target Series aligned to features
    task      : 'classification' or 'regression'
    threshold : minimum MI score to keep a feature

    Returns
    -------
    Series of MI scores indexed by feature name, sorted descending
    """
    if not SKLEARN_OK:
        raise ImportError("scikit-learn required for mutual info ranking.")

    X = features.fillna(0).values
    y = labels.values

    if task == "classification":
        mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)
    else:
        mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)

    scores = pd.Series(mi, index=features.columns).sort_values(ascending=False)
    n_above = (scores >= threshold).sum()
    logger.info(
        f"Mutual info: {n_above}/{len(scores)} features above threshold {threshold} "
        f"| top 5: {scores.head(5).to_dict()}"
    )
    return scores


# ---------------------------------------------------------------------------
# 4. Random Matrix Theory (Marchenko-Pastur) denoising
# ---------------------------------------------------------------------------

def rmt_denoise(
    features: pd.DataFrame,
    threshold_pct: float = RMT_THRESHOLD,
) -> list[str]:
    """
    Use Marchenko-Pastur distribution to identify signal vs noise eigenvalues
    in the feature correlation matrix.

    Features corresponding to noise eigenvalues are removed.

    Parameters
    ----------
    features      : feature DataFrame (standardised)
    threshold_pct : fraction of variance explained by signal eigenvalues to keep

    Returns
    -------
    List of feature names corresponding to signal components
    """
    if not SKLEARN_OK:
        logger.warning("sklearn not available — skipping RMT denoising.")
        return features.columns.tolist()

    X = StandardScaler().fit_transform(features.fillna(0))
    T, N = X.shape

    if T < N:
        logger.warning(f"RMT: T={T} < N={N} — insufficient observations. Skipping.")
        return features.columns.tolist()

    # Correlation matrix eigenvalues
    corr = np.corrcoef(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Marchenko-Pastur upper bound
    q = T / N
    lambda_max = (1 + np.sqrt(1 / q)) ** 2

    # Signal eigenvalues exceed lambda_max
    signal_mask = eigenvalues > lambda_max
    n_signal = signal_mask.sum()

    if n_signal == 0:
        logger.warning("RMT: no signal eigenvalues found — keeping all features.")
        return features.columns.tolist()

    # Project features onto signal eigenvectors
    signal_eigenvecs = eigenvectors[:, signal_mask]
    feature_loadings = np.abs(signal_eigenvecs).sum(axis=1)

    # Keep features with non-trivial loading on signal components
    loading_threshold = np.percentile(feature_loadings, (1 - threshold_pct) * 100)
    keep_mask = feature_loadings >= loading_threshold
    keep = features.columns[keep_mask].tolist()

    logger.info(
        f"RMT: {n_signal} signal eigenvalues (lambda_max={lambda_max:.2f}) | "
        f"Kept {len(keep)}/{len(features.columns)} features"
    )
    return keep


# ---------------------------------------------------------------------------
# Master selection pipeline
# ---------------------------------------------------------------------------

def select_features(
    features: pd.DataFrame,
    labels: pd.Series,
    task: str = "classification",
    use_rmt: bool = True,
    mi_threshold: float = MI_THRESHOLD,
    corr_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature selection pipeline:
      1. Variance filter
      2. Correlation filter
      3. Mutual information ranking
      4. RMT denoising (optional)

    Returns
    -------
    (selected_features, mi_scores)
    """
    logger.info(f"Feature selection: starting with {features.shape[1]} features")

    # Align labels to features index
    labels = labels.reindex(features.index).dropna()
    features = features.reindex(labels.index)

    # Step 1: variance
    keep = variance_filter(features)
    features = features[keep]

    # Step 2: correlation
    keep = correlation_filter(features, threshold=corr_threshold)
    features = features[keep]

    # Step 3: mutual information
    mi_scores = mutual_info_ranking(features, labels, task=task, threshold=mi_threshold)
    keep = mi_scores[mi_scores >= mi_threshold].index.tolist()
    features = features[keep]

    # Step 4: RMT
    if use_rmt and len(features.columns) > 10:
        keep = rmt_denoise(features)
        features = features[keep]

    logger.info(f"Feature selection complete: {features.shape[1]} features retained")
    return features, mi_scores


if __name__ == "__main__":
    from utils.logger import setup_logger
    from features.merge import load_feature_matrix
    from config import FORWARD_HOURS
    setup_logger()
    feats, labels = load_feature_matrix()
    label_col = f"direction_{FORWARD_HOURS}h"
    selected, scores = select_features(feats, labels[label_col])
    print(f"Selected features: {selected.shape[1]}")
    print(scores.head(20))