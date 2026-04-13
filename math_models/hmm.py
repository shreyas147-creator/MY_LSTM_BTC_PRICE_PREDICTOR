"""
hmm.py — Hidden Markov Models for Regime Detection
Covers: Gaussian HMM, regime labelling (bull/bear/sideways), Viterbi decoding,
        transition matrix analysis, regime-conditioned statistics.
Maps to: Hidden Markov Models block (regime detection).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMMLEARN_OK = True
except ImportError:
    HMMLEARN_OK = False
    logger.warning("hmmlearn not found. Run: pip install hmmlearn")


# ---------------------------------------------------------------------------
# 1. Gaussian HMM Regime Detector
# ---------------------------------------------------------------------------

REGIME_LABELS = {
    "2": {0: "bear", 1: "bull"},
    "3": {0: "bear", 1: "sideways", 2: "bull"},
    "4": {0: "crash", 1: "bear", 2: "bull", 3: "rally"},
}


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector with automatic state labelling by mean return.

    Parameters
    ----------
    n_states    : number of hidden states (2=bull/bear, 3=bull/side/bear, …)
    n_iter      : EM iterations
    covariance  : 'full' | 'diag' | 'spherical' | 'tied'
    n_init      : number of random initialisations (best BIC kept)
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        covariance: str = "full",
        n_init: int = 5,
    ):
        if not HMMLEARN_OK:
            raise ImportError("Install hmmlearn: pip install hmmlearn")
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance = covariance
        self.n_init = n_init
        self.model = None
        self.state_means_ = None
        self._state_order = None  # sorted index (bear→bull by mean return)

    def fit(self, features: np.ndarray) -> "HMMRegimeDetector":
        """
        Fit the HMM.

        Parameters
        ----------
        features : (T, n_features) array  — typically [return, vol, volume_change, …]
        """
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        best_model = None
        best_score = -np.inf

        for trial in range(self.n_init):
            model = hmmlearn_hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance,
                n_iter=self.n_iter,
                random_state=trial * 7,
                tol=1e-4,
                verbose=False,
            )
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.debug(f"HMM trial {trial} failed: {e}")

        self.model = best_model

        # Sort states by mean return of first feature (ascending = bear→bull)
        means_0 = self.model.means_[:, 0]
        self._state_order = np.argsort(means_0)   # original_idx → sorted_rank
        self.state_means_ = self.model.means_

        logger.info(
            f"HMM fitted: {self.n_states} states, log-lik={best_score:.2f}, "
            f"state means (feat0): {means_0}"
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Decode most likely state sequence via Viterbi.

        Returns
        -------
        np.ndarray of int state indices (sorted: 0=most bearish, n-1=most bullish)
        """
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        raw_states = self.model.predict(X)

        # Remap to sorted order
        rank_map = np.argsort(self._state_order)   # original → sorted rank
        return rank_map[raw_states]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Return posterior state probabilities (forward-backward smoothed).

        Returns
        -------
        (T, n_states) probability matrix, columns sorted bear→bull
        """
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        proba = self.model.predict_proba(X)
        rank_map = np.argsort(self._state_order)
        return proba[:, self._state_order]   # reorder columns

    def label_regimes(self, features: np.ndarray, index=None) -> pd.DataFrame:
        """
        Full regime analysis: state, label, probabilities, transition stats.
        """
        states = self.predict(features)
        proba = self.predict_proba(features)

        label_map = REGIME_LABELS.get(str(self.n_states), {i: f"state_{i}" for i in range(self.n_states)})
        labels = np.array([label_map.get(s, f"state_{s}") for s in states])

        cols = {
            "regime_state": states,
            "regime_label": labels,
        }
        for i in range(self.n_states):
            lbl = label_map.get(i, f"state_{i}")
            cols[f"p_{lbl}"] = proba[:, i]

        df = pd.DataFrame(cols, index=index)
        logger.info(f"Regime distribution:\n{df['regime_label'].value_counts()}")
        return df

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """Transition probability matrix with labels."""
        label_map = REGIME_LABELS.get(str(self.n_states), {i: f"state_{i}" for i in range(self.n_states)})
        sorted_labels = [label_map.get(i, f"state_{i}") for i in range(self.n_states)]

        # Reorder transition matrix
        A_raw = self.model.transmat_
        A = A_raw[np.ix_(self._state_order, self._state_order)]
        return pd.DataFrame(A, index=sorted_labels, columns=sorted_labels)

    @property
    def bic(self) -> float:
        """BIC score (lower = better)."""
        if self.model is None:
            return np.inf
        n_params = (
            self.n_states ** 2  # transition
            + self.n_states * self.model.means_.shape[1]  # means
            + self.n_states * self.model.covars_.shape[-1] ** 2  # covariances
        )
        # need stored X length — approximate via model internals
        return -2 * self.model.monitor_.history[-1] + n_params * np.log(1000)


# ---------------------------------------------------------------------------
# 2. Regime-conditioned statistics
# ---------------------------------------------------------------------------

def regime_statistics(
    returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Compute per-regime statistics: mean return, vol, Sharpe, hit rate, duration.

    Parameters
    ----------
    returns : pd.Series of log-returns
    regimes : pd.Series of regime labels (same index)
    """
    stats = []
    for regime in regimes.unique():
        mask = regimes == regime
        r = returns[mask]
        mean_r = float(r.mean() * 252)          # annualised
        vol_r = float(r.std() * np.sqrt(252))
        sharpe = mean_r / vol_r if vol_r > 0 else np.nan

        # Average duration
        changes = mask.ne(mask.shift()).cumsum()
        durations = mask[mask].groupby(changes[mask]).size()
        avg_dur = float(durations.mean()) if len(durations) else np.nan

        stats.append({
            "regime": regime,
            "count": int(mask.sum()),
            "mean_return_ann": mean_r,
            "vol_ann": vol_r,
            "sharpe": sharpe,
            "hit_rate": float((r > 0).mean()),
            "avg_duration": avg_dur,
        })

    return pd.DataFrame(stats).set_index("regime").sort_values("mean_return_ann")


# ---------------------------------------------------------------------------
# 3. Best N-state selection by BIC
# ---------------------------------------------------------------------------

def select_hmm_states(
    features: np.ndarray,
    n_range: range = range(2, 6),
    covariance: str = "diag",
    n_iter: int = 100,
) -> dict:
    """
    Fit HMMs for different state counts and select by BIC.

    Returns
    -------
    dict with 'best_n', 'scores', 'best_model'
    """
    X = np.asarray(features, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    scores = {}
    models = {}

    for n in n_range:
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n,
            covariance_type=covariance,
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )
        try:
            model.fit(X)
            ll = model.score(X)
            n_params = n ** 2 + n * X.shape[1] + n * X.shape[1]
            bic = -2 * ll + n_params * np.log(len(X))
            scores[n] = bic
            models[n] = model
            logger.info(f"HMM n={n}: log-lik={ll:.2f}, BIC={bic:.2f}")
        except Exception as e:
            logger.warning(f"HMM n={n} failed: {e}")

    best_n = min(scores, key=scores.get)
    logger.info(f"Best HMM: n_states={best_n}")
    return {"best_n": best_n, "bic_scores": scores, "best_raw_model": models[best_n]}


# ---------------------------------------------------------------------------
# 4. Feature builder for HMM
# ---------------------------------------------------------------------------

def build_hmm_features(
    prices: pd.Series,
    window: int = 20,
) -> np.ndarray:
    """
    Build a compact feature matrix for HMM regime detection:
    [log_return, rolling_vol, volume_z, return_lag1]

    Parameters
    ----------
    prices : pd.Series (price or close)
    window : volatility window

    Returns
    -------
    np.ndarray (T, n_features) with NaN rows dropped
    """
    r = np.log(prices / prices.shift(1))
    vol = r.rolling(window).std()
    r_lag1 = r.shift(1)
    r_lag2 = r.shift(2)

    df = pd.DataFrame({"r": r, "vol": vol, "r1": r_lag1, "r2": r_lag2}).dropna()
    return df.values, df.index