"""
models/regime_classifier.py — Market regime classifier.
Combines HMM (unsupervised) with XGBoost (supervised) to label
and predict bull / bear / sideways regimes.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger
from config import N_REGIMES, HMM_N_ITER, HMM_COV_TYPE, MODELS_DIR

logger = get_logger()

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    HMM_OK = False
    logger.warning("hmmlearn not found.")


# ---------------------------------------------------------------------------
# HMM regime detection
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """
    Fits a Gaussian HMM to price features to discover latent market regimes.
    Uses: log_return, realised_vol, volume_ratio as observation sequence.
    """

    def __init__(self, n_regimes: int = N_REGIMES):
        if not HMM_OK:
            raise ImportError("hmmlearn required.")
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components = n_regimes,
            covariance_type = HMM_COV_TYPE,
            n_iter = HMM_N_ITER,
            random_state = 42,
        )
        self._regime_map = {}   # maps HMM state → named regime

    def _build_obs(self, df: pd.DataFrame) -> np.ndarray:
        """Build observation matrix from feature DataFrame."""
        cols = []
        if "log_return" in df.columns:
            cols.append(df["log_return"].fillna(0).values)
        if "realvol_24h" in df.columns:
            cols.append(df["realvol_24h"].fillna(0).values)
        if "volume_ratio" in df.columns:
            cols.append(df["volume_ratio"].fillna(1).values)
        if not cols:
            raise ValueError("Need log_return, realvol_24h, volume_ratio columns.")
        return np.column_stack(cols)

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        obs = self._build_obs(df)
        self.model.fit(obs)
        states = self.model.predict(obs)

        # Label regimes by mean return: highest = bull, lowest = bear
        means = {}
        for s in range(self.n_regimes):
            mask = states == s
            means[s] = df["log_return"].fillna(0).values[mask].mean() if mask.any() else 0.0

        sorted_states = sorted(means, key=means.get)
        if self.n_regimes == 3:
            self._regime_map = {
                sorted_states[0]: "bear",
                sorted_states[1]: "sideways",
                sorted_states[2]: "bull",
            }
        else:
            for i, s in enumerate(sorted_states):
                self._regime_map[s] = f"regime_{i}"

        logger.info(f"HMM fitted | regime map: {self._regime_map}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        obs    = self._build_obs(df)
        states = self.model.predict(obs)
        labels = [self._regime_map.get(s, "unknown") for s in states]
        return pd.Series(labels, index=df.index, name="hmm_regime")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        obs   = self._build_obs(df)
        proba = self.model.predict_proba(obs)
        cols  = [self._regime_map.get(i, f"s{i}") for i in range(self.n_regimes)]
        return pd.DataFrame(proba, index=df.index, columns=cols)


# ---------------------------------------------------------------------------
# Supervised regime classifier (XGBoost on HMM labels)
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """
    Two-stage classifier:
      1. HMM labels historical data with regime
      2. XGBoost learns to predict the regime from features
         (so we can predict the regime for unseen data)
    """

    def __init__(self, n_regimes: int = N_REGIMES):
        self.hmm        = HMMRegimeDetector(n_regimes)
        self.classifier = None
        self.n_regimes  = n_regimes
        self.label_enc  = {"bear": 0, "sideways": 1, "bull": 2}

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "RegimeClassifier":
        import xgboost as xgb

        # Stage 1: HMM labels
        self.hmm.fit(df)
        hmm_labels = self.hmm.predict(df)

        # Stage 2: train XGBoost to predict HMM labels from features
        y = hmm_labels.map(self.label_enc).fillna(1).astype(int)
        X = df[feature_cols].fillna(0).values

        self.classifier = xgb.XGBClassifier(
            device="cuda",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.classifier.fit(X, y)
        logger.info("Regime classifier fitted.")
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
        X = df[feature_cols].fillna(0).values
        preds = self.classifier.predict(X)
        inv   = {v: k for k, v in self.label_enc.items()}
        return pd.Series([inv[p] for p in preds], index=df.index, name="regime")

    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        X     = df[feature_cols].fillna(0).values
        proba = self.classifier.predict_proba(X)
        return pd.DataFrame(
            proba,
            index=df.index,
            columns=["bear_prob", "sideways_prob", "bull_prob"],
        )


if __name__ == "__main__":
    from utils.logger import setup_logger
    from ingestion.fetch_ohlcv import load_ohlcv
    from features.technical import compute_technical_features
    setup_logger()
    ohlcv = load_ohlcv("1h")
    df = compute_technical_features(ohlcv)
    detector = HMMRegimeDetector(n_regimes=3)
    detector.fit(df)
    regimes = detector.predict(df)
    print(regimes.value_counts())