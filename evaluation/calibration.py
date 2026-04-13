"""
evaluation/calibration.py — Probability calibration for model outputs.
Ensures predicted probabilities match empirical frequencies.
Methods: Platt scaling, isotonic regression, reliability diagrams.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger()

try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ---------------------------------------------------------------------------
# Platt scaling (logistic calibration)
# ---------------------------------------------------------------------------

class PlattScaler:
    """
    Fits a logistic regression on raw model scores to calibrate probabilities.
    Simple, fast, works well for binary classification.
    """

    def __init__(self):
        if not SKLEARN_OK:
            raise ImportError("scikit-learn required.")
        self.model = LogisticRegression(C=1.0)
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray) -> "PlattScaler":
        X = raw_scores.reshape(-1, 1)
        self.model.fit(X, true_labels)
        self.fitted = True
        logger.info("Platt scaler fitted.")
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Isotonic regression calibration
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    Non-parametric monotone calibration via isotonic regression.
    Better than Platt scaling when the calibration curve is non-sigmoid.
    Requires more data (~1000+ samples).
    """

    def __init__(self):
        if not SKLEARN_OK:
            raise ImportError("scikit-learn required.")
        self.model  = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(raw_scores, true_labels)
        self.fitted = True
        logger.info("Isotonic calibrator fitted.")
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict(raw_scores)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

def reliability_data(
    probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute reliability diagram data (calibration curve).

    Returns DataFrame with:
        mean_predicted_prob — average predicted probability per bin
        fraction_positive   — actual fraction of positives per bin
        count               — number of samples per bin
        calibration_error   — |mean_predicted_prob - fraction_positive|
    """
    if not SKLEARN_OK:
        return pd.DataFrame()

    frac_pos, mean_pred = calibration_curve(
        true_labels, probs, n_bins=n_bins, strategy="uniform"
    )

    df = pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_positive":   frac_pos,
        "calibration_error":   np.abs(mean_pred - frac_pos),
    })
    return df


def expected_calibration_error(
    probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) — weighted average calibration error.
    Lower is better. Perfect calibration = 0.
    """
    rd = reliability_data(probs, true_labels, n_bins)
    if rd.empty:
        return float("nan")
    # Weight by bin count (approximate)
    return float(rd["calibration_error"].mean())


def brier_score(probs: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Brier score — mean squared error of probability forecasts.
    Range [0, 1]. Lower is better. Skill score vs naive = 0.25 baseline.
    """
    return float(np.mean((probs - true_labels) ** 2))


# ---------------------------------------------------------------------------
# Auto-calibrate ensemble predictions
# ---------------------------------------------------------------------------

def calibrate_predictions(
    raw_probs: np.ndarray,
    true_labels: np.ndarray,
    method: str = "isotonic",
    val_split: float = 0.2,
) -> tuple[np.ndarray, object]:
    """
    Fit a calibrator on a held-out portion and return calibrated probabilities.

    Parameters
    ----------
    raw_probs   : uncalibrated model probabilities
    true_labels : ground truth binary labels
    method      : 'platt' or 'isotonic'
    val_split   : fraction of data to use for calibration fitting

    Returns
    -------
    (calibrated_probs, fitted_calibrator)
    """
    n     = len(raw_probs)
    split = int(n * (1 - val_split))

    cal_X = raw_probs[split:]
    cal_y = true_labels[split:]

    calibrator = PlattScaler() if method == "platt" else IsotonicCalibrator()
    calibrator.fit(cal_X, cal_y)

    calibrated = calibrator.calibrate(raw_probs)

    before_ece = expected_calibration_error(raw_probs,  true_labels)
    after_ece  = expected_calibration_error(calibrated, true_labels)
    before_bs  = brier_score(raw_probs,  true_labels)
    after_bs   = brier_score(calibrated, true_labels)

    logger.info(
        f"Calibration ({method}) | "
        f"ECE: {before_ece:.4f} → {after_ece:.4f} | "
        f"Brier: {before_bs:.4f} → {after_bs:.4f}"
    )
    return calibrated, calibrator