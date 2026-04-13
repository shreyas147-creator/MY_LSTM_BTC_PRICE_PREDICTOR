"""
models/ensemble.py — Fixed weighted ensemble.
All models must output probability of UP move in [0,1].
"""

import json
import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.gpu import clear_vram
from models.optimiser import optimise_ensemble_weights
from config import ENSEMBLE_WEIGHTS, CONFIDENCE_THRESHOLD, MODELS_DIR

logger = get_logger()


class Ensemble:
    def __init__(self, model_names: list[str] = None):
        self.model_names = model_names or list(ENSEMBLE_WEIGHTS.keys())
        self.weights = np.array(
            [ENSEMBLE_WEIGHTS.get(m, 0.0) for m in self.model_names],
            dtype=np.float32,
        )
        if self.weights.sum() == 0:
            self.weights = np.ones(len(self.model_names)) / len(self.model_names)
        else:
            self.weights /= self.weights.sum()

        self._predictions: dict[str, np.ndarray] = {}

    def validate_proba(name: str, proba: np.ndarray):
        if not isinstance(proba, np.ndarray):
            raise TypeError(f"{name}: output not numpy array")

        if proba.ndim != 2 or proba.shape[1] != 3:
            raise ValueError(f"{name}: expected shape (N,3), got {proba.shape}")

        if not np.all(np.isfinite(proba)):
            raise ValueError(f"{name}: contains NaN/inf")

        row_sums = proba.sum(axis=1)

        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError(f"{name}: probabilities not normalized")

        if np.any(proba < 0) or np.any(proba > 1):
            raise ValueError(f"{name}: probabilities outside [0,1]")

    # -------------------------------------------------------
    # Collect predictions
    # -------------------------------------------------------

    def add_predictions(self, model_name: str, preds: np.ndarray) -> None:
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)

        if model_name not in self.model_names:
            logger.warning(f"Unknown model '{model_name}' — adding dynamically.")
            self.model_names.append(model_name)
            self.weights = np.append(self.weights, 0.0)

        if np.any((preds < 0) | (preds > 1)):
            logger.warning(f"{model_name} predictions not in [0,1], clipping.")
            preds = np.clip(preds, 0.0, 1.0)

        self._predictions[model_name] = preds
        logger.debug(f"{model_name} predictions added | shape={preds.shape}")

    # -------------------------------------------------------
    # Optimise weights (FIXED)
    # -------------------------------------------------------

    def fit_weights(self, true_labels: np.ndarray) -> np.ndarray:
        """
        Optimise ensemble weights on validation predictions.
        """
        available = [m for m in self.model_names if m in self._predictions]

        if len(available) < 2:
            logger.warning("Need at least 2 models to optimise ensemble weights.")
            return self.weights

        # --- FIX: build dict instead of matrix ---
        model_returns = {
            m: self._predictions[m] for m in available
        }

        opt_dict = optimise_ensemble_weights(model_returns)

        # Update weights
        for name, w in opt_dict.items():
            if name in self.model_names:
                idx = self.model_names.index(name)
                self.weights[idx] = w

        self.weights /= self.weights.sum() + 1e-10

        logger.info(
            f"Ensemble weights optimised: "
            f"{dict(zip(self.model_names, self.weights.round(3)))}"
        )

        return self.weights

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------

    def predict_proba(self) -> np.ndarray:
        available = [m for m in self.model_names if m in self._predictions]

        if not available:
            raise RuntimeError("No predictions registered.")

        stack = np.stack([self._predictions[m] for m in available], axis=0)
        w = np.array([self.weights[self.model_names.index(m)] for m in available])
        w /= w.sum()

        blended = (stack * w[:, None]).sum(axis=0)
        return blended

    def predict(self, threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
        prob = self.predict_proba()

        signal = np.where(
            prob >= threshold,
            1,
            np.where(prob <= 1 - threshold, -1, 0),
        )

        confidence = np.abs(prob - 0.5)

        return pd.DataFrame({
            "prob_up": prob,
            "signal": signal,
            "confidence": confidence,
            "act": confidence >= (threshold - 0.5),
        })

    # -------------------------------------------------------
    # Persistence
    # -------------------------------------------------------

    def save_weights(self, filename: str = "ensemble_weights.json") -> None:
        path = MODELS_DIR / filename
        data = {name: float(w) for name, w in zip(self.model_names, self.weights)}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Weights saved → {path}")

    def load_weights(self, filename: str = "ensemble_weights.json") -> None:
        path = MODELS_DIR / filename

        if not path.exists():
            logger.warning("No saved weights — using defaults.")
            return

        with open(path) as f:
            data = json.load(f)

        self.model_names = list(data.keys())
        self.weights = np.array(list(data.values()), dtype=np.float32)
        self.weights /= self.weights.sum()

        logger.info(f"Weights loaded ← {path}")


# -----------------------------------------------------------
# FULL PIPELINE (FIXED)
# -----------------------------------------------------------

def run_ensemble(
    feature_matrix: np.ndarray,
    prices: np.ndarray,
    lstm_trainer=None,
    transformer_trainer=None,
    xgb_model=None,
    lgbm_model=None,
    gp_model=None,
    rl_agent=None,
) -> pd.DataFrame:

    ensemble = Ensemble()
    ensemble.load_weights()

    # ---------------- LSTM ----------------
    if lstm_trainer is not None:
        try:
            probs = lstm_trainer.predict_proba(feature_matrix)
            # class 2 = UP (since you use [-1,0,1])
            preds = probs[:, 2]
            ensemble.add_predictions("lstm", preds)
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")

    # ---------------- TRANSFORMER ----------------
    if transformer_trainer is not None:
        try:
            probs = transformer_trainer.predict_proba(feature_matrix)
            preds = probs[:, 2]
            ensemble.add_predictions("transformer", preds)
        except Exception as e:
            logger.warning(f"Transformer prediction failed: {e}")

    # ---------------- XGBOOST ----------------
    if xgb_model is not None:
        try:
            probs = xgb_model.predict_proba(feature_matrix)
            preds = probs[:, 2]
            ensemble.add_predictions("xgboost", preds)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")

    # ---------------- LIGHTGBM (FIXED) ----------------
    if lgbm_model is not None:
        try:
            probs = lgbm_model.predict_proba(feature_matrix)
            preds = probs[:, 2]
            ensemble.add_predictions("lightgbm", preds)
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
    # ---------------- GAUSSIAN PROCESS (FIXED) ----------------
    if gp_model is not None:
        try:
            out = gp_model.predict(feature_matrix)
            mean = out["mean"]
            preds = 1 / (1 + np.exp(-mean))  # sigmoid
            ensemble.add_predictions("gp", preds)
        except Exception as e:
            logger.warning(f"GP prediction failed: {e}")

    # ---------------- RL (IMPROVED) ----------------
    if rl_agent is not None:
        try:
            result = rl_agent.predict(feature_matrix, prices)
            actions = np.array(result["actions"], dtype=np.int32)

            preds = np.full(len(actions), 0.5)
            preds[actions == 1] = 0.8  # buy
            preds[actions == 2] = 0.2  # sell

            ensemble.add_predictions("rl", preds)
        except Exception as e:
            logger.warning(f"RL prediction failed: {e}")

    clear_vram()
    return ensemble.predict()