"""
evaluation/explainability.py — SHAP values and feature importance.
Explains model predictions for debugging and feature validation.
Works with XGBoost, LightGBM, and LSTM.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger()

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    logger.warning("shap not installed. Run: pip install shap")


# ---------------------------------------------------------------------------
# Tree-based SHAP (XGBoost / LightGBM)
# ---------------------------------------------------------------------------

def shap_tree(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP values for a tree model (XGBoost or LightGBM).

    Returns DataFrame of mean |SHAP| per feature, sorted descending.
    """
    if not SHAP_OK:
        raise ImportError("Install shap: pip install shap")

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    # Get the underlying booster if it's a sklearn wrapper
    booster = model.model if hasattr(model, "model") else model

    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    result   = pd.Series(mean_abs, index=X.columns, name="mean_abs_shap")
    return result.sort_values(ascending=False).to_frame()


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic)
# ---------------------------------------------------------------------------

def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    metric_fn=None,
) -> pd.DataFrame:
    """
    Model-agnostic permutation feature importance.
    Shuffles each feature and measures the drop in performance.

    Parameters
    ----------
    model      : any model with a .predict() method
    metric_fn  : callable(y_true, y_pred) → float. Defaults to accuracy.
    """
    from sklearn.metrics import accuracy_score

    if metric_fn is None:
        def metric_fn(yt, yp):
            return accuracy_score(yt, (yp >= 0.5).astype(int))

    baseline = metric_fn(y, model.predict(X))
    scores   = {}

    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm      = X.copy()
            idx         = np.random.permutation(len(X_perm))
            X_perm[:, i] = X_perm[idx, i]
            score = metric_fn(y, model.predict(X_perm))
            drops.append(baseline - score)
        scores[name] = np.mean(drops)

    result = pd.Series(scores, name="importance").sort_values(ascending=False)
    logger.info(f"Permutation importance computed for {len(feature_names)} features")
    return result.to_frame()


# ---------------------------------------------------------------------------
# Feature importance summary across all models
# ---------------------------------------------------------------------------

def aggregate_importance(
    xgb_model=None,
    lgbm_model=None,
    feature_names: list[str] = None,
) -> pd.DataFrame:
    """
    Aggregate feature importance from XGBoost and LightGBM.
    Returns a combined DataFrame sorted by average importance.
    """
    frames = {}

    if xgb_model is not None and xgb_model.model is not None:
        try:
            imp = xgb_model.feature_importance(feature_names)
            frames["xgboost"] = imp / (imp.sum() + 1e-8)
        except Exception as e:
            logger.warning(f"XGBoost importance failed: {e}")

    if lgbm_model is not None and lgbm_model.model is not None:
        try:
            imp = lgbm_model.feature_importance(feature_names)
            frames["lightgbm"] = imp / (imp.sum() + 1e-8)
        except Exception as e:
            logger.warning(f"LightGBM importance failed: {e}")

    if not frames:
        logger.warning("No models available for importance aggregation.")
        return pd.DataFrame()

    combined = pd.DataFrame(frames)
    combined["mean_importance"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_importance", ascending=False)

    logger.info(f"Top 10 features:\n{combined['mean_importance'].head(10)}")
    return combined


# ---------------------------------------------------------------------------
# Quick summary report
# ---------------------------------------------------------------------------

def importance_report(
    xgb_model=None,
    lgbm_model=None,
    X_val: pd.DataFrame = None,
    feature_names: list[str] = None,
    top_n: int = 20,
) -> dict:
    """
    Generate a full importance report dict with:
        - aggregated tree importance
        - SHAP values (if X_val provided and shap installed)
    """
    report = {}

    agg = aggregate_importance(xgb_model, lgbm_model, feature_names)
    if not agg.empty:
        report["aggregated"] = agg.head(top_n)

    if SHAP_OK and X_val is not None and xgb_model is not None:
        try:
            shap_df = shap_tree(xgb_model, X_val, max_samples=500)
            report["shap_xgb"] = shap_df.head(top_n)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

    return report