"""
features/labels.py — Forward return, direction, and regime label generation.
These are the targets the models are trained to predict.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger
from config import FORWARD_HOURS, DIRECTION_THRESHOLD, N_REGIMES

logger = get_logger()


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame, horizon: int = FORWARD_HOURS) -> pd.DataFrame:
    """
    Add forward log return and percentage return labels.

    forward_return_{h}h = log(close_{t+h} / close_t)
    forward_pct_{h}h    = (close_{t+h} - close_t) / close_t

    These will be NaN for the last `horizon` rows — drop before training.
    """
    df[f"forward_return_{horizon}h"] = np.log(
        df["close"].shift(-horizon) / df["close"]
    )
    df[f"forward_pct_{horizon}h"] = (
        df["close"].shift(-horizon) - df["close"]
    ) / df["close"]
    return df


# ---------------------------------------------------------------------------
# Direction labels (classification target)
# ---------------------------------------------------------------------------

def add_direction_label(
    df: pd.DataFrame,
    horizon: int = FORWARD_HOURS,
    threshold: float = DIRECTION_THRESHOLD,
) -> pd.DataFrame:
    """
    Three-class direction label based on forward return:
        1  = Up   (forward_return > +threshold)
        0  = Flat (|forward_return| <= threshold)
       -1  = Down (forward_return < -threshold)

    Also adds binary label (up=1, not-up=0) for binary classifiers.
    """
    col = f"forward_return_{horizon}h"
    if col not in df.columns:
        df = add_forward_returns(df, horizon)

    ret = df[col]
    label = pd.Series(0, index=df.index, name=f"direction_{horizon}h")
    label[ret >  threshold] =  1
    label[ret < -threshold] = -1

    df[f"direction_{horizon}h"] = label
    df[f"up_binary_{horizon}h"] = (label == 1).astype(int)

    counts = label.value_counts()
    logger.info(
        f"Direction labels ({horizon}h): "
        f"Up={counts.get(1,0):,} | "
        f"Flat={counts.get(0,0):,} | "
        f"Down={counts.get(-1,0):,}"
    )
    return df


# ---------------------------------------------------------------------------
# Volatility regime labels
# ---------------------------------------------------------------------------

def add_volatility_regime(
    df: pd.DataFrame,
    window: int = 168,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """
    Simple percentile-based volatility regime:
        0 = low vol
        1 = medium vol
        2 = high vol

    Based on rolling realised volatility percentile.
    """
    if "realvol_24h" not in df.columns:
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["realvol_24h"] = df["log_return"].rolling(24).std() * np.sqrt(24)

    vol = df["realvol_24h"]
    pct = vol.rolling(window).rank(pct=True)

    if n_regimes == 3:
        regime = pd.cut(
            pct,
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(float)
    else:
        regime = pd.qcut(pct, q=n_regimes, labels=False, duplicates="drop")

    df["vol_regime"] = regime
    return df


# ---------------------------------------------------------------------------
# Trend regime labels (simple EMA crossover)
# ---------------------------------------------------------------------------

def add_trend_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend regime based on EMA crossover:
        1  = bullish (fast EMA > slow EMA)
        -1 = bearish (fast EMA < slow EMA)
        0  = transitioning
    """
    if "ema_21" not in df.columns:
        df["ema_21"] = df["close"].ewm(span=21).mean()
    if "ema_50" not in df.columns:
        df["ema_50"] = df["close"].ewm(span=50).mean()

    trend = pd.Series(0, index=df.index, name="trend_regime")
    trend[df["ema_21"] > df["ema_50"] * 1.001] =  1
    trend[df["ema_21"] < df["ema_50"] * 0.999] = -1

    df["trend_regime"] = trend
    return df


# ---------------------------------------------------------------------------
# Master label pipeline
# ---------------------------------------------------------------------------

def compute_labels(
    df: pd.DataFrame,
    horizon: int = FORWARD_HOURS,
    threshold: float = DIRECTION_THRESHOLD,
) -> pd.DataFrame:
    """
    Add all label columns to the OHLCV + features DataFrame.

    Labels added:
        forward_return_{h}h  — continuous regression target
        forward_pct_{h}h     — percentage return target
        direction_{h}h       — 3-class: -1/0/1
        up_binary_{h}h       — binary: 0/1
        vol_regime           — 0/1/2 (low/med/high vol)
        trend_regime         — -1/0/1

    Note: Last `horizon` rows will have NaN forward labels.
          Drop them before training with df.dropna(subset=[label_col])
    """
    logger.info("Computing labels...")
    df = add_forward_returns(df, horizon)
    df = add_direction_label(df, horizon, threshold)
    df = add_volatility_regime(df)
    df = add_trend_regime(df)
    return df


def get_label_cols(horizon: int = FORWARD_HOURS) -> dict:
    """Return dict of label column names for easy reference."""
    return {
        "regression":   f"forward_return_{horizon}h",
        "pct_return":   f"forward_pct_{horizon}h",
        "direction":    f"direction_{horizon}h",
        "binary":       f"up_binary_{horizon}h",
        "vol_regime":   "vol_regime",
        "trend_regime": "trend_regime",
    }


if __name__ == "__main__":
    from utils.logger import setup_logger
    from ingestion.fetch_ohlcv import load_ohlcv
    setup_logger()
    ohlcv = load_ohlcv("1h")
    labeled = compute_labels(ohlcv)
    label_cols = list(get_label_cols().values())
    print(labeled[label_cols].tail(10))
    print(labeled[label_cols].isna().sum())