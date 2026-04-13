"""
utils/time_utils.py — Timestamp alignment, resampling, timezone helpers.
All timestamps in this project are UTC. Never store local time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# UTC helpers
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    """Current time in UTC."""
    return datetime.now(timezone.utc)


def ts_to_utc(ts) -> pd.Timestamp:
    """Coerce any timestamp-like value to UTC pandas Timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def ms_to_utc(ms: int) -> pd.Timestamp:
    """Convert Unix milliseconds → UTC Timestamp (Binance format)."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def utc_to_ms(ts: pd.Timestamp) -> int:
    """Convert UTC Timestamp → Unix milliseconds."""
    return int(ts.timestamp() * 1000)


def days_ago(n: int) -> pd.Timestamp:
    """Return UTC timestamp N days ago."""
    return pd.Timestamp.utcnow() - pd.Timedelta(days=n)


# ---------------------------------------------------------------------------
# DatetimeIndex helpers
# ---------------------------------------------------------------------------

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a UTC DatetimeIndex. Localises if naive."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def drop_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping the last occurrence."""
    n_before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.debug(f"Dropped {n_dropped} duplicate timestamps.")
    return df


def sort_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a coarser timeframe.
    e.g. 1h → 4h, 1h → 1d

    Parameters
    ----------
    df        : DataFrame with columns open/high/low/close/volume and UTC DatetimeIndex
    timeframe : pandas offset alias e.g. '4h', '1d', '1W'
    """
    df = ensure_utc_index(df)
    # Map ccxt aliases to pandas aliases
    alias_map = {"1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"}
    pd_alias = alias_map.get(timeframe, timeframe)

    resampled = df.resample(pd_alias).agg(OHLCV_AGG).dropna()
    logger.debug(f"Resampled {len(df)} → {len(resampled)} bars at {timeframe}")
    return resampled


# ---------------------------------------------------------------------------
# Alignment — merge multiple DataFrames to a common UTC index
# ---------------------------------------------------------------------------

def align_to_index(
    base: pd.DataFrame,
    other: pd.DataFrame,
    method: str = "ffill",
    limit: Optional[int] = 5,
) -> pd.DataFrame:
    """
    Reindex `other` to match `base` index using forward-fill.
    Used to align on-chain / sentiment data to OHLCV timestamps.
    """
    other = ensure_utc_index(other)
    aligned = other.reindex(base.index, method=method, limit=limit)
    return aligned


def merge_on_index(frames: list[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    """
    Merge a list of DataFrames on their DatetimeIndex.
    Forward-fills any gaps after merging.
    """
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how=how)
    combined = combined.ffill().sort_index()
    logger.debug(f"Merged {len(frames)} frames → {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Train/val/test split by time (no shuffle — preserves temporal order)
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_days: int = 365,
    val_days: int = 30,
    test_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / val / test by date.
    Uses the last (test_days + val_days) rows as test/val,
    everything before as train.
    """
    df = sort_index(ensure_utc_index(df))
    end = df.index[-1]

    test_start  = end - pd.Timedelta(days=test_days)
    val_start   = test_start - pd.Timedelta(days=val_days)
    train_end   = val_start

    train   = df[df.index < train_end]
    val     = df[(df.index >= val_start) & (df.index < test_start)]
    test    = df[df.index >= test_start]

    logger.info(
        f"Split: train={len(train):,} | val={len(val):,} | test={len(test):,} rows"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Walk-forward fold generator
# ---------------------------------------------------------------------------

def walkforward_folds(
    df: pd.DataFrame,
    train_days: int = 365,
    val_days: int = 30,
    step_days: int = 30,
) -> list[dict]:
    """
    Generate walk-forward folds for time-series cross-validation.
    Each fold: {train: df, val: df, fold_id: int}
    """
    df = sort_index(ensure_utc_index(df))
    start = df.index[0]
    end   = df.index[-1]

    folds = []
    fold_start = start
    fold_id = 0

    while True:
        train_end = fold_start + pd.Timedelta(days=train_days)
        val_end   = train_end + pd.Timedelta(days=val_days)

        if val_end > end:
            break

        train_fold  = df[(df.index >= fold_start) & (df.index < train_end)]
        val_fold    = df[(df.index >= train_end) & (df.index < val_end)]

        if len(train_fold) == 0 or len(val_fold) == 0:
            break

        folds.append({"fold_id": fold_id, "train": train_fold, "val": val_fold})
        fold_start += pd.Timedelta(days=step_days)
        fold_id += 1

    logger.info(f"Walk-forward: {len(folds)} folds | train={train_days}d, val={val_days}d, step={step_days}d")
    return folds