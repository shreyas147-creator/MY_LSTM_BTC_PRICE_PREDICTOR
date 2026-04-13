"""
features/onchain_features.py — On-chain feature engineering.
Normalises and derives features from hash rate, difficulty,
mempool stats, tx volume, and fee rates.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.time_utils import ensure_utc_index, align_to_index
from ingestion.fetch_onchain import load_onchain

logger = get_logger()


# ---------------------------------------------------------------------------
# Hash rate features
# ---------------------------------------------------------------------------

def add_hashrate_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hashrate_eh_s" not in df.columns:
        return df

    df["hashrate_log"]      = np.log1p(df["hashrate_eh_s"])
    df["hashrate_ma_7d"]    = df["hashrate_eh_s"].rolling(7 * 24).mean()
    df["hashrate_ratio"]    = df["hashrate_eh_s"] / df["hashrate_ma_7d"]
    df["hashrate_pct_chg"]  = df["hashrate_eh_s"].pct_change(24)
    return df


# ---------------------------------------------------------------------------
# Difficulty features
# ---------------------------------------------------------------------------

def add_difficulty_features(df: pd.DataFrame) -> pd.DataFrame:
    if "difficulty" not in df.columns:
        return df

    df["difficulty_log"]    = np.log1p(df["difficulty"])
    df["difficulty_chg"]    = df["difficulty"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Mempool features
# ---------------------------------------------------------------------------

def add_mempool_features(df: pd.DataFrame) -> pd.DataFrame:
    if "mempool_tx_count" not in df.columns:
        return df

    df["mempool_tx_log"]    = np.log1p(df["mempool_tx_count"])
    df["mempool_size_log"]  = np.log1p(df["mempool_vsize"]) if "mempool_vsize" in df.columns else np.nan
    df["fee_pressure"]      = df.get("fee_fastest_sat_vb", 0) / (df.get("fee_economy_sat_vb", 1) + 1e-8)
    return df


# ---------------------------------------------------------------------------
# NVT-proxy (network value to transactions ratio)
# ---------------------------------------------------------------------------

def add_nvt_proxy(df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """
    NVT proxy = market cap / tx volume proxy.
    Requires close price aligned to the same index.
    High NVT → overvalued, Low NVT → undervalued.
    """
    if "n_tx" not in df.columns:
        return df

    price_aligned = price.reindex(df.index, method="ffill")
    df["nvt_proxy"] = price_aligned / (df["n_tx"].replace(0, np.nan))
    df["nvt_proxy_log"] = np.log1p(df["nvt_proxy"])
    df["nvt_ma_30d"] = df["nvt_proxy"].rolling(30 * 24).mean()
    df["nvt_ratio"]  = df["nvt_proxy"] / df["nvt_ma_30d"]
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def compute_onchain_features(
    base_df: pd.DataFrame,
    price_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Load on-chain data, compute features, and align to base_df index.

    Parameters
    ----------
    base_df      : OHLCV DataFrame — defines the target time index
    price_series : close price Series for NVT calculation (optional)

    Returns
    -------
    DataFrame of on-chain features aligned to base_df index
    """
    onchain = load_onchain()
    if onchain.empty:
        logger.warning("No on-chain data available — skipping on-chain features.")
        return pd.DataFrame(index=base_df.index)

    onchain = ensure_utc_index(onchain)

    onchain = add_hashrate_features(onchain)
    onchain = add_difficulty_features(onchain)
    onchain = add_mempool_features(onchain)

    if price_series is not None:
        onchain = add_nvt_proxy(onchain, price_series)

    # Keep only engineered feature columns
    drop_raw = ["hashrate_eh_s", "difficulty", "mempool_tx_count",
                "mempool_vsize", "n_tx", "total_btc_sent",
                "block_count", "hash_rate_gh_s"]
    onchain = onchain.drop(columns=[c for c in drop_raw if c in onchain.columns])

    # Align to hourly OHLCV index via forward-fill (on-chain updates daily)
    aligned = align_to_index(base_df, onchain, method="ffill", limit=48)

    logger.info(f"On-chain features: {aligned.shape[1]} columns aligned to {len(aligned):,} rows")
    return aligned


if __name__ == "__main__":
    from utils.logger import setup_logger
    from ingestion.fetch_ohlcv import load_ohlcv
    setup_logger()
    ohlcv = load_ohlcv("1h")
    feats = compute_onchain_features(ohlcv, price_series=ohlcv["close"])
    print(feats.shape)
    print(feats.tail())