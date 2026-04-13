"""
features/macro.py — Macro features: fear/greed index, funding rates.
All free, no API keys required.
"""

import requests
import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.time_utils import ensure_utc_index, align_to_index
from utils.storage import append_parquet, parquet_exists, load_parquet
from config import RAW_DIR

logger = get_logger()

MACRO_DIR  = RAW_DIR / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT    = 15


# ---------------------------------------------------------------------------
# Fear & Greed Index (alternative.me — free, no key)
# ---------------------------------------------------------------------------

def fetch_fear_greed(limit: int = 365) -> pd.DataFrame:
    """
    Fetch historical Fear & Greed Index from alternative.me.
    Returns DataFrame with UTC DatetimeIndex and columns:
        fng_value (0-100), fng_class (Fear/Greed/etc.)
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return pd.DataFrame()

    rows = [
        {
            "datetime":  pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"),
            "fng_value": int(d["value"]),
            "fng_class": d["value_classification"],
        }
        for d in data
    ]

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    logger.info(f"Fear & Greed: {len(df)} days fetched")
    return df


def fetch_and_store_fear_greed() -> pd.DataFrame:
    path = MACRO_DIR / "fear_greed.parquet"
    df = fetch_fear_greed(limit=500)
    if not df.empty:
        append_parquet(df, path)
    return load_parquet(path) if parquet_exists(path) else df


# ---------------------------------------------------------------------------
# Funding rates (Binance perpetual — free public endpoint)
# ---------------------------------------------------------------------------

def fetch_funding_rates(symbol: str = "BTCUSDT", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance perpetual futures.
    Funding rate > 0 → longs pay shorts (bullish excess)
    Funding rate < 0 → shorts pay longs (bearish excess)
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}

    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"Funding rate fetch failed: {e}")
        return pd.DataFrame()

    rows = [
        {
            "datetime":     pd.Timestamp(int(d["fundingTime"]), unit="ms", tz="UTC"),
            "funding_rate": float(d["fundingRate"]),
        }
        for d in data
    ]

    df = pd.DataFrame(rows).set_index("datetime").sort_index()

    # Derived features
    df["funding_rate_ma_7d"] = df["funding_rate"].rolling(7 * 3).mean()  # 3 per day
    df["funding_cumulative"]  = df["funding_rate"].cumsum()
    df["funding_z"]           = (
        (df["funding_rate"] - df["funding_rate"].rolling(30 * 3).mean())
        / df["funding_rate"].rolling(30 * 3).std()
    )

    logger.info(f"Funding rates: {len(df)} records")
    return df


def fetch_and_store_funding() -> pd.DataFrame:
    path = MACRO_DIR / "funding_rates.parquet"
    df = fetch_funding_rates()
    if not df.empty:
        append_parquet(df, path)
    return load_parquet(path) if parquet_exists(path) else df


# ---------------------------------------------------------------------------
# Open interest (Binance — free public endpoint)
# ---------------------------------------------------------------------------

def fetch_open_interest(symbol: str = "BTCUSDT", period: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance futures.
    Rising OI + rising price = bullish confirmation.
    Rising OI + falling price = bearish pressure.
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}

    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"Open interest fetch failed: {e}")
        return pd.DataFrame()

    rows = [
        {
            "datetime":      pd.Timestamp(int(d["timestamp"]), unit="ms", tz="UTC"),
            "open_interest": float(d["sumOpenInterest"]),
            "oi_value_usd":  float(d["sumOpenInterestValue"]),
        }
        for d in data
    ]

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df["oi_pct_chg"]   = df["open_interest"].pct_change()
    df["oi_ma_24h"]    = df["open_interest"].rolling(24).mean()
    df["oi_ratio"]     = df["open_interest"] / df["oi_ma_24h"]

    logger.info(f"Open interest: {len(df)} records")
    return df


def fetch_and_store_oi() -> pd.DataFrame:
    path = MACRO_DIR / "open_interest.parquet"
    df = fetch_open_interest()
    if not df.empty:
        append_parquet(df, path)
    return load_parquet(path) if parquet_exists(path) else df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def compute_macro_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch all macro features and align to base_df index.
    Returns DataFrame aligned to base_df's UTC DatetimeIndex.
    """
    logger.info("Computing macro features...")

    fng    = fetch_and_store_fear_greed()
    funding = fetch_and_store_funding()
    oi     = fetch_and_store_oi()

    frames = []

    if not fng.empty:
        fng = ensure_utc_index(fng)
        # Encode fng_class as numeric
        class_map = {
            "Extreme Fear": -2, "Fear": -1, "Neutral": 0,
            "Greed": 1, "Extreme Greed": 2,
        }
        fng["fng_encoded"] = fng["fng_class"].map(class_map).fillna(0)
        fng = fng[["fng_value", "fng_encoded"]]
        frames.append(align_to_index(base_df, fng, method="ffill", limit=48))

    if not funding.empty:
        funding = ensure_utc_index(funding)
        frames.append(align_to_index(base_df, funding, method="ffill", limit=8))

    if not oi.empty:
        oi = ensure_utc_index(oi)
        frames.append(align_to_index(base_df, oi, method="ffill", limit=2))

    if not frames:
        logger.warning("No macro features available.")
        return pd.DataFrame(index=base_df.index)

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    combined = combined.ffill()

    logger.info(f"Macro features: {combined.shape[1]} columns | {len(combined):,} rows")
    return combined


if __name__ == "__main__":
    from utils.logger import setup_logger
    from ingestion.fetch_ohlcv import load_ohlcv
    setup_logger()
    ohlcv = load_ohlcv("1h")
    macro = compute_macro_features(ohlcv)
    print(macro.shape)
    print(macro.tail())