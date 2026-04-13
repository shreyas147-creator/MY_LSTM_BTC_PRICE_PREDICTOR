"""
ingestion/fetch_onchain.py — On-chain data from mempool.space + blockchain.info.
Both APIs are free with no key required.
Fetches: hash rate, difficulty, tx count, mempool size, fee rates, block time.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from utils.logger import get_logger
from utils.storage import append_parquet, load_parquet, parquet_exists
from utils.time_utils import now_utc
from config import MEMPOOL_BASE_URL, BLOCKCHAIN_BASE_URL, ONCHAIN_DIR

logger = get_logger()

TIMEOUT = 15  # seconds per request


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, retries: int = 3) -> dict | list | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning(f"Request failed ({attempt+1}/{retries}): {url} — {e}")
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# mempool.space endpoints
# ---------------------------------------------------------------------------

def fetch_mempool_stats() -> dict:
    """
    Fetch current mempool stats: tx count, total fees, fee rates.
    Returns a flat dict with a UTC timestamp.
    """
    data = _get(f"{MEMPOOL_BASE_URL}/mempool")
    if not data:
        return {}

    fee_data = _get(f"{MEMPOOL_BASE_URL}/v1/fees/recommended")
    fees = fee_data if fee_data else {}

    return {
        "datetime":             now_utc(),
        "mempool_tx_count":     data.get("count", 0),
        "mempool_vsize":        data.get("vsize", 0),
        "mempool_total_fee":    data.get("total_fee", 0),
        "fee_fastest_sat_vb":   fees.get("fastestFee", 0),
        "fee_30min_sat_vb":     fees.get("halfHourFee", 0),
        "fee_1h_sat_vb":        fees.get("hourFee", 0),
        "fee_economy_sat_vb":   fees.get("economyFee", 0),
    }


def fetch_hashrate_history(days: int = 180) -> pd.DataFrame:
    """
    Fetch historical hash rate + difficulty from mempool.space.
    Returns DataFrame with UTC DatetimeIndex.
    """
    data = _get(f"{MEMPOOL_BASE_URL}/v1/mining/hashrate/{days}d")
    if not data:
        return pd.DataFrame()

    hashrate_rows = [
        {
            "datetime": pd.Timestamp(r["timestamp"], unit="s", tz="UTC"),
            "hashrate_eh_s": r["avgHashrate"] / 1e18,
        }
        for r in data.get("hashrates", [])
    ]

    diff_rows = {
        pd.Timestamp(r["time"], unit="s", tz="UTC"): r["difficulty"]
        for r in data.get("difficulty", [])
    }

    df = pd.DataFrame(hashrate_rows).set_index("datetime")
    df["difficulty"] = df.index.map(diff_rows)
    df = df.ffill()
    logger.info(f"Hash rate history: {len(df)} rows")
    return df


def fetch_block_stats_recent(n_blocks: int = 144) -> pd.DataFrame:
    """
    Fetch per-block stats for the last n_blocks (~1 day at 144 blocks/day).
    Returns: timestamp, tx_count, avg_fee, size, weight per block.
    """
    data = _get(f"{MEMPOOL_BASE_URL}/v1/mining/blocks/fee-rates/1w")
    if not data:
        return pd.DataFrame()

    rows = []
    for block in data[:n_blocks]:
        rows.append({
            "datetime":     pd.Timestamp(block["timestamp"], unit="s", tz="UTC"),
            "block_height": block.get("avgHeight", 0),
            "avg_fee_rate": block.get("avgFee", 0),
            "med_fee_rate": block.get("medFee", 0),
        })

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df


# ---------------------------------------------------------------------------
# blockchain.info endpoints
# ---------------------------------------------------------------------------

def fetch_blockchain_stats() -> dict:
    """
    Fetch aggregate blockchain stats from blockchain.info simple API.
    Returns a snapshot dict with UTC timestamp.
    """
    endpoints = {
        "n_tx":             "n_tx",
        "total_btc_sent":   "totalbc",
        "difficulty":       "getdifficulty",
        "block_count":      "getblockcount",
        "hash_rate_gh_s":   "hashrate",
    }

    stats = {"datetime": now_utc()}
    for key, endpoint in endpoints.items():
        result = _get(f"{BLOCKCHAIN_BASE_URL}/{endpoint}")
        if result is not None:
            try:
                stats[key] = float(result)
            except (TypeError, ValueError):
                stats[key] = None

    return stats


# ---------------------------------------------------------------------------
# Build / update on-chain parquet
# ---------------------------------------------------------------------------

def fetch_and_store_onchain(history_days: int = 180) -> pd.DataFrame:
    """
    Full fetch of on-chain history.
    Saves to data/raw/onchain/onchain_history.parquet
    """
    logger.info("Fetching on-chain history...")
    hashrate_df = fetch_hashrate_history(days=history_days)

    if hashrate_df.empty:
        logger.warning("No on-chain data returned.")
        return pd.DataFrame()

    path = ONCHAIN_DIR / "onchain_history.parquet"
    append_parquet(hashrate_df, path)
    return hashrate_df


def fetch_onchain_snapshot() -> pd.DataFrame:
    """
    Fetch current on-chain snapshot (mempool + blockchain stats).
    Appends a single row to onchain_snapshot.parquet.
    """
    mempool = fetch_mempool_stats()
    chain   = fetch_blockchain_stats()

    row = {**mempool, **{k: v for k, v in chain.items() if k != "datetime"}}
    if not row:
        return pd.DataFrame()

    df = pd.DataFrame([row]).set_index("datetime")
    path = ONCHAIN_DIR / "onchain_snapshot.parquet"
    append_parquet(df, path)
    logger.info(f"On-chain snapshot saved: {df.index[0]}")
    return df


def load_onchain() -> pd.DataFrame:
    """Load stored on-chain history."""
    path = ONCHAIN_DIR / "onchain_history.parquet"
    return load_parquet(path) if parquet_exists(path) else pd.DataFrame()


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    df = fetch_and_store_onchain(history_days=90)
    print(df.tail())
    snap = fetch_onchain_snapshot()
    print(snap)