"""
ingestion/fetch_ohlcv.py — Fetch BTC OHLCV from Binance via ccxt.
Stores per-timeframe parquet files. Supports initial full fetch
and incremental updates (only fetches missing bars).
"""

import time
import ccxt
import pandas as pd
from pathlib import Path
from utils.logger import get_logger
from utils.storage import append_parquet, load_parquet, parquet_exists, SQLiteStore
from utils.time_utils import ms_to_utc, utc_to_ms, days_ago, ensure_utc_index
from config import (
    SYMBOL_CCXT, TIMEFRAMES, HISTORY_DAYS, OHLCV_DIR, DATA_DIR
)

logger = get_logger()

STORE = SQLiteStore(DATA_DIR / "fetch_state.db")
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Exchange init
# ---------------------------------------------------------------------------

def get_exchange() -> ccxt.binance:
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return exchange


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_ohlcv_since(
    exchange: ccxt.binance,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch all OHLCV bars from since_ms to now, paginating automatically.
    Returns DataFrame with UTC DatetimeIndex.
    """
    all_bars = []
    current_since = since_ms

    while True:
        try:
            bars = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit,
            )
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e} — retrying in 5s")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            break

        if not bars:
            break

        all_bars.extend(bars)
        last_ts = bars[-1][0]

        if len(bars) < limit:
            break

        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_bars:
        logger.warning(f"No bars returned for {symbol} {timeframe}")
        return pd.DataFrame(columns=OHLCV_COLS)

    df = pd.DataFrame(all_bars, columns=["timestamp"] + OHLCV_COLS)
    df.index = df["timestamp"].apply(ms_to_utc)
    df.index.name = "datetime"
    df = df[OHLCV_COLS].astype(float)
    logger.info(f"Fetched {len(df):,} bars | {symbol} {timeframe} | {df.index[0]} → {df.index[-1]}")
    return df


# ---------------------------------------------------------------------------
# Full initial fetch
# ---------------------------------------------------------------------------

def fetch_full(
    timeframe: str = "1h",
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Fetch full history from history_days ago to now."""
    exchange = get_exchange()
    since_ms = utc_to_ms(days_ago(history_days))
    df = fetch_ohlcv_since(exchange, SYMBOL_CCXT, timeframe, since_ms)

    if df.empty:
        return df

    path = OHLCV_DIR / f"btc_{timeframe}.parquet"
    append_parquet(df, path)
    STORE.log_fetch("binance_ohlcv", timeframe, str(df.index[-1]), len(df))
    return df


# ---------------------------------------------------------------------------
# Incremental update — only fetch new bars
# ---------------------------------------------------------------------------

def fetch_incremental(timeframe: str = "1h") -> pd.DataFrame:
    """
    Fetch only bars newer than the last stored timestamp.
    Creates a full fetch if no data exists yet.
    """
    path = OHLCV_DIR / f"btc_{timeframe}.parquet"

    if not parquet_exists(path):
        logger.info(f"No existing data for {timeframe} — running full fetch.")
        return fetch_full(timeframe)

    existing = load_parquet(path, verbose=False)
    existing = ensure_utc_index(existing)
    last_ts = existing.index[-1]
    since_ms = utc_to_ms(last_ts) + 1

    logger.info(f"Incremental fetch {timeframe} from {last_ts}")
    exchange = get_exchange()
    new_df = fetch_ohlcv_since(exchange, SYMBOL_CCXT, timeframe, since_ms)

    if new_df.empty:
        logger.info(f"No new bars for {timeframe}.")
        return existing

    combined = append_parquet(new_df, path)
    STORE.log_fetch("binance_ohlcv", timeframe, str(new_df.index[-1]), len(new_df))
    return combined


# ---------------------------------------------------------------------------
# Fetch all timeframes
# ---------------------------------------------------------------------------

def fetch_all_timeframes(incremental: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch all configured timeframes.
    Returns dict: {timeframe: DataFrame}
    """
    results = {}
    for tf in TIMEFRAMES:
        logger.info(f"--- Fetching {tf} ---")
        if incremental:
            results[tf] = fetch_incremental(tf)
        else:
            results[tf] = fetch_full(tf)
    return results


# ---------------------------------------------------------------------------
# Load cached data
# ---------------------------------------------------------------------------

def load_ohlcv(timeframe: str = "1h") -> pd.DataFrame:
    """Load locally stored OHLCV parquet for a given timeframe."""
    path = OHLCV_DIR / f"btc_{timeframe}.parquet"
    return load_parquet(path)


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    data = fetch_all_timeframes(incremental=False)
    for tf, df in data.items():
        print(f"{tf}: {len(df):,} rows | {df.index[0]} → {df.index[-1]}")