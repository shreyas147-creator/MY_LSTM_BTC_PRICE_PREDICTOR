"""
ingestion/fetch_stocktwits.py — Crypto sentiment from StockTwits free API.
No API key required. Returns posts with built-in bullish/bearish labels.
Symbol: BTC.X (StockTwits ticker for Bitcoin)
"""

import time
import requests
import pandas as pd
from utils.logger import get_logger
from utils.storage import append_parquet, parquet_exists, load_parquet
from utils.time_utils import now_utc
from config import REDDIT_DIR   # reusing the same raw dir, renamed below

logger = get_logger()

BASE_URL    = "https://api.stocktwits.com/api/2"
SYMBOL      = "BTC.X"
TIMEOUT     = 15
STOCKTWITS_DIR = REDDIT_DIR.parent / "stocktwits"   # data/raw/stocktwits/


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = 60
                logger.warning(f"Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning(f"Request failed ({attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Fetch latest stream
# ---------------------------------------------------------------------------

def fetch_stream(
    symbol: str = SYMBOL,
    max_id: int = None,
) -> tuple[list[dict], int | None]:
    """
    Fetch one page of messages from the StockTwits stream.
    Returns (list of message dicts, cursor_max_id for next page).

    Parameters
    ----------
    max_id : fetch messages older than this id (pagination cursor)
    """
    params = {"limit": 30}   # StockTwits free tier max per request
    if max_id:
        params["max"] = max_id

    data = _get(f"{BASE_URL}/streams/symbol/{symbol}.json", params=params)
    if not data or "messages" not in data:
        return [], None

    messages = data["messages"]
    cursor = data.get("cursor", {})
    next_max_id = cursor.get("max") if not cursor.get("is_last", True) else None

    rows = []
    for msg in messages:
        sentiment = None
        if msg.get("entities", {}).get("sentiment"):
            sentiment = msg["entities"]["sentiment"].get("basic")  # Bullish / Bearish

        rows.append({
            "datetime":     pd.Timestamp(msg["created_at"]).tz_convert("UTC"),
            "msg_id":       msg["id"],
            "body":         msg.get("body", "")[:500],
            "sentiment":    sentiment,           # "Bullish", "Bearish", or None
            "likes":        msg.get("likes", {}).get("total", 0),
            "symbol":       symbol,
        })

    return rows, next_max_id


# ---------------------------------------------------------------------------
# Fetch multiple pages
# ---------------------------------------------------------------------------

def fetch_stocktwits(
    symbol: str = SYMBOL,
    n_pages: int = 5,
    sleep_between: float = 1.5,   # respect free tier rate limit
) -> pd.DataFrame:
    """
    Fetch n_pages of StockTwits messages for a symbol.
    Each page = 30 messages → 5 pages = ~150 messages.

    Returns DataFrame with UTC DatetimeIndex.
    """
    all_rows = []
    max_id = None

    for page in range(n_pages):
        rows, max_id = fetch_stream(symbol=symbol, max_id=max_id)
        if not rows:
            break
        all_rows.extend(rows)
        logger.debug(f"StockTwits page {page+1}: {len(rows)} messages")

        if max_id is None:
            break
        time.sleep(sleep_between)

    if not all_rows:
        logger.warning("No StockTwits messages fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()
    df = df[~df["msg_id"].duplicated(keep="first")]

    n_bull = (df["sentiment"] == "Bullish").sum()
    n_bear = (df["sentiment"] == "Bearish").sum()
    logger.info(
        f"StockTwits {symbol}: {len(df)} messages | "
        f"Bullish={n_bull} | Bearish={n_bear}"
    )
    return df


# ---------------------------------------------------------------------------
# Sentiment aggregation — hourly index
# ---------------------------------------------------------------------------

def aggregate_sentiment(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Aggregate raw StockTwits messages into a hourly sentiment index.

    Columns produced:
        bull_count    — number of bullish posts
        bear_count    — number of bearish posts
        total_count   — total posts
        bull_ratio    — bullish / total (NaN if no posts)
        sentiment_score — bull_ratio * 2 - 1  (range: -1 to +1)
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["is_bull"] = (df["sentiment"] == "Bullish").astype(int)
    df["is_bear"] = (df["sentiment"] == "Bearish").astype(int)

    agg = df.resample(freq).agg(
        bull_count=("is_bull", "sum"),
        bear_count=("is_bear", "sum"),
        total_count=("msg_id", "count"),
    )

    agg["bull_ratio"] = agg["bull_count"] / agg["total_count"].replace(0, float("nan"))
    agg["sentiment_score"] = agg["bull_ratio"] * 2 - 1   # -1 to +1
    agg["sentiment_score"] = agg["sentiment_score"].fillna(0)

    return agg


# ---------------------------------------------------------------------------
# Store / update
# ---------------------------------------------------------------------------

def fetch_and_store_stocktwits(n_pages: int = 5) -> pd.DataFrame:
    """
    Fetch latest StockTwits posts and append to parquet.
    Deduplicates on msg_id.
    """
    STOCKTWITS_DIR.mkdir(parents=True, exist_ok=True)
    path = STOCKTWITS_DIR / "stocktwits_btc.parquet"

    new_df = fetch_stocktwits(n_pages=n_pages)
    if new_df.empty:
        return load_parquet(path) if parquet_exists(path) else pd.DataFrame()

    if parquet_exists(path):
        existing = load_parquet(path, verbose=False)
        existing_ids = set(existing["msg_id"].values)
        new_df = new_df[~new_df["msg_id"].isin(existing_ids)]
        logger.info(f"New messages after dedup: {len(new_df)}")

    if not new_df.empty:
        append_parquet(new_df, path)

    return load_parquet(path)


def load_stocktwits(
    since: pd.Timestamp = None,
    until: pd.Timestamp = None,
) -> pd.DataFrame:
    """Load stored StockTwits data, optionally filtered by date range."""
    STOCKTWITS_DIR.mkdir(parents=True, exist_ok=True)
    path = STOCKTWITS_DIR / "stocktwits_btc.parquet"
    if not parquet_exists(path):
        return pd.DataFrame()

    df = load_parquet(path, verbose=False)
    if since is not None:
        df = df[df.index >= since]
    if until is not None:
        df = df[df.index <= until]
    return df


def load_stocktwits_sentiment(freq: str = "1h") -> pd.DataFrame:
    """Load and return aggregated hourly sentiment index."""
    df = load_stocktwits()
    if df.empty:
        return pd.DataFrame()
    return aggregate_sentiment(df, freq=freq)


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    df = fetch_and_store_stocktwits(n_pages=3)
    print(f"Total stored messages: {len(df)}")
    print(df[["body", "sentiment", "likes"]].tail(10))
    print("\nHourly sentiment index:")
    print(load_stocktwits_sentiment().tail(10))