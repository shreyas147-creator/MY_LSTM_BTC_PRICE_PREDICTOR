"""
ingestion/fetch_news.py — Crypto news via RSS feeds (feedparser).
Sources: CoinDesk, CoinTelegraph, Bitcoin Magazine, Decrypt.
No API keys required.
"""

import json
import hashlib
import feedparser
import pandas as pd
from datetime import timezone
from email.utils import parsedate_to_datetime
from utils.logger import get_logger
from utils.storage import append_parquet, parquet_exists, load_parquet
from utils.time_utils import now_utc
from config import RSS_FEEDS, NEWS_DIR

logger = get_logger()


# ---------------------------------------------------------------------------
# Parse a single RSS feed
# ---------------------------------------------------------------------------

def _parse_feed(url: str) -> list[dict]:
    """
    Fetch and parse a single RSS feed URL.
    Returns list of article dicts.
    """
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning(f"Failed to parse feed {url}: {e}")
        return []

    articles = []
    for entry in feed.entries:
        # Parse published date
        try:
            pub_dt = parsedate_to_datetime(entry.get("published", ""))
            pub_ts = pd.Timestamp(pub_dt).tz_convert("UTC")
        except Exception:
            pub_ts = now_utc()

        title   = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        link    = entry.get("link", "").strip()

        # Deduplicate by content hash
        content_hash = hashlib.md5(f"{title}{link}".encode()).hexdigest()[:12]

        articles.append({
            "datetime":     pub_ts,
            "source":       feed.feed.get("title", url),
            "title":        title,
            "summary":      summary[:500],   # cap at 500 chars for storage
            "link":         link,
            "content_hash": content_hash,
        })

    logger.debug(f"Parsed {len(articles)} articles from {url}")
    return articles


# ---------------------------------------------------------------------------
# Fetch all feeds
# ---------------------------------------------------------------------------

def fetch_all_news(feeds: list[str] = None) -> pd.DataFrame:
    """
    Fetch articles from all configured RSS feeds.
    Deduplicates by content_hash.
    Returns DataFrame sorted by datetime (UTC).
    """
    if feeds is None:
        feeds = RSS_FEEDS

    all_articles = []
    for url in feeds:
        articles = _parse_feed(url)
        all_articles.extend(articles)

    if not all_articles:
        logger.warning("No articles fetched from any feed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df = df.set_index("datetime").sort_index()
    df = df[~df["content_hash"].duplicated(keep="first")]

    logger.info(f"Fetched {len(df)} unique articles from {len(feeds)} feeds")
    return df


# ---------------------------------------------------------------------------
# Store / update
# ---------------------------------------------------------------------------

def fetch_and_store_news() -> pd.DataFrame:
    """
    Fetch latest news and append to news parquet.
    Deduplicates against existing stored articles.
    """
    path = NEWS_DIR / "news.parquet"
    new_df = fetch_all_news()

    if new_df.empty:
        return load_parquet(path) if parquet_exists(path) else pd.DataFrame()

    if parquet_exists(path):
        existing = load_parquet(path, verbose=False)
        existing_hashes = set(existing["content_hash"].values)
        new_df = new_df[~new_df["content_hash"].isin(existing_hashes)]
        logger.info(f"New articles after dedup: {len(new_df)}")

    if not new_df.empty:
        append_parquet(new_df, path)

    return load_parquet(path)


def load_news(
    since: pd.Timestamp = None,
    until: pd.Timestamp = None,
) -> pd.DataFrame:
    """Load stored news, optionally filtered by date range."""
    path = NEWS_DIR / "news.parquet"
    if not parquet_exists(path):
        return pd.DataFrame()

    df = load_parquet(path, verbose=False)
    if since is not None:
        df = df[df.index >= since]
    if until is not None:
        df = df[df.index <= until]
    return df


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    df = fetch_and_store_news()
    print(f"Total stored articles: {len(df)}")
    print(df[["source", "title"]].tail(10))