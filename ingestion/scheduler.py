"""
ingestion/scheduler.py — Polling loop for live data updates.
Uses `schedule` to run each fetcher on its own interval.
Run this as a background process alongside the model.
"""

import time
import schedule
from utils.logger import get_logger, setup_logger
from ingestion.fetch_ohlcv import fetch_all_timeframes
from ingestion.fetch_onchain import fetch_onchain_snapshot
from ingestion.fetch_news import fetch_and_store_news
from ingestion.fetch_stocktwits import fetch_and_store_stocktwits
from config import (
    FETCH_INTERVAL_MINUTES,
    SENTIMENT_INTERVAL_MINS,
    ONCHAIN_INTERVAL_MINS,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Job wrappers — each catches exceptions so scheduler keeps running
# ---------------------------------------------------------------------------

def job_ohlcv():
    try:
        logger.info("[ SCHEDULER ] Fetching OHLCV...")
        fetch_all_timeframes(incremental=True)
    except Exception as e:
        logger.error(f"OHLCV job failed: {e}")


def job_onchain():
    try:
        logger.info("[ SCHEDULER ] Fetching on-chain snapshot...")
        fetch_onchain_snapshot()
    except Exception as e:
        logger.error(f"On-chain job failed: {e}")


def job_news():
    try:
        logger.info("[ SCHEDULER ] Fetching news...")
        fetch_and_store_news()
    except Exception as e:
        logger.error(f"News job failed: {e}")


def job_stocktwits():
    try:
        logger.info("[ SCHEDULER ] Fetching StockTwits...")
        fetch_and_store_stocktwits()
    except Exception as e:
        logger.error(f"StockTwits job failed: {e}")


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def start_scheduler(run_immediately: bool = True) -> None:
    """
    Register all jobs and start the blocking scheduler loop.

    Parameters
    ----------
    run_immediately : if True, run all jobs once before starting the loop
    """
    # Register intervals
    schedule.every(FETCH_INTERVAL_MINUTES).minutes.do(job_ohlcv)
    schedule.every(ONCHAIN_INTERVAL_MINS).minutes.do(job_onchain)
    schedule.every(SENTIMENT_INTERVAL_MINS).minutes.do(job_news)
    schedule.every(SENTIMENT_INTERVAL_MINS).minutes.do(job_stocktwits)

    logger.info(
        f"Scheduler registered | "
        f"OHLCV every {FETCH_INTERVAL_MINUTES}m | "
        f"On-chain every {ONCHAIN_INTERVAL_MINS}m | "
        f"News/StockTwits every {SENTIMENT_INTERVAL_MINS}m"
    )

    if run_immediately:
        logger.info("Running all jobs immediately on startup...")
        job_ohlcv()
        job_onchain()
        job_news()
        job_stocktwits()

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)     # check every 30 seconds


if __name__ == "__main__":
    setup_logger()
    start_scheduler(run_immediately=True)