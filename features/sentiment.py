"""
features/sentiment.py — Sentiment feature engineering.
Two sources:
  1. FinBERT — local GPU inference on news headlines
  2. StockTwits — pre-labelled bullish/bearish aggregation
Outputs a combined hourly sentiment index aligned to OHLCV.
"""

import numpy as np
import pandas as pd
import torch
from utils.logger import get_logger
from utils.time_utils import ensure_utc_index, align_to_index
from utils.gpu import get_device, clear_vram
from ingestion.fetch_news import load_news
from ingestion.fetch_stocktwits import load_stocktwits_sentiment
from config import (
    FINBERT_MODEL, SENTIMENT_BATCH, SENTIMENT_MAX_LEN,
    SENTIMENT_WINDOW, PROCESSED_DIR,
)

logger = get_logger()

SENTIMENT_PATH = PROCESSED_DIR / "sentiment_scores.parquet"


# ---------------------------------------------------------------------------
# FinBERT loader (lazy — only loads when first called)
# ---------------------------------------------------------------------------

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline
        device = 0 if get_device().type == "cuda" else -1
        logger.info(f"Loading FinBERT on {'GPU' if device == 0 else 'CPU'}...")
        _finbert_pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=device,
            truncation=True,
            max_length=SENTIMENT_MAX_LEN,
            batch_size=SENTIMENT_BATCH,
        )
        logger.info("FinBERT loaded.")
    except Exception as e:
        logger.error(f"Failed to load FinBERT: {e}")
        _finbert_pipeline = None

    return _finbert_pipeline


# ---------------------------------------------------------------------------
# Score text with FinBERT
# ---------------------------------------------------------------------------

def score_texts(texts: list[str]) -> list[float]:
    """
    Score a list of texts with FinBERT.
    Returns list of floats in [-1, +1]:
        positive → +score
        negative → -score
        neutral  → 0
    """
    pipe = _get_finbert()
    if pipe is None or not texts:
        return [0.0] * len(texts)

    # Clean inputs
    clean = [str(t)[:512] if t else "" for t in texts]

    try:
        results = pipe(clean)
    except Exception as e:
        logger.warning(f"FinBERT inference failed: {e}")
        return [0.0] * len(clean)

    scores = []
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Score news DataFrame
# ---------------------------------------------------------------------------

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FinBERT on news titles + summaries.
    Returns DataFrame with datetime index and finbert_score column.
    """
    if news_df.empty:
        return pd.DataFrame()

    texts = (news_df["title"].fillna("") + ". " + news_df["summary"].fillna("")).tolist()
    logger.info(f"Scoring {len(texts)} news articles with FinBERT...")

    scores = score_texts(texts)
    clear_vram()

    result = pd.DataFrame({
        "finbert_score": scores,
        "source": news_df["source"].values,
    }, index=news_df.index)

    return result


# ---------------------------------------------------------------------------
# Aggregate to hourly sentiment index
# ---------------------------------------------------------------------------

def aggregate_news_sentiment(
    scored_df: pd.DataFrame,
    freq: str = SENTIMENT_WINDOW,
) -> pd.DataFrame:
    """
    Resample FinBERT scores to hourly sentiment index.
    """
    if scored_df.empty:
        return pd.DataFrame()

    agg = scored_df["finbert_score"].resample(freq).agg(
        finbert_mean="mean",
        finbert_std="std",
        finbert_count="count",
        finbert_positive=lambda x: (x > 0.1).sum(),
        finbert_negative=lambda x: (x < -0.1).sum(),
    )
    agg["finbert_std"]  = agg["finbert_std"].fillna(0)
    agg["finbert_bull_ratio"] = agg["finbert_positive"] / agg["finbert_count"].replace(0, np.nan)
    return agg


# ---------------------------------------------------------------------------
# Combine FinBERT + StockTwits
# ---------------------------------------------------------------------------

def compute_sentiment_features(
    base_df: pd.DataFrame,
    force_rescore: bool = False,
) -> pd.DataFrame:
    """
    Build combined sentiment feature DataFrame aligned to base_df index.

    Sources:
      - FinBERT scores on news headlines
      - StockTwits pre-labelled sentiment aggregation

    Parameters
    ----------
    base_df       : OHLCV DataFrame — defines target time index
    force_rescore : re-run FinBERT even if cached scores exist

    Returns
    -------
    DataFrame with sentiment columns aligned to base_df index
    """
    from utils.storage import save_parquet, load_parquet, parquet_exists

    # --- FinBERT on news ---
    if parquet_exists(SENTIMENT_PATH) and not force_rescore:
        logger.info("Loading cached FinBERT sentiment scores...")
        scored = load_parquet(SENTIMENT_PATH, verbose=False)
    else:
        news = load_news()
        if not news.empty:
            scored = score_news(news)
            save_parquet(scored, SENTIMENT_PATH)
        else:
            scored = pd.DataFrame()

    finbert_agg = aggregate_news_sentiment(scored) if not scored.empty else pd.DataFrame()

    # --- StockTwits ---
    st_sentiment = load_stocktwits_sentiment(freq=SENTIMENT_WINDOW)

    # --- Combine ---
    frames = []
    if not finbert_agg.empty:
        frames.append(ensure_utc_index(finbert_agg))
    if not st_sentiment.empty:
        frames.append(ensure_utc_index(st_sentiment))

    if not frames:
        logger.warning("No sentiment data available.")
        return pd.DataFrame(index=base_df.index)

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    combined = combined.ffill()

    # Combined score — weighted average of both signals
    cols = []
    if "finbert_mean" in combined.columns:
        cols.append(combined["finbert_mean"].fillna(0))
    if "sentiment_score" in combined.columns:
        cols.append(combined["sentiment_score"].fillna(0))

    if cols:
        combined["combined_sentiment"] = sum(cols) / len(cols)

    # Align to OHLCV index
    aligned = align_to_index(base_df, combined, method="ffill", limit=24)
    logger.info(f"Sentiment features: {aligned.shape[1]} columns | {len(aligned):,} rows")
    return aligned


if __name__ == "__main__":
    from utils.logger import setup_logger
    from ingestion.fetch_ohlcv import load_ohlcv
    setup_logger()
    ohlcv = load_ohlcv("1h")
    sent = compute_sentiment_features(ohlcv)
    print(sent.shape)
    print(sent.tail())