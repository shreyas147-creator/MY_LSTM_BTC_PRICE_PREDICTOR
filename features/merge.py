"""
features/merge.py — Joins all feature tables into one model-ready matrix.
Single entry point: build_feature_matrix() → saves features.parquet + labels.parquet
"""

import pandas as pd
from utils.logger import get_logger
from utils.storage import save_parquet, load_parquet, parquet_exists
from utils.time_utils import ensure_utc_index, sort_index
from features.technical import compute_technical_features
from features.onchain_features import compute_onchain_features
from features.sentiment import compute_sentiment_features
from features.macro import compute_macro_features
from features.labels import compute_labels, get_label_cols
from ingestion.fetch_ohlcv import load_ohlcv
from config import FEATURES_PATH, LABELS_PATH, FORWARD_HOURS

logger = get_logger()


# ---------------------------------------------------------------------------
# Build full feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    timeframe: str = "1h",
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the complete feature matrix from all sources.

    Steps:
      1. Load OHLCV
      2. Compute technical indicators
      3. Add labels (forward returns, direction, regimes)
      4. Align on-chain features
      5. Align sentiment features
      6. Align macro features
      7. Drop NaN rows from forward-label window
      8. Save features.parquet + labels.parquet

    Returns
    -------
    (features_df, labels_df) — features without label cols, labels only
    """
    if not force_rebuild and parquet_exists(FEATURES_PATH) and parquet_exists(LABELS_PATH):
        logger.info("Loading cached feature matrix...")
        return load_parquet(FEATURES_PATH), load_parquet(LABELS_PATH)

    logger.info("Building feature matrix from scratch...")

    # --- 1. OHLCV base ---
    ohlcv = load_ohlcv(timeframe)
    if ohlcv.empty:
        raise RuntimeError(f"No OHLCV data found for {timeframe}. Run fetch_ohlcv first.")

    # --- 2. Technical features ---
    df = compute_technical_features(ohlcv)

    # --- 3. Labels ---
    df = compute_labels(df)

    # --- 4. On-chain ---
    onchain = compute_onchain_features(df, price_series=df["close"])
    if not onchain.empty:
        df = df.join(onchain, how="left", rsuffix="_oc")

    # --- 5. Sentiment ---
    sentiment = compute_sentiment_features(df)
    if not sentiment.empty:
        df = df.join(sentiment, how="left", rsuffix="_sent")

    # --- 6. Macro ---
    macro = compute_macro_features(df)
    if not macro.empty:
        df = df.join(macro, how="left", rsuffix="_macro")

    # --- 7. Forward-fill any remaining gaps, drop NaN label rows ---
    label_cols = list(get_label_cols(FORWARD_HOURS).values())
    feature_cols = [c for c in df.columns if c not in label_cols]

    df[feature_cols] = df[feature_cols].ffill()
    df.dropna(subset=[get_label_cols()["regression"]], inplace=True)

    # Final sort and dedup
    df = sort_index(ensure_utc_index(df))
    df = df[~df.index.duplicated(keep="last")]

    # --- 8. Split and save ---
    features = df[feature_cols]
    labels   = df[label_cols]

    save_parquet(features, FEATURES_PATH)
    save_parquet(labels,   LABELS_PATH)

    logger.info(
        f"Feature matrix built: {features.shape} | "
        f"Labels: {labels.shape} | "
        f"Date range: {df.index[0]} → {df.index[-1]}"
    )
    return features, labels


# ---------------------------------------------------------------------------
# Quick load helpers
# ---------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    return load_parquet(FEATURES_PATH)


def load_labels() -> pd.DataFrame:
    return load_parquet(LABELS_PATH)


def load_feature_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_features(), load_labels()


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    features, labels = build_feature_matrix(force_rebuild=True)
    print(f"Features : {features.shape}")
    print(f"Labels   : {labels.shape}")
    print(f"Feature cols: {features.columns.tolist()}")
    print(f"Label cols  : {labels.columns.tolist()}")