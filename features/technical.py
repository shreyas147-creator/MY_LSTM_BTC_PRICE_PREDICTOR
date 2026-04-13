"""
features/technical.py — Technical indicators via pandas-ta-openbb.
Computes RSI, MACD, Bollinger Bands, ATR, OBV, EMA, lag features.
All indicators operate on the 1h OHLCV DataFrame.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from utils.logger import get_logger
from utils.time_utils import ensure_utc_index
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD, OBV_SMOOTH, LAG_PERIODS,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df[f"rsi_{RSI_PERIOD}"] = ta.rsi(df["close"], length=RSI_PERIOD)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None:
        df["macd"]        = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]   = macd.iloc[:, 2]
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    bb = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_mid"]   = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    df[f"atr_{ATR_PERIOD}"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    obv = ta.obv(df["close"], df["volume"])
    if obv is not None:
        df["obv"] = obv
        df["obv_ema"] = ta.ema(obv, length=OBV_SMOOTH)
    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    for period in [9, 21, 50, 200]:
        df[f"ema_{period}"] = ta.ema(df["close"], length=period)
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted average price (resets daily)."""
    vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    if vwap is not None:
        df["vwap"] = vwap
    return df


def add_stoch(df: pd.DataFrame) -> pd.DataFrame:
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    return df


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    adx = ta.adx(df["high"], df["low"], df["close"])
    if adx is not None:
        df["adx"]    = adx.iloc[:, 0]
        df["dmp"]    = adx.iloc[:, 1]   # +DI
        df["dmn"]    = adx.iloc[:, 2]   # -DI
    return df


def add_williams_r(df: pd.DataFrame) -> pd.DataFrame:
    df["willr"] = ta.willr(df["high"], df["low"], df["close"])
    return df


def add_cci(df: pd.DataFrame) -> pd.DataFrame:
    df["cci"] = ta.cci(df["high"], df["low"], df["close"])
    return df


# ---------------------------------------------------------------------------
# Returns and volatility
# ---------------------------------------------------------------------------

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns and rolling realised volatility."""
    df["log_return"]   = np.log(df["close"] / df["close"].shift(1))
    df["return_1h"]    = df["close"].pct_change(1)
    df["return_4h"]    = df["close"].pct_change(4)
    df["return_24h"]   = df["close"].pct_change(24)
    df["return_168h"]  = df["close"].pct_change(168)

    # Realised volatility (rolling std of log returns)
    for window in [24, 72, 168]:
        df[f"realvol_{window}h"] = df["log_return"].rolling(window).std() * np.sqrt(window)

    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag the close price and log return by LAG_PERIODS hours.
    Gives the model direct access to recent price history.
    """
    for lag in LAG_PERIODS:
        df[f"close_lag_{lag}h"]  = df["close"].shift(lag)
        df[f"return_lag_{lag}h"] = df["log_return"].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Price-derived features
# ---------------------------------------------------------------------------

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """High-low range, candle body, upper/lower wicks."""
    df["hl_range"]      = df["high"] - df["low"]
    df["hl_range_pct"]  = df["hl_range"] / df["close"]
    df["body"]          = abs(df["close"] - df["open"])
    df["body_pct"]      = df["body"] / df["close"]
    df["upper_wick"]    = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]    = df[["open", "close"]].min(axis=1) - df["low"]
    df["is_bullish"]    = (df["close"] > df["open"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_ema_24"] = ta.ema(df["volume"], length=24)
    df["volume_ratio"]  = df["volume"] / df["volume_ema_24"]
    df["volume_log"]    = np.log1p(df["volume"])
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all technical indicator functions on an OHLCV DataFrame.
    Input  : DataFrame with open/high/low/close/volume + UTC DatetimeIndex
    Output : Same DataFrame with all indicator columns appended.
    """
    df = ensure_utc_index(df.copy())

    logger.info("Computing technical indicators...")
    df = add_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_ema(df)
    df = add_vwap(df)
    df = add_stoch(df)
    df = add_adx(df)
    df = add_williams_r(df)
    df = add_cci(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_lag_features(df)

    n_before = len(df)
    df.dropna(inplace=True)
    logger.info(
        f"Technical features: {df.shape[1]} columns | "
        f"{len(df):,} rows (dropped {n_before - len(df)} NaN rows)"
    )
    return df


if __name__ == "__main__":
    from utils.logger import setup_logger
    from ingestion.fetch_ohlcv import load_ohlcv
    setup_logger()
    ohlcv = load_ohlcv("1h")
    features = compute_technical_features(ohlcv)
    print(features.shape)
    print(features.columns.tolist())