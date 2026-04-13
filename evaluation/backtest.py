"""
evaluation/backtest.py — Walk-forward backtesting engine.
Simulates trading on historical data using model signals.
No lookahead bias — each fold only uses data available at that time.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.time_utils import walkforward_folds
from config import (
    WALKFORWARD_TRAIN_DAYS, WALKFORWARD_VAL_DAYS,
    WALKFORWARD_STEP_DAYS, RL_TRADE_FEE, RL_INITIAL_CAPITAL,
    CONFIDENCE_THRESHOLD,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Single-fold backtest
# ---------------------------------------------------------------------------

def backtest_fold(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = RL_INITIAL_CAPITAL,
    fee: float             = RL_TRADE_FEE,
    confidence: pd.Series  = None,
    threshold: float       = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:

    if prices is None or signals is None or len(signals) == 0:
        return pd.DataFrame()

    prices  = prices.reindex(signals.index).ffill()
    signals = signals.copy()

    if confidence is not None:
        conf = confidence.reindex(signals.index).fillna(0)
        signals[conf < threshold] = 0

    cash     = float(initial_capital)
    position = 0.0
    records  = []

    for ts, signal in signals.items():
        price = prices.loc[ts]

        if not np.isfinite(price) or price <= 0:
            continue

        # --- EXECUTION ---
        if signal == 1 and position == 0 and cash > 0:
            position = (cash * (1 - fee)) / price
            cash = 0.0

        elif signal == -1 and position > 0:
            cash = position * price * (1 - fee)
            position = 0.0

        pv = cash + position * price

        records.append({
            "datetime": ts,
            "price": float(price),
            "signal": int(signal),
            "position": float(position),
            "cash": float(cash),
            "portfolio_value": float(pv),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("datetime")

    # --- FORCE LIQUIDATION AT END ---
    if df["position"].iloc[-1] > 0:
        final_price = df["price"].iloc[-1]
        final_cash  = df["position"].iloc[-1] * final_price * (1 - fee)
        df.iloc[-1, df.columns.get_loc("cash")] = final_cash
        df.iloc[-1, df.columns.get_loc("position")] = 0.0
        df.iloc[-1, df.columns.get_loc("portfolio_value")] = final_cash

    # --- RETURNS ---
    df["returns"] = df["portfolio_value"].pct_change().fillna(0)

    peak = df["portfolio_value"].cummax()
    df["drawdown"] = (df["portfolio_value"] - peak) / peak

    # --- HARD CONTRACT ENFORCEMENT ---
    required = ["portfolio_value", "returns", "drawdown"]
    if any(col not in df.columns for col in required):
        raise ValueError(f"Backtest output missing required columns: {required}")

    return df


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walkforward_backtest(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    prices: pd.Series,
    model_fn,           # callable(train_X, train_y, val_X) -> signal array
    label_col: str      = "direction_24h",
    train_days: int     = WALKFORWARD_TRAIN_DAYS,
    val_days: int       = WALKFORWARD_VAL_DAYS,
    step_days: int      = WALKFORWARD_STEP_DAYS,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run walk-forward backtest across multiple folds.

    Parameters
    ----------
    features  : full feature DataFrame
    labels    : full labels DataFrame
    prices    : close price Series
    model_fn  : function(train_X, train_y, val_X) → np.ndarray of signals
    label_col : which label column to use as training target

    Returns
    -------
    (combined_results_df, fold_metrics_list)
    """
    folds = walkforward_folds(
        features,
        train_days=train_days,
        val_days=val_days,
        step_days=step_days,
    )

    all_results  = []
    fold_metrics = []

    for fold in folds:
        fid = fold["fold_id"]
        train_df = fold["train"]
        val_df = fold["val"]

        train_X = train_df.values
        train_y = labels.loc[train_df.index, label_col].values
        val_X = val_df.values

        logger.info(
            f"Fold {fid} | train={len(train_df):,} | val={len(val_df):,} | "
            f"{train_df.index[0].date()} → {val_df.index[-1].date()}"
        )

        # ---------------- MODEL INFERENCE ----------------
        try:
            signals_arr = model_fn(train_X, train_y, val_X)
        except Exception as e:
            logger.error(f"Fold {fid} model_fn failed: {e}")
            continue

        if signals_arr is None:
            logger.error(f"Fold {fid}: model returned None")
            continue

        signals_arr = np.asarray(signals_arr).reshape(-1)

        if len(signals_arr) != len(val_df):
            logger.error(
                f"Fold {fid}: signal length mismatch "
                f"({len(signals_arr)} vs {len(val_df)})"
            )
            continue

        # enforce discrete signals {-1, 0, 1}
        signals_arr = np.where(signals_arr > 0, 1,
                               np.where(signals_arr < 0, -1, 0))

        signals = pd.Series(signals_arr, index=val_df.index, name="signal")

        fold_prices = prices.reindex(val_df.index).ffill()

        if fold_prices.isna().all():
            logger.error(f"Fold {fid}: all prices NaN after reindex")
            continue

        # ---------------- BACKTEST ----------------
        result = backtest_fold(fold_prices, signals)

        if result is None or result.empty:
            logger.error(f"Fold {fid}: empty backtest result")
            continue

        required_cols = {"portfolio_value", "returns", "drawdown"}
        if not required_cols.issubset(result.columns):
            logger.error(
                f"Fold {fid}: missing required columns {required_cols}"
            )
            continue

        if not np.isfinite(result["portfolio_value"]).all():
            logger.error(f"Fold {fid}: non-finite portfolio values detected")
            continue

        result["fold_id"] = fid
        all_results.append(result)

        # ---------------- METRICS ----------------
        from evaluation.metrics import compute_metrics

        try:
            m = compute_metrics(result)
        except Exception as e:
            logger.error(f"Fold {fid}: compute_metrics failed: {e}")
            continue

        if "total_return" not in m:
            logger.error(f"Fold {fid}: metrics missing total_return")
            continue

        m["fold_id"] = fid
        fold_metrics.append(m)

        logger.info(
            f"Fold {fid} | Sharpe={m.get('sharpe', 0):.2f} | "
            f"Return={m.get('total_return_pct', 0):.1f}% | "
            f"MaxDD={m.get('max_drawdown_pct', 0):.1f}%"
        )

        m = compute_metrics(result)
        m["fold_id"] = fid
        fold_metrics.append(m)
        logger.info(
            f"Fold {fid} | Sharpe={m.get('sharpe', 0):.2f} | "
            f"Return={m.get('total_return_pct', 0):.1f}% | "
            f"MaxDD={m.get('max_drawdown_pct', 0):.1f}%"
        )

    if not all_results:
        return pd.DataFrame(), fold_metrics

    combined = pd.concat(all_results).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    logger.info(
        f"Walk-forward complete | {len(folds)} folds | "
        f"avg Sharpe={np.mean([m.get('sharpe', 0) for m in fold_metrics]):.2f}"
    )
    return combined, fold_metrics


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------

def buy_and_hold(
    prices: pd.Series,
    initial_capital: float = RL_INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Benchmark: buy at start, hold until end."""
    btc  = initial_capital / prices.iloc[0]
    pv   = prices * btc
    df   = pd.DataFrame({"portfolio_value": pv, "price": prices})
    df["returns"]  = df["portfolio_value"].pct_change().fillna(0)
    df["cum_ret"]  = (1 + df["returns"]).cumprod() - 1
    peak           = df["portfolio_value"].cummax()
    df["drawdown"] = (df["portfolio_value"] - peak) / peak
    return df