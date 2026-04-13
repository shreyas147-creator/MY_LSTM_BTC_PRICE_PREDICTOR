"""
evaluation/metrics.py — Trading performance metrics.
Single entry point: compute_metrics(backtest_df) → dict
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger
from config import RISK_FREE_RATE, TRADING_DAYS

logger = get_logger()


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / (TRADING_DAYS * 24)
    std = np.std(excess)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS * 24))


def sortino_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / (TRADING_DAYS * 24)
    downside = excess[excess < 0]
    dd_std = np.std(downside) if len(downside) > 1 else 1e-10
    return float(np.mean(excess) / dd_std * np.sqrt(TRADING_DAYS * 24))


def max_drawdown(pv: pd.Series) -> float:
    if pv is None or len(pv) == 0:
        return 0.0

    peak = pv.cummax()
    dd = (pv - peak) / peak
    return float(dd.min())  # ALWAYS ≤ 0


def calmar_ratio(annual_return: float, max_dd: float) -> float:
    if max_dd < 1e-10:
        return 0.0
    return float(annual_return / max_dd)


def hit_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def avg_win_loss_ratio(returns: np.ndarray) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or len(wins) == 0:
        return 0.0
    return float(wins.mean() / (-losses.mean()))


# ---------------- PUBLIC API FIX ----------------
def win_loss_ratio(returns: np.ndarray) -> float:
    """Backward-compatible alias (DO NOT REMOVE)."""
    return avg_win_loss_ratio(returns)


# ---------------------------------------------------------------------------
# Master compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, *args, **kwargs) -> dict:
    if df is None or len(df) == 0:
        return {}

    if "portfolio_value" not in df:
        raise ValueError("Missing 'portfolio_value'")

    pv = df["portfolio_value"]
    returns = df["returns"] if "returns" in df else pv.pct_change().fillna(0)

    total_return_pct = (pv.iloc[-1] / pv.iloc[0] - 1) * 100

    sharpe  = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    max_dd  = max_drawdown(pv) * 100

    calmar = total_return_pct / abs(max_dd) if max_dd != 0 else 0.0

    n_trades = 0
    if "signal" in df:
        n_trades = int((df["signal"].diff().abs() > 0).sum())

    win_loss = win_loss_ratio(returns)
    pf       = profit_factor(returns)

    metrics = {
        "total_return_pct": float(total_return_pct),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown_pct": float(max_dd),
        "calmar": float(calmar),
        "n_trades": int(n_trades),
        "win_loss_ratio": float(win_loss),
        "profit_factor": float(pf),
    }

    # HARD ASSERT (so this never breaks again silently)
    required = {
        "total_return_pct", "sharpe", "sortino",
        "max_drawdown_pct", "calmar", "n_trades",
        "win_loss_ratio", "profit_factor",
    }

    missing = required - set(metrics.keys())
    if missing:
        raise ValueError(f"compute_metrics missing keys: {missing}")

    return metrics


def _empty_metrics() -> dict:
    return {
        "total_return": 0.0, "total_return_pct": 0.0,
        "annualised_return": 0.0, "final_value": 0.0,
        "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
        "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
        "volatility": 0.0, "hit_rate": 0.0,
        "profit_factor": 0.0, "win_loss_ratio": 0.0,
        "n_trades": 0, "n_periods": 0, "n_days": 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def compare_to_benchmark(strategy_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
    """
    Compare strategy vs buy-and-hold.
    """

    strat_metrics = compute_metrics(strategy_df)
    bench_metrics = compute_metrics(benchmark_df)

    alpha = strat_metrics["total_return_pct"] - bench_metrics["total_return_pct"]

    return {
        "strategy": strat_metrics,
        "benchmark": bench_metrics,
        "alpha_return_pct": float(alpha),
    }


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------

def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}

    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    summary = {}

    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals))

    return summary