"""
evaluator.py — Trade performance metrics and daily summary.

Calculates:
  - Total P&L, Win Rate, Profit Factor
  - Sharpe Ratio (annualised)
  - Max Drawdown
  - Persists results to performance_summary table

Scheduled via Celery Beat (see celery_app.py).

DISCLAIMER: For educational use only. All financial risk is assumed by the user.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

from config.settings import get_settings
from trader_bot.risk_manager import get_engine

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute performance metrics from a trades DataFrame.

    Args:
        df: DataFrame with columns [pnl] (float).

    Returns:
        dict of computed metrics.
    """
    if df.empty or "pnl" not in df.columns:
        return {}

    pnl = df["pnl"].dropna()
    total_trades = len(pnl)
    winning = pnl[pnl > 0]
    losing = pnl[pnl <= 0]

    win_rate = len(winning) / total_trades * 100 if total_trades else 0.0
    total_pnl = float(pnl.sum())

    gross_profit = float(winning.sum())
    gross_loss = abs(float(losing.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    sharpe = _sharpe_ratio(pnl)
    max_dd = _max_drawdown(pnl)

    return {
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 2),
    }


def _sharpe_ratio(pnl: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Annualised Sharpe Ratio (assuming 252 trading days)."""
    if len(pnl) < 2:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = pnl - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def _max_drawdown(pnl: pd.Series) -> float:
    """Maximum drawdown as an absolute INR value."""
    cumulative = pnl.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Database fetch
# ---------------------------------------------------------------------------

def fetch_trades(trade_date: Optional[date] = None) -> pd.DataFrame:
    """
    Fetch closed trades from PostgreSQL.
    If trade_date is None, fetches all time.
    """
    if trade_date:
        sql = text("SELECT * FROM trades WHERE status = 'CLOSED' AND trade_date = :d ORDER BY created_at")
        params = {"d": trade_date}
    else:
        sql = text("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY created_at")
        params = {}

    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        return df
    except Exception as exc:
        logger.error("Failed to fetch trades: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Persist summary
# ---------------------------------------------------------------------------

def save_summary(metrics: dict, summary_date: date) -> None:
    """Upsert daily performance summary into the database."""
    if not metrics:
        return
    sql = text(
        """
        INSERT INTO performance_summary
            (summary_date, total_trades, winning_trades, losing_trades,
             total_pnl, win_rate, profit_factor, sharpe_ratio, max_drawdown)
        VALUES
            (:summary_date, :total_trades, :winning_trades, :losing_trades,
             :total_pnl, :win_rate, :profit_factor, :sharpe_ratio, :max_drawdown)
        ON CONFLICT (summary_date) DO UPDATE SET
            total_trades   = EXCLUDED.total_trades,
            winning_trades = EXCLUDED.winning_trades,
            losing_trades  = EXCLUDED.losing_trades,
            total_pnl      = EXCLUDED.total_pnl,
            win_rate       = EXCLUDED.win_rate,
            profit_factor  = EXCLUDED.profit_factor,
            sharpe_ratio   = EXCLUDED.sharpe_ratio,
            max_drawdown   = EXCLUDED.max_drawdown
        """
    )
    try:
        with get_engine().connect() as conn:
            conn.execute(sql, {"summary_date": summary_date, **metrics})
            conn.commit()
        logger.info("Performance summary saved for %s.", summary_date)
    except Exception as exc:
        logger.error("Failed to save summary: %s", exc)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(target_date: Optional[date] = None) -> dict:
    """
    Run full evaluation for a given date (defaults to today).
    Returns computed metrics dict.
    """
    target_date = target_date or date.today()
    logger.info("Running evaluation for %s", target_date)

    df = fetch_trades(target_date)
    if df.empty:
        logger.info("No closed trades found for %s.", target_date)
        return {}

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    metrics = compute_metrics(df)
    save_summary(metrics, target_date)

    logger.info("Evaluation complete: %s", metrics)
    return metrics
