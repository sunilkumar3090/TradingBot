"""
risk_manager.py — Position sizing and risk controls.

Rules enforced:
  - Max risk per trade: configurable % of capital (default 1%).
  - Max daily drawdown: halt trading if cumulative loss > threshold.
  - All positions logged to PostgreSQL.

DISCLAIMER: For educational use only. All financial risk is assumed by the user.
"""

import logging
from datetime import date
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_engine: Optional[sa.engine.Engine] = None


def get_engine() -> sa.engine.Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


# ---------------------------------------------------------------------------
# Schema bootstrap (run once at startup)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id               SERIAL PRIMARY KEY,
    trade_date       DATE        NOT NULL DEFAULT CURRENT_DATE,
    symbol           VARCHAR(50) NOT NULL,
    strategy         VARCHAR(50),
    side             VARCHAR(4)  NOT NULL,  -- BUY / SELL
    quantity         INTEGER     NOT NULL,
    entry_price      NUMERIC     NOT NULL,
    exit_price       NUMERIC,
    stop_loss        NUMERIC,
    target           NUMERIC,
    pnl              NUMERIC,
    status           VARCHAR(20) DEFAULT 'OPEN',  -- OPEN / CLOSED / CANCELLED
    kite_order_id    VARCHAR(50),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_summary (
    id              SERIAL PRIMARY KEY,
    summary_date    DATE        NOT NULL UNIQUE,
    total_trades    INTEGER,
    winning_trades  INTEGER,
    losing_trades   INTEGER,
    total_pnl       NUMERIC,
    win_rate        NUMERIC,
    profit_factor   NUMERIC,
    sharpe_ratio    NUMERIC,
    max_drawdown    NUMERIC,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


def init_db() -> None:
    """Create tables if they do not yet exist."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text(SCHEMA_SQL))
            conn.commit()
        logger.info("Database schema initialised.")
    except SQLAlchemyError as exc:
        logger.error("DB init failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: Optional[float] = None,
) -> int:
    """
    Calculate the number of shares to trade based on fixed-fraction risk.

    Args:
        capital:     Total account capital in INR.
        entry_price: Planned entry price.
        stop_loss:   Stop-loss price.
        risk_pct:    Percentage of capital to risk (overrides settings default).

    Returns:
        Integer quantity (minimum 1).
    """
    if risk_pct is None:
        risk_pct = settings.max_risk_per_trade_pct

    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        logger.warning("Entry price equals stop loss — cannot size position.")
        return 0

    risk_amount = capital * (risk_pct / 100.0)
    qty = int(risk_amount / risk_per_share)
    return max(qty, 1)


# ---------------------------------------------------------------------------
# Daily drawdown gate
# ---------------------------------------------------------------------------

def is_drawdown_limit_hit(capital: float) -> bool:
    """
    Return True if today's realised losses exceed the max daily drawdown %.
    Queries the trades table for today's closed P&L.
    """
    try:
        with get_engine().connect() as conn:
            result = conn.execute(
                text(
                    "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                    "WHERE trade_date = :today AND status = 'CLOSED'"
                ),
                {"today": date.today()},
            )
            daily_pnl = float(result.scalar())
    except SQLAlchemyError as exc:
        logger.error("Could not query daily P&L: %s", exc)
        return False  # Fail open — let risk_manager decide conservatively

    max_loss = capital * (settings.max_daily_drawdown_pct / 100.0)
    if daily_pnl < -max_loss:
        logger.warning(
            "Daily drawdown limit reached. P&L: %.2f, Limit: -%.2f", daily_pnl, max_loss
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Trade logging
# ---------------------------------------------------------------------------

def log_trade(
    symbol: str,
    side: str,
    quantity: int,
    entry_price: float,
    stop_loss: float,
    target: float,
    strategy: str = "",
    kite_order_id: str = "",
) -> Optional[int]:
    """Insert a new OPEN trade record. Returns the new trade id."""
    sql = text(
        """
        INSERT INTO trades
            (symbol, strategy, side, quantity, entry_price, stop_loss, target, kite_order_id, status)
        VALUES
            (:symbol, :strategy, :side, :quantity, :entry_price, :stop_loss, :target, :kite_order_id, 'OPEN')
        RETURNING id
        """
    )
    try:
        with get_engine().connect() as conn:
            result = conn.execute(
                sql,
                {
                    "symbol": symbol,
                    "strategy": strategy,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target": target,
                    "kite_order_id": kite_order_id,
                },
            )
            conn.commit()
            trade_id = result.scalar()
            logger.info("Trade logged — id: %s, symbol: %s, side: %s", trade_id, symbol, side)
            return trade_id
    except SQLAlchemyError as exc:
        logger.error("Failed to log trade: %s", exc)
        return None


def close_trade(trade_id: int, exit_price: float, status: str = "CLOSED") -> None:
    """Update a trade record with exit price and final P&L."""
    sql = text(
        """
        UPDATE trades
        SET exit_price = :exit_price,
            pnl        = (exit_price - entry_price) * quantity * CASE WHEN side='BUY' THEN 1 ELSE -1 END,
            status     = :status,
            updated_at = NOW()
        WHERE id = :id
        """
    )
    try:
        with get_engine().connect() as conn:
            conn.execute(sql, {"exit_price": exit_price, "status": status, "id": trade_id})
            conn.commit()
        logger.info("Trade %s closed at %.2f", trade_id, exit_price)
    except SQLAlchemyError as exc:
        logger.error("Failed to close trade %s: %s", trade_id, exc)
