"""
main.py — TraderBot FastAPI application and trading loop.

Endpoints:
  GET  /status        — health check
  POST /start         — start the trading loop
  POST /stop          — emergency stop (cancel all orders)
  GET  /positions     — view current positions
  POST /manual_order  — place a manual order

DISCLAIMER: For educational use only. All financial risk is assumed by the user.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from config.settings import get_settings
from trader_bot.auth import get_kite_client
from trader_bot.data_stream import TickStreamer, get_latest_tick
from trader_bot.execution import (
    cancel_all_orders,
    get_positions,
    place_bracket_order,
    place_order,
)
from trader_bot.risk_manager import (
    calculate_position_size,
    close_trade,
    init_db,
    is_drawdown_limit_hit,
    log_trade,
)
from trader_bot.strategies import MomentumStrategy, MovingAverageCrossoverStrategy, SignalType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_trading_active = False
_trading_thread: Optional[threading.Thread] = None
_streamer: Optional[TickStreamer] = None

# Instruments to trade: {symbol: instrument_token}
WATCHLIST = {
    "RELIANCE": 738561,
    "INFY": 408065,
}
ACCOUNT_CAPITAL = 500_000  # INR — replace with live balance fetch


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("TraderBot API started.")
    yield
    _stop_trading()
    logger.info("TraderBot API shut down.")


app = FastAPI(
    title="TraderBot API",
    description="AI-Powered Intraday Trading System — Zerodha Kite Connect",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ManualOrderRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    transaction_type: str  # BUY or SELL
    quantity: int
    order_type: str = "MARKET"
    price: float = 0.0


class StatusResponse(BaseModel):
    status: str
    trading_active: bool
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(
        status="ok",
        trading_active=_trading_active,
        message="TraderBot is running." if _trading_active else "TraderBot is idle.",
    )


@app.post("/start", response_model=StatusResponse)
def start_bot():
    global _trading_active, _trading_thread, _streamer

    if _trading_active:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bot already running.")

    kite = get_kite_client()

    # Start WebSocket tick streaming
    tokens = list(WATCHLIST.values())
    _streamer = TickStreamer(instrument_tokens=tokens)
    _streamer.start()

    # Start trading loop in background thread
    _trading_active = True
    _trading_thread = threading.Thread(target=_trading_loop, daemon=True)
    _trading_thread.start()

    logger.info("Trading loop started.")
    return StatusResponse(status="started", trading_active=True, message="Bot started successfully.")


@app.post("/stop", response_model=StatusResponse)
def stop_bot():
    cancelled = _stop_trading()
    return StatusResponse(
        status="stopped",
        trading_active=False,
        message=f"Bot stopped. {cancelled} pending orders cancelled.",
    )


@app.get("/positions")
def view_positions():
    try:
        kite = get_kite_client()
        return get_positions(kite)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/manual_order")
def manual_order(req: ManualOrderRequest):
    if req.transaction_type not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="transaction_type must be BUY or SELL.")
    if req.quantity <= 0:
        raise HTTPException(status_code=400, detail="quantity must be > 0.")

    try:
        kite = get_kite_client()
        order_id = place_order(
            kite=kite,
            symbol=req.symbol,
            exchange=req.exchange,
            transaction_type=req.transaction_type,
            quantity=req.quantity,
            order_type=req.order_type,
            price=req.price,
        )
        return {"order_id": order_id, "status": "placed"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Trading loop (runs in background thread)
# ---------------------------------------------------------------------------

def _trading_loop() -> None:
    """
    Main strategy loop — runs every 5 minutes during market hours.
    Analyses cached ticks, generates signals, manages positions.
    """
    strategy = MomentumStrategy()
    interval_seconds = 300  # 5 minutes

    while _trading_active:
        try:
            if is_drawdown_limit_hit(ACCOUNT_CAPITAL):
                logger.warning("Daily drawdown limit hit — halting trading loop.")
                break

            kite = get_kite_client()

            for symbol, token in WATCHLIST.items():
                tick = get_latest_tick(token)
                if not tick:
                    continue

                # Build a minimal OHLCV DataFrame from historical data
                df = _fetch_ohlcv(kite, symbol)
                if df is None or df.empty:
                    continue

                signal = strategy.generate_signal(df, symbol)
                if signal.signal_type == SignalType.HOLD:
                    continue

                qty = calculate_position_size(
                    capital=ACCOUNT_CAPITAL,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                )
                if qty == 0:
                    continue

                logger.info(
                    "Signal: %s %s qty=%d entry=%.2f sl=%.2f tgt=%.2f",
                    signal.signal_type, symbol, qty,
                    signal.entry_price, signal.stop_loss, signal.target,
                )

                result = place_bracket_order(
                    kite=kite,
                    symbol=symbol,
                    exchange=settings.trading_exchange,
                    transaction_type=signal.signal_type.value,
                    quantity=qty,
                    entry_price=signal.entry_price,
                    target_price=signal.target,
                    stop_loss_trigger=signal.stop_loss,
                    stop_loss_limit=round(signal.stop_loss * 0.999, 2),
                )

                if result.get("status") == "PLACED":
                    log_trade(
                        symbol=symbol,
                        side=signal.signal_type.value,
                        quantity=qty,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        target=signal.target,
                        strategy=signal.strategy_name,
                        kite_order_id=result["entry_order_id"],
                    )

        except Exception as exc:
            logger.error("Error in trading loop: %s", exc, exc_info=True)

        if _trading_active:
            time.sleep(interval_seconds)


def _fetch_ohlcv(kite, symbol: str, interval: str = "5minute", days: int = 5) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV candles from Kite API."""
    from datetime import date, timedelta

    try:
        instrument = f"NSE:{symbol}"
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        records = kite.historical_data(
            instrument_token=None,  # fetched by symbol via quote
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )
        df = pd.DataFrame(records)
        df.rename(columns={"date": "datetime"}, inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        logger.error("Failed to fetch OHLCV for %s: %s", symbol, exc)
        return None


def _stop_trading() -> int:
    global _trading_active, _streamer
    cancelled = 0
    _trading_active = False
    if _streamer:
        _streamer.stop()
        _streamer = None
    try:
        kite = get_kite_client()
        cancelled = cancel_all_orders(kite)
    except Exception as exc:
        logger.error("Could not cancel orders on stop: %s", exc)
    return cancelled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("trader_bot.main:app", host="0.0.0.0", port=8000, reload=False)
