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
from trader_bot.market_context import classify_regime, is_safe_to_trade, is_strategy_allowed
from trader_bot.strategies import (
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    ORBStrategy,
    SignalType,
    VCPStrategy,
    VWAPStrategy,
)

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
# Nifty 50 instrument token — used for regime detection
NIFTY_TOKEN = 256265
ACCOUNT_CAPITAL = 500_000  # INR — replace with live balance fetch

# Strategy pool — evaluated in priority order each cycle
STRATEGY_POOL = [
    ORBStrategy(opening_range_minutes=15, volume_multiplier=1.5, rr_ratio=2.0),
    VWAPStrategy(band_multiplier=1.0, atr_multiplier=1.2),
    MomentumStrategy(rsi_period=14, rsi_oversold=35, rsi_overbought=65),
    VCPStrategy(lookback=60),
    MovingAverageCrossoverStrategy(fast_period=9, slow_period=21),
]


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


class MarketContextResponse(BaseModel):
    regime: str
    adx: float
    volatility_ratio: float
    nifty_bias: str
    vix_level: Optional[float]
    position_size_multiplier: float
    allowed_strategies: list
    reason: str


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


@app.get("/market_context", response_model=MarketContextResponse)
def get_market_context():
    """Return current market regime analysis based on Nifty 50."""
    try:
        kite = get_kite_client()
        index_df = _fetch_ohlcv(kite, "NIFTY 50", interval="day", days=60, token=NIFTY_TOKEN)
        if index_df is None or index_df.empty:
            raise HTTPException(status_code=503, detail="Could not fetch Nifty data.")
        context = classify_regime(index_df)
        return MarketContextResponse(**context.__dict__)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
    1. Classifies market regime using Nifty 50 daily data.
    2. Filters strategies to those matching the regime.
    3. For each watchlist symbol, runs allowed strategies and acts on first signal.
    """
    interval_seconds = 300  # 5 minutes

    while _trading_active:
        try:
            if is_drawdown_limit_hit(ACCOUNT_CAPITAL):
                logger.warning("Daily drawdown limit hit — halting trading loop.")
                break

            kite = get_kite_client()

            # ── Step 1: Classify market regime ──────────────────────────────
            index_df = _fetch_ohlcv(kite, "NIFTY 50", interval="day", days=60, token=NIFTY_TOKEN)
            context = classify_regime(index_df) if index_df is not None else None

            if context is None or not is_safe_to_trade(context):
                logger.info("Market not safe to trade — skipping cycle.")
                if _trading_active:
                    time.sleep(interval_seconds)
                continue

            logger.info(
                "Regime: %s | ADX=%.1f | VR=%.2f | Bias=%s | SizeMult=%.1f",
                context.regime, context.adx, context.volatility_ratio,
                context.nifty_bias, context.position_size_multiplier,
            )

            # ── Step 2: Iterate watchlist ────────────────────────────────────
            for symbol in WATCHLIST:
                # Intraday 5-min candles for strategy signals
                df_intraday = _fetch_ohlcv(kite, symbol, interval="5minute", days=3)
                if df_intraday is None or df_intraday.empty:
                    continue

                # ── Step 3: Try each strategy in priority order ──────────────
                for strategy in STRATEGY_POOL:
                    strategy_name = strategy.__class__.__name__

                    # Map class name to allowed strategy keys used in context
                    if not _any_variant_allowed(strategy_name, context.allowed_strategies):
                        continue

                    try:
                        signal = strategy.generate_signal(df_intraday, symbol)
                    except Exception as exc:
                        logger.warning("Strategy %s failed for %s: %s", strategy_name, symbol, exc)
                        continue

                    if signal.signal_type == SignalType.HOLD:
                        continue

                    # ── Step 4: Size position with regime multiplier ─────────
                    base_qty = calculate_position_size(
                        capital=ACCOUNT_CAPITAL,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                    )
                    qty = max(int(base_qty * context.position_size_multiplier), 1)

                    logger.info(
                        "[%s] Signal: %s %s qty=%d entry=%.2f sl=%.2f tgt=%.2f (confidence=%.0f%%)",
                        context.regime, signal.signal_type, symbol, qty,
                        signal.entry_price, signal.stop_loss, signal.target,
                        signal.confidence * 100,
                    )

                    # ── Step 5: Place bracket order ──────────────────────────
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
                    # Only fire one strategy per symbol per cycle
                    break

        except Exception as exc:
            logger.error("Error in trading loop: %s", exc, exc_info=True)

        if _trading_active:
            time.sleep(interval_seconds)


def _any_variant_allowed(class_name: str, allowed: list) -> bool:
    """
    Match a strategy class name against the allowed strategy keys.
    e.g. 'VWAPStrategy' matches 'VWAP_Bounce' and 'VWAP_Retest_Support'.
    """
    normalized = class_name.lower().replace("strategy", "")
    return any(normalized in key.lower() for key in allowed)


def _fetch_ohlcv(
    kite,
    symbol: str,
    interval: str = "5minute",
    days: int = 5,
    token: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV candles from Kite API."""
    from datetime import date, timedelta

    try:
        to_date = date.today()
        from_date = to_date - timedelta(days=days)

        # Resolve instrument token if not provided
        if token is None:
            quote = kite.quote(f"NSE:{symbol}")
            token = quote[f"NSE:{symbol}"]["instrument_token"]

        records = kite.historical_data(
            instrument_token=token,
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
