"""
execution.py — Order placement and tracking via Kite Connect.

Implements:
  - Market / Limit order placement (MIS intraday).
  - Bracket order logic (entry + SL + target as three linked orders).
  - Rate limiting to stay within Kite's 100 orders/min limit.
  - Order status polling and cancellation.

DISCLAIMER: For educational use only. All financial risk is assumed by the user.
"""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Optional

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from config.settings import get_settings
from trader_bot.auth import get_kite_client

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Simple token-bucket rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Allow at most `max_calls` within a rolling `period_seconds` window."""

    def __init__(self, max_calls: int = 100, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._timestamps: deque = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        # Remove timestamps outside the rolling window
        while self._timestamps and now - self._timestamps[0] > self.period_seconds:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.period_seconds - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.warning("Rate limit approaching — sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)

        self._timestamps.append(time.monotonic())


_limiter = _RateLimiter(max_calls=settings.kite_api_rate_limit, period_seconds=60)


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------

def place_order(
    kite: KiteConnect,
    symbol: str,
    exchange: str,
    transaction_type: str,  # "BUY" or "SELL"
    quantity: int,
    order_type: str = "MARKET",  # "MARKET" or "LIMIT"
    price: float = 0.0,
    trigger_price: float = 0.0,
    tag: str = "TraderBot",
) -> Optional[str]:
    """
    Place a single MIS order. Returns the Kite order_id on success.

    Raises:
        KiteException: propagated from the API on validation / connectivity errors.
    """
    _limiter.acquire()
    try:
        order_id = kite.place_order(
            tradingsymbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=settings.order_product_type,  # MIS
            price=price if order_type == "LIMIT" else None,
            trigger_price=trigger_price if trigger_price else None,
            tag=tag,
            variety=kite.VARIETY_REGULAR,
        )
        logger.info(
            "Order placed — id: %s | %s %s x%d @ %s",
            order_id, transaction_type, symbol, quantity, price or "MARKET",
        )
        return str(order_id)
    except KiteException as exc:
        logger.error("Order placement failed for %s: %s", symbol, exc)
        raise


def place_bracket_order(
    kite: KiteConnect,
    symbol: str,
    exchange: str,
    transaction_type: str,
    quantity: int,
    entry_price: float,
    target_price: float,
    stop_loss_trigger: float,
    stop_loss_limit: float,
    tag: str = "TraderBot_BO",
) -> dict:
    """
    Place a bracket order as three coordinated legs:
      1. Entry LIMIT order
      2. Target LIMIT order  (placed after entry fills)
      3. Stop-Loss SL-M order (placed after entry fills)

    Returns a dict with order IDs for all three legs.

    Note: Kite Connect BO variety may not be available on all accounts.
          This implementation uses manual leg placement as a fallback.
    """
    logger.info("Placing bracket order for %s — entry: %.2f, SL: %.2f, target: %.2f",
                symbol, entry_price, stop_loss_trigger, target_price)

    # Leg 1: Entry
    entry_order_id = place_order(
        kite=kite,
        symbol=symbol,
        exchange=exchange,
        transaction_type=transaction_type,
        quantity=quantity,
        order_type="LIMIT",
        price=entry_price,
        tag=tag,
    )

    # Wait for entry to fill before placing exit legs
    filled = _wait_for_fill(kite, entry_order_id, timeout_seconds=30)
    if not filled:
        logger.warning("Entry order %s did not fill in time. Cancelling.", entry_order_id)
        cancel_order(kite, entry_order_id)
        return {"entry_order_id": entry_order_id, "status": "ENTRY_NOT_FILLED"}

    exit_side = "SELL" if transaction_type == "BUY" else "BUY"

    # Leg 2: Target
    target_order_id = place_order(
        kite=kite,
        symbol=symbol,
        exchange=exchange,
        transaction_type=exit_side,
        quantity=quantity,
        order_type="LIMIT",
        price=target_price,
        tag=f"{tag}_TGT",
    )

    # Leg 3: Stop-loss
    sl_order_id = place_order(
        kite=kite,
        symbol=symbol,
        exchange=exchange,
        transaction_type=exit_side,
        quantity=quantity,
        order_type="SL-M",
        trigger_price=stop_loss_trigger,
        tag=f"{tag}_SL",
    )

    return {
        "entry_order_id": entry_order_id,
        "target_order_id": target_order_id,
        "sl_order_id": sl_order_id,
        "status": "PLACED",
    }


def cancel_order(kite: KiteConnect, order_id: str) -> bool:
    """Cancel a pending order. Returns True on success."""
    _limiter.acquire()
    try:
        kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
        logger.info("Order %s cancelled.", order_id)
        return True
    except KiteException as exc:
        logger.error("Failed to cancel order %s: %s", order_id, exc)
        return False


def cancel_all_orders(kite: KiteConnect) -> int:
    """Cancel all open/pending orders. Returns count of cancelled orders."""
    try:
        orders = kite.orders()
    except KiteException as exc:
        logger.error("Could not fetch orders for bulk cancel: %s", exc)
        return 0

    cancelled = 0
    for order in orders:
        if order.get("status") in ("OPEN", "TRIGGER PENDING"):
            if cancel_order(kite, order["order_id"]):
                cancelled += 1
    logger.info("Bulk cancel: %d orders cancelled.", cancelled)
    return cancelled


def get_positions(kite: KiteConnect) -> dict:
    """Return current intraday (day) and carry-forward (net) positions."""
    try:
        return kite.positions()
    except KiteException as exc:
        logger.error("Failed to fetch positions: %s", exc)
        return {"day": [], "net": []}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wait_for_fill(kite: KiteConnect, order_id: str, timeout_seconds: int = 30) -> bool:
    """Poll order status until COMPLETE or timeout."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            history = kite.order_history(order_id)
            latest_status = history[-1].get("status", "") if history else ""
            if latest_status == "COMPLETE":
                return True
            if latest_status in ("CANCELLED", "REJECTED"):
                logger.warning("Order %s ended with status: %s", order_id, latest_status)
                return False
        except KiteException as exc:
            logger.error("Error polling order %s: %s", order_id, exc)
        time.sleep(2)
    return False
