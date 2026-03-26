"""
data_stream.py — WebSocket tick streaming via KiteTicker.

Ticks are published to Redis (low-latency cache) and persisted
to PostgreSQL via a background thread.

DISCLAIMER: For educational use only. All financial risk is assumed by the user.
"""

import json
import logging
import threading
from typing import Callable, List, Optional

import redis
from kiteconnect import KiteTicker

from config.settings import get_settings
from trader_bot.auth import get_kite_client

logger = logging.getLogger(__name__)
settings = get_settings()

# Redis client (shared across threads — redis-py is thread-safe)
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


class TickStreamer:
    """
    Manages a KiteTicker WebSocket connection.
    Subscribes to a list of instrument tokens and caches ticks in Redis.
    """

    def __init__(
        self,
        instrument_tokens: List[int],
        on_tick_callback: Optional[Callable] = None,
    ):
        self.instrument_tokens = instrument_tokens
        self.on_tick_callback = on_tick_callback
        self._ticker: Optional[KiteTicker] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Initialise and connect the WebSocket in a background thread."""
        kite = get_kite_client()
        self._ticker = KiteTicker(
            api_key=settings.kite_api_key,
            access_token=kite.access_token,
        )
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect
        self._ticker.on_noreconnect = self._on_noreconnect

        logger.info("Starting KiteTicker WebSocket...")
        # reconnect=True with max 50 attempts, 5s interval
        self._ticker.connect(threaded=True, disable_ssl_verification=False)

    def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._stop_event.set()
        if self._ticker:
            try:
                self._ticker.close()
                logger.info("KiteTicker connection closed.")
            except Exception as exc:
                logger.error("Error closing ticker: %s", exc)

    # ------------------------------------------------------------------
    # KiteTicker callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, ws, response) -> None:
        logger.info("WebSocket connected. Subscribing to tokens: %s", self.instrument_tokens)
        ws.subscribe(self.instrument_tokens)
        ws.set_mode(ws.MODE_FULL, self.instrument_tokens)

    def _on_ticks(self, ws, ticks: list) -> None:
        r = get_redis()
        for tick in ticks:
            token = tick.get("instrument_token")
            if not token:
                continue
            try:
                # Cache latest tick per token (TTL: 60s — stale data protection)
                r.setex(f"tick:{token}", 60, json.dumps(tick, default=str))
            except redis.RedisError as exc:
                logger.error("Redis write failed for token %s: %s", token, exc)

            if self.on_tick_callback:
                try:
                    self.on_tick_callback(tick)
                except Exception as exc:
                    logger.error("Tick callback error: %s", exc)

    def _on_close(self, ws, code, reason) -> None:
        logger.warning("WebSocket closed — code: %s, reason: %s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        logger.error("WebSocket error — code: %s, reason: %s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.info("WebSocket reconnecting... attempt #%s", attempts_count)

    def _on_noreconnect(self, ws) -> None:
        logger.critical("WebSocket could not reconnect. Manual intervention required.")


def get_latest_tick(instrument_token: int) -> Optional[dict]:
    """Fetch the most recent cached tick for an instrument token from Redis."""
    r = get_redis()
    try:
        raw = r.get(f"tick:{instrument_token}")
        return json.loads(raw) if raw else None
    except (redis.RedisError, json.JSONDecodeError) as exc:
        logger.error("Failed to fetch tick for %s: %s", instrument_token, exc)
        return None
