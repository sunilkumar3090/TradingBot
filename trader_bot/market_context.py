"""
market_context.py — Daily market regime detection and trade filter.

Runs at 9:15 AM IST before any trades are placed.
Classifies the market into one of three regimes and adjusts
strategy selection and position sizing accordingly.

Regime classification uses:
  - ADX      : trend strength
  - ATR ratio: current volatility vs 20-day average (Hurst proxy)
  - Nifty50 vs 20-day EMA: directional bias

DISCLAIMER: For educational use only. All financial risk is assumed by the user.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"       # Strong uptrend — momentum strategies
    TRENDING_BEAR = "TRENDING_BEAR"       # Strong downtrend — short momentum
    MEAN_REVERTING = "MEAN_REVERTING"     # Range-bound — VWAP bounce, mean reversion
    HIGH_VOLATILITY = "HIGH_VOLATILITY"   # Choppy/news-driven — reduce size or sit out
    UNKNOWN = "UNKNOWN"                   # Insufficient data


@dataclass
class MarketContext:
    regime: MarketRegime
    adx: float                    # 0–100; >25 = trending
    volatility_ratio: float       # current ATR / 20-day avg ATR
    nifty_bias: str               # "BULLISH" | "BEARISH" | "NEUTRAL"
    vix_level: Optional[float]    # India VIX if available
    position_size_multiplier: float  # scale position sizes by this factor
    allowed_strategies: list      # which strategy names are permitted
    reason: str                   # human-readable explanation


# ---------------------------------------------------------------------------
# Core regime classifier
# ---------------------------------------------------------------------------

def classify_regime(
    index_df: pd.DataFrame,
    vix: Optional[float] = None,
) -> MarketContext:
    """
    Classify the current market regime using the Nifty 50 OHLCV DataFrame.

    Args:
        index_df: Daily OHLCV DataFrame for Nifty 50 (instrument token 256265).
                  Must have columns: open, high, low, close, volume.
                  At least 30 rows required.
        vix:      India VIX value (optional, fetched separately).

    Returns:
        MarketContext with regime, risk flags, and allowed strategies.
    """
    if len(index_df) < 30:
        logger.warning("Insufficient data for regime classification.")
        return MarketContext(
            regime=MarketRegime.UNKNOWN,
            adx=0.0,
            volatility_ratio=1.0,
            nifty_bias="NEUTRAL",
            vix_level=vix,
            position_size_multiplier=0.5,
            allowed_strategies=[],
            reason="Insufficient data — trading paused.",
        )

    df = index_df.copy().tail(60)

    adx_val = _adx(df)
    vol_ratio = _volatility_ratio(df)
    ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    current_close = float(df["close"].iloc[-1])
    nifty_bias = "BULLISH" if current_close > ema20 else "BEARISH"

    # ── High volatility override ─────────────────────────────────────────────
    if (vix is not None and vix > 20) or vol_ratio > 1.8:
        return MarketContext(
            regime=MarketRegime.HIGH_VOLATILITY,
            adx=adx_val,
            volatility_ratio=vol_ratio,
            nifty_bias=nifty_bias,
            vix_level=vix,
            position_size_multiplier=0.4,
            allowed_strategies=["VWAP_Bounce"],  # only mean-reversion, smaller size
            reason=(
                f"High volatility detected — VIX={vix}, ATR ratio={vol_ratio:.2f}. "
                "Reduce exposure."
            ),
        )

    # ── Trending market ──────────────────────────────────────────────────────
    if adx_val >= 25:
        if nifty_bias == "BULLISH":
            return MarketContext(
                regime=MarketRegime.TRENDING_BULL,
                adx=adx_val,
                volatility_ratio=vol_ratio,
                nifty_bias=nifty_bias,
                vix_level=vix,
                position_size_multiplier=1.0,
                allowed_strategies=[
                    "ORB_Bullish",
                    "Momentum_RSI",
                    "VWAP_Retest_Support",
                    "VCP",
                    "MA_Crossover",
                ],
                reason=f"Strong bull trend — ADX={adx_val:.1f}, price above EMA20.",
            )
        else:
            return MarketContext(
                regime=MarketRegime.TRENDING_BEAR,
                adx=adx_val,
                volatility_ratio=vol_ratio,
                nifty_bias=nifty_bias,
                vix_level=vix,
                position_size_multiplier=0.8,
                allowed_strategies=[
                    "ORB_Bearish",
                    "VWAP_Rejection",
                    "MA_Crossover",
                ],
                reason=f"Bear trend — ADX={adx_val:.1f}, price below EMA20.",
            )

    # ── Mean-reverting / range-bound ─────────────────────────────────────────
    return MarketContext(
        regime=MarketRegime.MEAN_REVERTING,
        adx=adx_val,
        volatility_ratio=vol_ratio,
        nifty_bias=nifty_bias,
        vix_level=vix,
        position_size_multiplier=0.7,
        allowed_strategies=[
            "VWAP_Bounce",
            "VWAP_Retest_Support",
            "Momentum_RSI",
        ],
        reason=(
            f"Range-bound market — ADX={adx_val:.1f}. "
            "Use mean-reversion strategies only."
        ),
    )


# ---------------------------------------------------------------------------
# Choppy / low-edge detection (call before entering any trade)
# ---------------------------------------------------------------------------

def is_safe_to_trade(context: MarketContext) -> bool:
    """
    Return False if conditions are unfavourable for trading.
    Logs the reason so the operator can review.
    """
    if context.regime == MarketRegime.UNKNOWN:
        logger.warning("Market regime UNKNOWN — blocking trades.")
        return False
    if context.regime == MarketRegime.HIGH_VOLATILITY and context.position_size_multiplier < 0.5:
        logger.warning("Extreme volatility — blocking trades. %s", context.reason)
        return False
    return True


def is_strategy_allowed(strategy_name: str, context: MarketContext) -> bool:
    """Return True if the given strategy is compatible with the current regime."""
    allowed = strategy_name in context.allowed_strategies
    if not allowed:
        logger.info(
            "Strategy '%s' not allowed in regime '%s'. Skipping.",
            strategy_name, context.regime,
        )
    return allowed


# ---------------------------------------------------------------------------
# Technical indicators (internal)
# ---------------------------------------------------------------------------

def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ADX (Average Directional Index) — trend strength indicator."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # If both DM positive, keep only the larger and zero the other
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0.0)
    minus_dm = minus_dm.where(~mask, 0.0)

    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0


def _volatility_ratio(df: pd.DataFrame, period: int = 14, lookback: int = 20) -> float:
    """
    Current ATR divided by the rolling mean of ATR over `lookback` days.
    Ratio > 1.5 indicates elevated volatility.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    avg_atr = atr.rolling(lookback).mean()
    current_atr = float(atr.iloc[-1])
    mean_atr = float(avg_atr.iloc[-1])
    if mean_atr == 0:
        return 1.0
    return round(current_atr / mean_atr, 3)
