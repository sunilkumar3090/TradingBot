"""
strategies.py — Trading strategy implementations.

Strategies return a Signal (BUY / SELL / HOLD) based on OHLCV data.

DISCLAIMER: For educational use only. Strategies are NOT financial advice.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    entry_price: float
    stop_loss: float
    target: float
    strategy_name: str
    confidence: float = 0.0  # 0.0 – 1.0
    notes: str = ""


class Strategy(ABC):
    """Base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyse a DataFrame of OHLCV data and return a trading Signal.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].
                Index must be a DatetimeIndex sorted ascending.
            symbol: Instrument symbol (e.g. 'RELIANCE').
        """
        ...

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 20) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if len(df) < min_rows:
            raise ValueError(
                f"Insufficient data: need at least {min_rows} rows, got {len(df)}."
            )


# ---------------------------------------------------------------------------
# Strategy 1: Moving Average Crossover (simple baseline)
# ---------------------------------------------------------------------------

class MovingAverageCrossoverStrategy(Strategy):
    """
    Generates BUY when the fast EMA crosses above the slow EMA,
    SELL when it crosses below.
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21, atr_multiplier: float = 1.5):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_multiplier = atr_multiplier

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        self._validate_df(df, min_rows=self.slow_period + 1)
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_period, adjust=False).mean()

        prev, curr = df.iloc[-2], df.iloc[-1]
        entry = float(curr["close"])
        atr = self._atr(df)

        if prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]:
            stop_loss = round(entry - self.atr_multiplier * atr, 2)
            target = round(entry + 2 * self.atr_multiplier * atr, 2)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="MA_Crossover",
                confidence=0.6,
            )
        if prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]:
            stop_loss = round(entry + self.atr_multiplier * atr, 2)
            target = round(entry - 2 * self.atr_multiplier * atr, 2)
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="MA_Crossover",
                confidence=0.6,
            )
        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            entry_price=entry,
            stop_loss=0.0,
            target=0.0,
            strategy_name="MA_Crossover",
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Strategy 2: Momentum (RSI + Volume Spike)
# ---------------------------------------------------------------------------

class MomentumStrategy(Strategy):
    """
    BUY when RSI crosses above oversold threshold on a volume spike.
    SELL when RSI crosses below overbought threshold.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        volume_spike_multiplier: float = 1.5,
        atr_multiplier: float = 1.5,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_spike_multiplier = volume_spike_multiplier
        self.atr_multiplier = atr_multiplier

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        self._validate_df(df, min_rows=self.rsi_period + 5)
        df = df.copy()
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        avg_volume = df["volume"].rolling(20).mean()
        df["volume_spike"] = df["volume"] > (avg_volume * self.volume_spike_multiplier)

        prev, curr = df.iloc[-2], df.iloc[-1]
        entry = float(curr["close"])
        atr = MovingAverageCrossoverStrategy._atr(df)

        if (
            prev["rsi"] < self.rsi_oversold
            and curr["rsi"] >= self.rsi_oversold
            and curr["volume_spike"]
        ):
            stop_loss = round(entry - self.atr_multiplier * atr, 2)
            target = round(entry + 2 * self.atr_multiplier * atr, 2)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="Momentum_RSI",
                confidence=0.7,
                notes=f"RSI: {curr['rsi']:.1f}, Volume spike detected",
            )
        if (
            prev["rsi"] > self.rsi_overbought
            and curr["rsi"] <= self.rsi_overbought
        ):
            stop_loss = round(entry + self.atr_multiplier * atr, 2)
            target = round(entry - 2 * self.atr_multiplier * atr, 2)
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="Momentum_RSI",
                confidence=0.65,
                notes=f"RSI: {curr['rsi']:.1f}",
            )
        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            entry_price=entry,
            stop_loss=0.0,
            target=0.0,
            strategy_name="Momentum_RSI",
        )

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Strategy 3: Volatility Contraction Pattern (VCP)
# ---------------------------------------------------------------------------

class VCPStrategy(Strategy):
    """
    Detects Volatility Contraction Patterns:
    - 3+ progressively tighter pullbacks
    - Final contraction < 50% of the first
    - Breakout on above-average volume
    """

    def __init__(self, lookback: int = 60, breakout_volume_multiplier: float = 1.3):
        self.lookback = lookback
        self.breakout_volume_multiplier = breakout_volume_multiplier

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        self._validate_df(df, min_rows=self.lookback)
        df = df.copy().tail(self.lookback)

        entry = float(df["close"].iloc[-1])
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]
        is_volume_breakout = df["volume"].iloc[-1] > avg_volume * self.breakout_volume_multiplier

        contractions = self._measure_contractions(df)
        is_vcp = (
            len(contractions) >= 3
            and all(contractions[i] > contractions[i + 1] for i in range(len(contractions) - 1))
            and contractions[-1] < contractions[0] * 0.5
        )

        if is_vcp and is_volume_breakout:
            atr = MovingAverageCrossoverStrategy._atr(df)
            stop_loss = round(df["low"].tail(5).min(), 2)
            target = round(entry + 3 * (entry - stop_loss), 2)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="VCP",
                confidence=0.8,
                notes=f"VCP detected: {len(contractions)} contractions",
            )
        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            entry_price=entry,
            stop_loss=0.0,
            target=0.0,
            strategy_name="VCP",
        )

    @staticmethod
    def _measure_contractions(df: pd.DataFrame) -> list:
        """Measure the range of successive swing highs-to-lows as a % of high."""
        window = 10
        contractions = []
        for i in range(0, len(df) - window, window):
            chunk = df.iloc[i: i + window]
            high = chunk["high"].max()
            low = chunk["low"].min()
            if high > 0:
                contractions.append((high - low) / high * 100)
        return contractions


# ---------------------------------------------------------------------------
# Strategy 4: VWAP + VWAP Band Strategy (institutional anchor)
# ---------------------------------------------------------------------------

class VWAPStrategy(Strategy):
    """
    Trades mean-reversion and trend continuation around VWAP.

    Long setups:
      - Price dips to VWAP lower band (1 std dev) and bounces with volume
      - Price breaks above VWAP and retests it as support (trend continuation)

    Short setups:
      - Price spikes to VWAP upper band and rejects with volume
      - Price breaks below VWAP and retests it as resistance

    Requires intraday data (1min or 5min candles for the current session).
    The 'date' column or index must be a DatetimeIndex.
    """

    def __init__(self, band_multiplier: float = 1.0, atr_multiplier: float = 1.2):
        self.band_multiplier = band_multiplier
        self.atr_multiplier = atr_multiplier

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        self._validate_df(df, min_rows=10)
        df = df.copy()

        # Compute VWAP and bands for the session
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["cum_tp_vol"] = df["tp_vol"].cumsum()
        df["cum_vol"] = df["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)

        # Rolling std dev of typical price for bands
        df["vwap_std"] = df["typical_price"].rolling(20).std()
        df["upper_band"] = df["vwap"] + self.band_multiplier * df["vwap_std"]
        df["lower_band"] = df["vwap"] - self.band_multiplier * df["vwap_std"]

        prev, curr = df.iloc[-2], df.iloc[-1]
        entry = float(curr["close"])
        atr = MovingAverageCrossoverStrategy._atr(df)

        vwap = float(curr["vwap"])
        upper = float(curr["upper_band"])
        lower = float(curr["lower_band"])

        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        vol_confirm = curr["volume"] > avg_vol * 1.2

        # --- Long: bounce off lower VWAP band ---------------------------------
        if (
            prev["close"] <= prev["lower_band"]
            and curr["close"] > curr["lower_band"]
            and vol_confirm
        ):
            stop_loss = round(entry - self.atr_multiplier * atr, 2)
            target = round(vwap + (vwap - stop_loss), 2)  # target = opposite VWAP side
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="VWAP_Bounce",
                confidence=0.72,
                notes=f"Bounce off lower band. VWAP={vwap:.2f}, Band={lower:.2f}",
            )

        # --- Long: retest of VWAP as support (trend continuation) -------------
        if (
            prev["close"] > prev["vwap"]
            and curr["low"] <= curr["vwap"]
            and curr["close"] > curr["vwap"]
            and vol_confirm
        ):
            stop_loss = round(lower, 2)
            target = round(upper, 2)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="VWAP_Retest_Support",
                confidence=0.75,
                notes=f"VWAP retest as support. VWAP={vwap:.2f}",
            )

        # --- Short: rejection at upper VWAP band ------------------------------
        if (
            prev["close"] >= prev["upper_band"]
            and curr["close"] < curr["upper_band"]
            and vol_confirm
        ):
            stop_loss = round(entry + self.atr_multiplier * atr, 2)
            target = round(vwap - (stop_loss - vwap), 2)
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="VWAP_Rejection",
                confidence=0.70,
                notes=f"Rejection at upper band. VWAP={vwap:.2f}, Band={upper:.2f}",
            )

        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            entry_price=entry,
            stop_loss=0.0,
            target=0.0,
            strategy_name="VWAP",
        )


# ---------------------------------------------------------------------------
# Strategy 5: Opening Range Breakout (ORB)
# ---------------------------------------------------------------------------

class ORBStrategy(Strategy):
    """
    Opening Range Breakout — one of the highest win-rate intraday patterns.

    Logic:
      - Define the opening range: high/low of the first N minutes (default 15).
      - BUY when price breaks above the opening range high with volume.
      - SELL when price breaks below the opening range low with volume.
      - Stop loss: opposite side of the opening range.
      - Target: at least 1:2 R:R (configurable).

    Requires 1-min or 5-min intraday data. The DataFrame index must be
    a DatetimeIndex. Call this only after the opening range candle has closed.
    """

    def __init__(
        self,
        opening_range_minutes: int = 15,
        volume_multiplier: float = 1.5,
        rr_ratio: float = 2.0,
    ):
        self.opening_range_minutes = opening_range_minutes
        self.volume_multiplier = volume_multiplier
        self.rr_ratio = rr_ratio

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        self._validate_df(df, min_rows=5)
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("ORBStrategy requires a DatetimeIndex.")

        session_start = df.index[0]
        or_end = session_start + pd.Timedelta(minutes=self.opening_range_minutes)

        or_df = df[df.index <= or_end]
        if or_df.empty:
            return self._hold(symbol, df)

        or_high = float(or_df["high"].max())
        or_low = float(or_df["low"].min())
        or_range = or_high - or_low

        # Only trade if we're past the opening range
        post_or_df = df[df.index > or_end]
        if post_or_df.empty:
            return self._hold(symbol, df)

        curr = post_or_df.iloc[-1]
        prev = post_or_df.iloc[-2] if len(post_or_df) >= 2 else or_df.iloc[-1]

        entry = float(curr["close"])
        avg_vol = df["volume"].rolling(10).mean().iloc[-1]
        vol_confirm = float(curr["volume"]) > avg_vol * self.volume_multiplier

        # --- Bullish breakout -------------------------------------------------
        if (
            float(prev["close"]) <= or_high
            and entry > or_high
            and vol_confirm
        ):
            stop_loss = round(or_low, 2)
            risk = entry - stop_loss
            target = round(entry + self.rr_ratio * risk, 2)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="ORB_Bullish",
                confidence=0.78,
                notes=(
                    f"ORB breakout above {or_high:.2f}. "
                    f"Range={or_range:.2f}, Vol={curr['volume']:.0f}"
                ),
            )

        # --- Bearish breakdown ------------------------------------------------
        if (
            float(prev["close"]) >= or_low
            and entry < or_low
            and vol_confirm
        ):
            stop_loss = round(or_high, 2)
            risk = stop_loss - entry
            target = round(entry - self.rr_ratio * risk, 2)
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                strategy_name="ORB_Bearish",
                confidence=0.75,
                notes=(
                    f"ORB breakdown below {or_low:.2f}. "
                    f"Range={or_range:.2f}, Vol={curr['volume']:.0f}"
                ),
            )

        return self._hold(symbol, df)

    def _hold(self, symbol: str, df: pd.DataFrame) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            entry_price=float(df["close"].iloc[-1]),
            stop_loss=0.0,
            target=0.0,
            strategy_name="ORB",
        )
