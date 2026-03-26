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
