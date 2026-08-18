"""Trading strategies used by the backtest engine.

Each strategy exposes ``generate_signals(candles) -> list[int]`` where the
integer is ``+1`` (long entry), ``-1`` (long exit / go flat), or ``0`` (no
action). Strategies are kept dependency-free and testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from trading_agents.analysis import indicators as ta


class Strategy(ABC):
    """Interface every strategy implements."""

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, candles) -> List[int]:
        """Return per-bar signals (+1 enter long, -1 exit, 0 none)."""
        raise NotImplementedError


class SMACrossStrategy(Strategy):
    """Golden/death cross on short vs long SMA. Classic trend follower."""

    name = "sma_cross"

    def __init__(self, short: int = 20, long: int = 50) -> None:
        self.short = short
        self.long = long
        if short >= long:
            raise ValueError("short period must be < long period")

    def generate_signals(self, candles) -> List[int]:
        closes = [c.close for c in candles]
        sma_s = ta.sma(closes, self.short)
        sma_l = ta.sma(closes, self.long)
        signals: List[int] = [0] * len(closes)
        in_position = False
        for i in range(1, len(closes)):
            s_now, l_now = sma_s[i], sma_l[i]
            s_prev, l_prev = sma_s[i - 1], sma_l[i - 1]
            if None in (s_now, l_now, s_prev, l_prev) or any(
                _is_nan(v) for v in (s_now, l_now, s_prev, l_prev)
            ):
                continue
            if not in_position and s_now > l_now and s_prev <= l_prev:
                signals[i] = 1
                in_position = True
            elif in_position and s_now < l_now and s_prev >= l_prev:
                signals[i] = -1
                in_position = False
        return signals


class RSIMeanReversionStrategy(Strategy):
    """Buy oversold, sell overbought. Mean-reversion classic."""

    name = "rsi_mean_reversion"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, candles) -> List[int]:
        closes = [c.close for c in candles]
        rsi = ta.rsi(closes, self.period)
        signals: List[int] = [0] * len(closes)
        in_position = False
        for i in range(1, len(closes)):
            r = rsi[i]
            if _is_nan(r):
                continue
            if not in_position and r < self.oversold:
                signals[i] = 1
                in_position = True
            elif in_position and r > self.overbought:
                signals[i] = -1
                in_position = False
        return signals


class MACDStrategy(Strategy):
    """Follows MACD line / signal-line crossovers."""

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, candles) -> List[int]:
        closes = [c.close for c in candles]
        macd_line, signal_line, _ = ta.macd(closes, self.fast, self.slow, self.signal)
        signals: List[int] = [0] * len(closes)
        in_position = False
        for i in range(1, len(closes)):
            m, s = macd_line[i], signal_line[i]
            m_prev, s_prev = macd_line[i - 1], signal_line[i - 1]
            if any(_is_nan(v) for v in (m, s, m_prev, s_prev)):
                continue
            if not in_position and m > s and m_prev <= s_prev:
                signals[i] = 1
                in_position = True
            elif in_position and m < s and m_prev >= s_prev:
                signals[i] = -1
                in_position = False
        return signals


class BollingerBandsStrategy(Strategy):
    """Mean reversion: buy at lower band, sell at upper band."""

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self.period = period
        self.num_std = num_std

    def generate_signals(self, candles) -> List[int]:
        closes = [c.close for c in candles]
        upper, _, lower = ta.bollinger_bands(closes, self.period, self.num_std)
        signals: List[int] = [0] * len(closes)
        in_position = False
        for i in range(1, len(closes)):
            u, lo = upper[i], lower[i]
            if _is_nan(u) or _is_nan(lo):
                continue
            if not in_position and closes[i] <= lo:
                signals[i] = 1
                in_position = True
            elif in_position and closes[i] >= u:
                signals[i] = -1
                in_position = False
        return signals


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "sma_cross": SMACrossStrategy,
    "sma": SMACrossStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "rsi": RSIMeanReversionStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerBandsStrategy,
    "bollinger_bands": BollingerBandsStrategy,
}


def build_strategy(name: str, **kwargs) -> Strategy:
    """Instantiate a strategy by name (case-insensitive)."""
    key = name.strip().lower()
    cls = _STRATEGIES.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {sorted(set(_STRATEGIES.values()), key=lambda c: c.__name__)}"
        )
    return cls(**kwargs)


def _is_nan(value: float) -> bool:
    import math

    return isinstance(value, float) and math.isnan(value)
