"""Pure-Python technical indicators.

All functions operate on sequences of floats (usually close prices) and return
lists aligned with the input (leading values are NaN). No external numpy/pandas
dependency is required, but ``pandas`` is used when available for speed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _values(series: Sequence[float]) -> List[float]:
    return [float(x) for x in series]


def sma(series: Sequence[float], period: int) -> List[float]:
    """Simple Moving Average (NaN-tolerant: windows containing NaN → NaN)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    values = _values(series)
    out: List[float] = [math.nan] * len(values)
    running = 0.0
    nan_count = 0
    for i, v in enumerate(values):
        if math.isnan(v):
            nan_count += 1
        else:
            running += v
        if i >= period:
            old = values[i - period]
            if math.isnan(old):
                nan_count -= 1
            else:
                running -= old
        if i >= period - 1 and nan_count == 0:
            out[i] = running / period
    return out


def ema(series: Sequence[float], period: int) -> List[float]:
    """Exponential Moving Average (standard α = 2/(period+1))."""
    if period <= 0:
        raise ValueError("period must be > 0")
    values = _values(series)
    out: List[float] = [math.nan] * len(values)
    if not values:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period if len(values) >= period else values[0]
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(series: Sequence[float], period: int = 14) -> List[float]:
    """Relative Strength Index (Wilder's smoothing)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    values = _values(series)
    out: List[float] = [math.nan] * len(values)
    if len(values) < period + 1:
        return out
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out[i] = 100.0 - 100.0 / (1.0 + rs) if rs != float("inf") else 100.0
    return out


def macd(
    series: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    """MACD line, signal line and histogram (aligned lists)."""
    values = _values(series)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [
        (f - s) if not (math.isnan(f) or math.isnan(s)) else math.nan
        for f, s in zip(ema_fast, ema_slow)
    ]
    # signal line: EMA of the non-NaN MACD prefix
    signal_line: List[float] = [math.nan] * len(values)
    valid = [(i, v) for i, v in enumerate(macd_line) if not math.isnan(v)]
    if valid:
        signal_values = ema([v for _, v in valid], signal)
        for (idx, _), sv in zip(valid, signal_values):
            signal_line[idx] = sv
    histogram = [
        (m - s) if not (math.isnan(m) or math.isnan(s)) else math.nan
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: Sequence[float],
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    """Upper band, middle band (SMA), lower band."""
    if period <= 0:
        raise ValueError("period must be > 0")
    values = _values(series)
    middle = sma(values, period)
    upper: List[float] = [math.nan] * len(values)
    lower: List[float] = [math.nan] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, middle, lower


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> List[float]:
    """Average True Range (Wilder's)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    highs_l = _values(highs)
    lows_l = _values(lows)
    closes_l = _values(closes)
    n = len(closes_l)
    out: List[float] = [math.nan] * n
    if n < 2:
        return out
    trs: List[float] = []
    for i in range(1, n):
        tr = max(
            highs_l[i] - lows_l[i],
            abs(highs_l[i] - closes_l[i - 1]),
            abs(lows_l[i] - closes_l[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return out
    first = sum(trs[:period]) / period
    out[period] = first
    prev = first
    for i in range(period, len(trs)):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i + 1] = prev
    return out


def stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
    smooth: int = 3,
) -> Tuple[List[float], List[float]]:
    """%K and %D stochastic oscillator (0-100)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    highs_l = _values(highs)
    lows_l = _values(lows)
    closes_l = _values(closes)
    n = len(closes_l)
    k_raw: List[float] = [math.nan] * n
    for i in range(period - 1, n):
        window_high = max(highs_l[i - period + 1 : i + 1])
        window_low = min(lows_l[i - period + 1 : i + 1])
        rng = window_high - window_low
        k_raw[i] = 100.0 * (closes_l[i] - window_low) / rng if rng else 50.0
    k = sma(k_raw, smooth) if smooth > 1 else list(k_raw)
    d = sma(k, smooth) if smooth > 1 else list(k)
    return k, d


def last_valid(series: Sequence[float]) -> Optional[float]:
    """Return the last non-NaN value of a series (or None)."""
    for v in reversed(series):
        if not math.isnan(v):
            return v
    return None


def percent_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / previous * 100.0
