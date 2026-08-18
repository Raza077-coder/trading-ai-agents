"""Technical Analysis Agent — computes indicators and rule-based signals.

Produces an :class:`IndicatorSnapshot` per symbol with RSI, MACD, moving
averages, Bollinger Bands, ATR and Stochastic readings, plus a plain-English
summary of the technical picture.
"""

from __future__ import annotations

from typing import List, Optional

from trading_agents.analysis import indicators as ta
from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig
from trading_agents.core.exceptions import InsufficientDataError
from trading_agents.core.models import Candle, IndicatorSnapshot


class TechnicalAnalysisAgent(BaseAgent):
    """Compute technical indicators and rule-based signals for a symbol.

    Example:
        >>> with TechnicalAnalysisAgent() as agent:
        ...     snap = agent.analyze(candles, symbol="AAPL")
        ...     print(snap.summary)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        sma_short: int = 20,
        sma_long: int = 50,
        ema_short: int = 12,
        ema_long: int = 26,
        atr_period: int = 14,
        stoch_period: int = 14,
    ) -> None:
        super().__init__("technical_analysis", config)
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.atr_period = atr_period
        self.stoch_period = stoch_period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, candles: List[Candle], symbol: Optional[str] = None) -> IndicatorSnapshot:
        """Analyze a candle series and produce a technical snapshot.

        Args:
            candles: OHLCV bars (oldest → newest).
            symbol: Symbol name (defaults to ``"UNKNOWN"``).

        Raises:
            InsufficientDataError: If fewer than ~60 bars are provided.
        """
        self._guard_initialized()
        if not candles or len(candles) < 60:
            raise InsufficientDataError(
                f"Need at least 60 candles for reliable indicators, got {len(candles or [])}"
            )
        symbol = symbol or "UNKNOWN"

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        snap = IndicatorSnapshot(symbol=symbol)
        last_price = closes[-1]

        # --- RSI ---
        rsi_series = ta.rsi(closes, self.rsi_period)
        rsi_now = ta.last_valid(rsi_series)
        snap.add("rsi", rsi_now or 50.0)
        if rsi_now is not None:
            if rsi_now >= 70:
                snap.add("rsi_signal", "overbought")
            elif rsi_now <= 30:
                snap.add("rsi_signal", "oversold")
            else:
                snap.add("rsi_signal", "neutral")

        # --- MACD ---
        macd_line, signal_line, hist = ta.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        m = ta.last_valid(macd_line)
        s = ta.last_valid(signal_line)
        h = ta.last_valid(hist)
        snap.add("macd", m or 0.0)
        snap.add("macd_signal", s or 0.0)
        snap.add("macd_histogram", h or 0.0)
        if m is not None and s is not None:
            snap.add("macd_signal", "bullish" if m > s else "bearish")

        # --- Moving averages ---
        sma_s = ta.sma(closes, self.sma_short)
        sma_l = ta.sma(closes, self.sma_long)
        ema_s = ta.ema(closes, self.ema_short)
        ema_l = ta.ema(closes, self.ema_long)
        sma_s_now = ta.last_valid(sma_s)
        sma_l_now = ta.last_valid(sma_l)
        ema_s_now = ta.last_valid(ema_s)
        ema_l_now = ta.last_valid(ema_l)
        snap.add("sma_short", sma_s_now or last_price)
        snap.add("sma_long", sma_l_now or last_price)
        snap.add("ema_short", ema_s_now or last_price)
        snap.add("ema_long", ema_l_now or last_price)
        if sma_s_now is not None and sma_l_now is not None:
            snap.add(
                "sma_trend",
                "golden_cross" if sma_s_now > sma_l_now else "death_cross",
            )
        if ema_s_now is not None and ema_l_now is not None:
            snap.add(
                "ema_trend",
                "bullish" if ema_s_now > ema_l_now else "bearish",
            )

        # --- Bollinger Bands ---
        bb_u, bb_m, bb_l = ta.bollinger_bands(closes, self.bb_period, self.bb_std)
        u = ta.last_valid(bb_u)
        mid = ta.last_valid(bb_m)
        lo = ta.last_valid(bb_l)
        snap.add("bb_upper", u or last_price)
        snap.add("bb_middle", mid or last_price)
        snap.add("bb_lower", lo or last_price)
        if u is not None and lo is not None:
            if last_price > u:
                snap.add("bb_signal", "above_upper")
            elif last_price < lo:
                snap.add("bb_signal", "below_lower")
            else:
                snap.add("bb_signal", "within")

        # --- ATR ---
        atr_series = ta.atr(highs, lows, closes, self.atr_period)
        atr_now = ta.last_valid(atr_series)
        snap.add("atr", atr_now or 0.0)

        # --- Stochastic ---
        k, d = ta.stochastic(highs, lows, closes, self.stoch_period)
        k_now = ta.last_valid(k)
        d_now = ta.last_valid(d)
        snap.add("stoch_k", k_now or 50.0)
        snap.add("stoch_d", d_now or 50.0)
        if k_now is not None and d_now is not None:
            if k_now > 80:
                snap.add("stoch_signal", "overbought")
            elif k_now < 20:
                snap.add("stoch_signal", "oversold")
            else:
                snap.add("stoch_signal", "neutral")

        # --- Volume trend ---
        vol_now = volumes[-1]
        vol_avg = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        snap.add("volume_ratio", vol_now / vol_avg if vol_avg else 1.0)

        snap.summary = self._summarize(snap, last_price)
        return snap

    def compute_indicators(self, candles: List[Candle]) -> dict:
        """Raw indicator series for charting (pandas-compatible lists)."""
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        macd_line, signal_line, hist = ta.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal
        )
        bb_u, bb_m, bb_l = ta.bollinger_bands(closes, self.bb_period, self.bb_std)
        return {
            "close": closes,
            "sma_20": ta.sma(closes, 20),
            "sma_50": ta.sma(closes, 50),
            "ema_12": ta.ema(closes, 12),
            "ema_26": ta.ema(closes, 26),
            "rsi": ta.rsi(closes, self.rsi_period),
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
            "bb_upper": bb_u,
            "bb_middle": bb_m,
            "bb_lower": bb_l,
            "atr": ta.atr(highs, lows, closes, self.atr_period),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(snap: IndicatorSnapshot, last_price: float) -> str:
        parts: List[str] = []
        rsi_val = snap.indicators.get("rsi", 50.0)
        if rsi_val >= 70:
            parts.append(f"RSI at {rsi_val:.0f} suggests overbought")
        elif rsi_val <= 30:
            parts.append(f"RSI at {rsi_val:.0f} suggests oversold")
        else:
            parts.append(f"RSI at {rsi_val:.0f} is neutral")

        if snap.signals.get("macd_signal") == "bullish":
            parts.append("MACD is bullish (line above signal)")
        elif snap.signals.get("macd_signal") == "bearish":
            parts.append("MACD is bearish (line below signal)")

        if snap.signals.get("sma_trend") == "golden_cross":
            parts.append("short SMA above long SMA (uptrend)")
        elif snap.signals.get("sma_trend") == "death_cross":
            parts.append("short SMA below long SMA (downtrend)")

        bb_sig = snap.signals.get("bb_signal")
        if bb_sig == "above_upper":
            parts.append("price pierced the upper Bollinger Band")
        elif bb_sig == "below_lower":
            parts.append("price pierced the lower Bollinger Band")
        else:
            parts.append("price inside Bollinger Bands")

        return " | ".join(parts) if parts else f"No clear technical setup at {last_price:.2f}"
