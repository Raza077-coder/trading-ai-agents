"""Trade Signal Generator — fuses technical, sentiment and risk inputs.

Each input contributes a weighted vote in [-1, +1]; the net score maps to
BUY / HOLD / SELL with a confidence derived from agreement between the
component signals. Optionally risk-adjusts the action (downgrades BUY to
HOLD when portfolio risk limits are tight).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig
from trading_agents.core.exceptions import SignalComputationError
from trading_agents.core.models import (
    IndicatorSnapshot,
    SentimentScore,
    Signal,
    SignalAction,
)

# Default vote weights — technical drivers dominate, sentiment is a tie-breaker
DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 0.30,      # moving average alignment
    "momentum": 0.25,   # MACD + RSI direction
    "mean_reversion": 0.15,  # Bollinger position
    "sentiment": 0.20,  # news/social score
    "volume": 0.10,     # volume confirmation
}


class SignalGeneratorAgent(BaseAgent):
    """Combine technical + sentiment inputs into a trade signal.

    Example:
        >>> with SignalGeneratorAgent() as agent:
        ...     sig = agent.generate(tech_snapshot, sentiment_score, price=180.0)
        ...     print(sig.action, sig.confidence)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = 0.25,
        sell_threshold: float = -0.25,
    ) -> None:
        super().__init__("signal_generator", config)
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        # normalize
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        technical: IndicatorSnapshot,
        sentiment: Optional[SentimentScore] = None,
        price: float = 0.0,
        risk_ok: bool = True,
    ) -> Signal:
        """Produce a combined signal for a symbol.

        Args:
            technical: Snapshot from the technical analysis agent.
            sentiment: Optional score from the sentiment agent.
            price: Current price (defaults to last close if known).
            risk_ok: False if portfolio risk checks failed (downgrades signal).

        Raises:
            SignalComputationError: If no usable inputs are provided.
        """
        self._guard_initialized()
        symbol = technical.symbol

        votes: Dict[str, float] = {}
        votes["trend"] = self._vote_trend(technical)
        votes["momentum"] = self._vote_momentum(technical)
        votes["mean_reversion"] = self._vote_mean_reversion(technical)
        votes["volume"] = self._vote_volume(technical)
        if sentiment is not None:
            votes["sentiment"] = sentiment.score
        else:
            votes["sentiment"] = 0.0

        strength = sum(self.weights[k] * votes[k] for k in self.weights)
        # cap at ±3 for display
        strength = max(-3.0, min(3.0, strength * 3.0))

        if strength >= self.buy_threshold * 3:
            action = SignalAction.BUY
        elif strength <= self.sell_threshold * 3:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        # risk adjustment: never BUY/SELL if portfolio risk is breached
        risk_adjusted = False
        if action != SignalAction.HOLD and not risk_ok:
            action = SignalAction.HOLD
            risk_adjusted = True

        confidence = self._confidence(votes)
        price = price or technical.indicators.get("close", 0.0) or 0.0

        rationale = self._rationale(votes, action, risk_adjusted)
        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            price=price,
            rationale=rationale,
            components={k: round(v, 3) for k, v in votes.items()},
            risk_adjusted=risk_adjusted,
        )

    def generate_from_components(
        self,
        symbol: str,
        votes: Dict[str, float],
        price: float = 0.0,
        risk_ok: bool = True,
    ) -> Signal:
        """Build a signal directly from component votes (advanced use)."""
        strength = sum(self.weights[k] * votes.get(k, 0.0) for k in self.weights)
        strength = max(-3.0, min(3.0, strength * 3.0))
        if strength >= self.buy_threshold * 3:
            action = SignalAction.BUY
        elif strength <= self.sell_threshold * 3:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
        risk_adjusted = False
        if action != SignalAction.HOLD and not risk_ok:
            action = SignalAction.HOLD
            risk_adjusted = True
        return Signal(
            symbol=symbol,
            action=action,
            confidence=self._confidence(votes),
            strength=strength,
            price=price,
            rationale=self._rationale(votes, action, risk_adjusted),
            components={k: round(v, 3) for k, v in votes.items()},
            risk_adjusted=risk_adjusted,
        )

    # ------------------------------------------------------------------
    # Component votes
    # ------------------------------------------------------------------
    @staticmethod
    def _vote_trend(tech: IndicatorSnapshot) -> float:
        vote = 0.0
        if tech.signals.get("sma_trend") == "golden_cross":
            vote += 0.5
        elif tech.signals.get("sma_trend") == "death_cross":
            vote -= 0.5
        if tech.signals.get("ema_trend") == "bullish":
            vote += 0.5
        elif tech.signals.get("ema_trend") == "bearish":
            vote -= 0.5
        return max(-1.0, min(1.0, vote))

    @staticmethod
    def _vote_momentum(tech: IndicatorSnapshot) -> float:
        vote = 0.0
        if tech.signals.get("macd_signal") == "bullish":
            vote += 0.5
        elif tech.signals.get("macd_signal") == "bearish":
            vote -= 0.5
        rsi = tech.indicators.get("rsi", 50.0)
        if rsi < 30:
            vote += 0.4  # oversold bounce potential
        elif rsi > 70:
            vote -= 0.4
        elif rsi < 45:
            vote += 0.15
        elif rsi > 55:
            vote -= 0.15
        return max(-1.0, min(1.0, vote))

    @staticmethod
    def _vote_mean_reversion(tech: IndicatorSnapshot) -> float:
        sig = tech.signals.get("bb_signal")
        if sig == "below_lower":
            return 0.7
        if sig == "above_upper":
            return -0.7
        price = tech.indicators.get("close", 0.0)
        lower = tech.indicators.get("bb_lower", 0.0)
        upper = tech.indicators.get("bb_upper", 0.0)
        if lower and upper:
            position = (price - lower) / (upper - lower) if upper > lower else 0.5
            return (0.5 - position) * 2.0  # low → +1, high → -1
        return 0.0

    @staticmethod
    def _vote_volume(tech: IndicatorSnapshot) -> float:
        ratio = tech.indicators.get("volume_ratio", 1.0)
        if ratio > 1.5:
            return 0.4  # heavy volume confirms the move
        if ratio < 0.5:
            return -0.2
        return 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _confidence(self, votes: Dict[str, float]) -> float:
        """Agreement-weighted confidence in [0, 1]."""
        signed = [v for v in votes.values() if abs(v) > 0.15]
        if not signed:
            return 0.1
        mean = sum(signed) / len(signed)
        agreement = 1.0 - (sum(abs(v - mean) for v in signed) / len(signed))
        magnitude = min(1.0, abs(mean))
        return round(max(0.05, min(1.0, 0.5 * agreement + 0.5 * magnitude)), 3)

    @staticmethod
    def _rationale(
        votes: Dict[str, float],
        action: SignalAction,
        risk_adjusted: bool,
    ) -> str:
        parts = []
        for name, vote in sorted(votes.items(), key=lambda kv: abs(kv[1]), reverse=True):
            if vote > 0.3:
                parts.append(f"{name} bullish ({vote:+.2f})")
            elif vote < -0.3:
                parts.append(f"{name} bearish ({vote:+.2f})")
        base = (
            f"{action.value} — " + ("; ".join(parts) if parts else "no strong drivers")
        )
        if risk_adjusted:
            base += " [downgraded to HOLD by risk control]"
        return base
