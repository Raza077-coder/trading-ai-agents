"""Typed data models shared across the trading agent suite.

These dataclasses form the contract between agents: the orchestrator pipes
:class:`Candle` frames from the data agent into the technical agent, wraps the
outputs into :class:`IndicatorSnapshot`, and finally produces a :class:`Signal`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalAction(str, Enum):
    """The three possible trade recommendations."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    """Direction of a trade order."""

    LONG = "LONG"
    SHORT = "SHORT"


class SentimentLabel(str, Enum):
    """Coarse sentiment buckets."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Quote:
    """A live quote for a symbol."""

    symbol: str
    price: float
    change: float
    change_pct: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class IndicatorSnapshot:
    """Result of the technical analysis agent for one symbol at a point in time."""

    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    indicators: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, str] = field(default_factory=dict)
    summary: str = ""

    def add(self, name: str, value: float, signal: Optional[str] = None) -> None:
        self.indicators[name] = value
        if signal:
            self.signals[name] = signal

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "indicators": self.indicators,
            "signals": self.signals,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsItem:
    """A single news headline with metadata."""

    title: str
    source: str = "unknown"
    url: str = ""
    published_at: Optional[datetime] = None
    body: str = ""


@dataclass
class SentimentScore:
    """Aggregated sentiment for a symbol."""

    symbol: str
    score: float  # -1.0 (very bearish) … +1.0 (very bullish)
    label: SentimentLabel
    headline_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    sample: List[NewsItem] = field(default_factory=list)
    confidence: float = 0.0
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def magnitude(self) -> float:
        """Absolute value of the score (signal strength)."""
        return abs(self.score)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSize:
    """The recommended position size for a trade."""

    symbol: str
    quantity: float
    notional: float
    risk_amount: float
    stop_loss: float
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    rationale: str = ""


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    """A closed (or open) trade produced by the backtest engine."""

    symbol: str
    side: OrderSide
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Full result of a backtest run with performance metrics."""

    symbol: str
    strategy_name: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int
    num_wins: int
    num_losses: int
    avg_trade_pct: float
    benchmark_return_pct: Optional[float] = None
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 3),
            "num_trades": self.num_trades,
            "benchmark_return_pct": (
                round(self.benchmark_return_pct, 2)
                if self.benchmark_return_pct is not None
                else None
            ),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A combined trading recommendation from the signal generator."""

    symbol: str
    action: SignalAction
    confidence: float  # 0.0 … 1.0
    price: float = 0.0
    strength: float = 0.0  # net weighted vote, typically -3 … +3
    rationale: str = ""
    components: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    risk_adjusted: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(self.confidence, 3),
            "strength": round(self.strength, 3),
            "price": round(self.price, 4),
            "rationale": self.rationale,
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "timestamp": self.timestamp.isoformat(),
            "risk_adjusted": self.risk_adjusted,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with a fallback when the denominator is zero/NaN."""
    try:
        if denominator is None or math.isclose(denominator, 0.0):
            return default
        result = numerator / denominator
        return result if math.isfinite(result) else default
    except (TypeError, ZeroDivisionError):
        return default
