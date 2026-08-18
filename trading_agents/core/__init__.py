"""Core framework: base agent, configuration, exceptions, logging, models."""

from trading_agents.core.base import BaseAgent
from trading_agents.core.config import (
    AgentConfig,
    DataConfig,
    RiskConfig,
    SentimentConfig,
    BacktestConfig,
    load_config,
)
from trading_agents.core.exceptions import (
    TradingAgentError,
    DataFetchError,
    InvalidSymbolError,
    InsufficientDataError,
    ProviderError,
    RiskLimitError,
)
from trading_agents.core.models import (
    Candle,
    Quote,
    IndicatorSnapshot,
    NewsItem,
    SentimentScore,
    PositionSize,
    Trade,
    BacktestResult,
    Signal,
)

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "DataConfig",
    "RiskConfig",
    "SentimentConfig",
    "BacktestConfig",
    "load_config",
    "TradingAgentError",
    "DataFetchError",
    "InvalidSymbolError",
    "InsufficientDataError",
    "ProviderError",
    "RiskLimitError",
    "Candle",
    "Quote",
    "IndicatorSnapshot",
    "NewsItem",
    "SentimentScore",
    "PositionSize",
    "Trade",
    "BacktestResult",
    "Signal",
]
