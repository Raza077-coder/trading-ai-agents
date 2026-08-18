"""Domain exceptions for the trading agent suite.

All exceptions inherit from :class:`TradingAgentError`, so callers can
catch a single base type and still inspect granular failure reasons.
"""


class TradingAgentError(Exception):
    """Base exception for all trading agent errors."""


class DataFetchError(TradingAgentError):
    """Raised when a market data provider fails to return data."""


class InvalidSymbolError(TradingAgentError):
    """Raised when a ticker symbol is invalid or unknown."""


class InsufficientDataError(TradingAgentError):
    """Raised when not enough data points are available for computation."""


class ProviderError(TradingAgentError):
    """Raised when a data provider is unavailable or misconfigured."""


class RiskLimitError(TradingAgentError):
    """Raised when a position/portfolio would breach configured risk limits."""


class SignalComputationError(TradingAgentError):
    """Raised when a signal cannot be computed from available inputs."""


class BacktestError(TradingAgentError):
    """Raised when the backtest engine cannot complete a run."""
