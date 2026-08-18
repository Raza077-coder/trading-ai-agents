"""Configuration dataclasses and loader for the trading agent suite.

Configuration is resolved in the following priority order:
    1. Environment variables (``TA_`` prefix)
    2. Values passed directly to the dataclasses
    3. Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Individual config blocks
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Base configuration shared by every agent."""

    name: str = "agent"
    log_level: str = field(default_factory=lambda: os.getenv("TA_LOG_LEVEL", "INFO"))
    timeout_seconds: float = float(os.getenv("TA_TIMEOUT_SECONDS", "30"))
    cache_ttl_seconds: int = int(os.getenv("TA_CACHE_TTL_SECONDS", "300"))
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for the market data layer."""

    provider: str = field(default_factory=lambda: os.getenv("TA_DATA_PROVIDER", "yfinance"))
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TA_ALPHAVANTAGE_KEY")
    )
    cache_dir: str = field(
        default_factory=lambda: os.getenv("TA_CACHE_DIR", ".cache/trading")
    )
    default_period: str = field(default_factory=lambda: os.getenv("TA_PERIOD", "1y"))
    default_interval: str = field(
        default_factory=lambda: os.getenv("TA_INTERVAL", "1d")
    )
    retries: int = int(os.getenv("TA_RETRIES", "3"))
    backoff_factor: float = float(os.getenv("TA_BACKOFF_FACTOR", "0.5"))


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    max_position_pct: float = float(os.getenv("TA_MAX_POSITION_PCT", "0.10"))
    max_portfolio_risk_pct: float = float(os.getenv("TA_MAX_PORTFOLIO_RISK_PCT", "0.02"))
    default_stop_loss_pct: float = float(os.getenv("TA_DEFAULT_STOP_LOSS_PCT", "0.05"))
    default_take_profit_pct: float = float(os.getenv("TA_DEFAULT_TAKE_PROFIT_PCT", "0.10"))
    max_leverage: float = float(os.getenv("TA_MAX_LEVERAGE", "1.0"))
    risk_free_rate: float = float(os.getenv("TA_RISK_FREE_RATE", "0.04"))
    atr_multiplier_stop: float = float(os.getenv("TA_ATR_MULT_STOP", "2.0"))
    atr_multiplier_target: float = float(os.getenv("TA_ATR_MULT_TARGET", "3.0"))


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""

    news_sources: list = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("TA_NEWS_SOURCES", "").split(",")
            if s.strip()
        ]
    )
    use_llm: bool = os.getenv("TA_SENTIMENT_USE_LLM", "false").lower() == "true"
    llm_model: str = field(default_factory=lambda: os.getenv("TA_LLM_MODEL", "gpt-4o-mini"))
    lexicon_path: Optional[str] = field(
        default_factory=lambda: os.getenv("TA_LEXICON_PATH")
    )
    lookback_days: int = int(os.getenv("TA_SENTIMENT_LOOKBACK_DAYS", "7"))


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""

    initial_capital: float = float(os.getenv("TA_BACKTEST_CAPITAL", "100000"))
    commission_pct: float = float(os.getenv("TA_COMMISSION_PCT", "0.001"))
    slippage_pct: float = float(os.getenv("TA_SLIPPAGE_PCT", "0.0005"))
    benchmark: str = field(default_factory=lambda: os.getenv("TA_BENCHMARK", "^GSPC"))
    risk_free_rate: float = float(os.getenv("TA_RISK_FREE_RATE", "0.04"))


# ---------------------------------------------------------------------------
# Bundle + loader
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Bundle of all configuration blocks."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables + defaults."""
        return cls()


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from an optional YAML/JSON file merged over env defaults.

    Args:
        path: Optional path to a YAML or JSON config file.

    Returns:
        A fully-populated :class:`Config`.
    """
    cfg = Config.from_env()
    if not path:
        return cfg

    import json

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore

            raw: Dict[str, Any] = yaml.safe_load(open(path, encoding="utf-8")) or {}
        except ImportError:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML config files")
    else:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    _apply(cfg.agent, raw.get("agent", {}))
    _apply(cfg.data, raw.get("data", {}))
    _apply(cfg.risk, raw.get("risk", {}))
    _apply(cfg.sentiment, raw.get("sentiment", {}))
    _apply(cfg.backtest, raw.get("backtest", {}))
    return cfg


def _apply(dc: Any, values: Dict[str, Any]) -> None:
    """Overwrite dataclass fields with values from a dict (type-coerced)."""
    for key, value in values.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(current, bool):
            value = str(value).lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(dc, key, value)
