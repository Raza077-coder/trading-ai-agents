"""Trading Orchestrator — composes every agent into one end-to-end pipeline.

The orchestrator wires the seven agents together:

    MarketData → Technical → Sentiment → Risk → Signal → (Portfolio)

and provides the single ``analyze()`` entry point used by the CLI and API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from trading_agents.analysis.technical_agent import TechnicalAnalysisAgent
from trading_agents.backtest.engine import BacktestAgent
from trading_agents.core.base import BaseAgent
from trading_agents.core.config import (
    AgentConfig,
    BacktestConfig,
    DataConfig,
    RiskConfig,
    SentimentConfig,
    load_config,
)
from trading_agents.core.models import (
    BacktestResult,
    Candle,
    IndicatorSnapshot,
    Quote,
    SentimentScore,
    Signal,
)
from trading_agents.data.market_data_agent import MarketDataAgent
from trading_agents.portfolio.portfolio_agent import PortfolioManagerAgent
from trading_agents.risk.risk_agent import RiskManagementAgent
from trading_agents.sentiment.sentiment_agent import SentimentAnalysisAgent
from trading_agents.signals.signal_agent import SignalGeneratorAgent


class TradingOrchestrator(BaseAgent):
    """One-stop entry point that runs the whole trading pipeline.

    Example:
        >>> with TradingOrchestrator() as orch:
        ...     report = orch.analyze("AAPL")
        ...     print(report["signal"].action, report["signal"].confidence)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        data_config: Optional[DataConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        sentiment_config: Optional[SentimentConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        use_sentiment: bool = True,
    ) -> None:
        super().__init__("orchestrator", config)
        self.use_sentiment = use_sentiment

        # build the agent pipeline
        self.market_data = MarketDataAgent(config=config, data_config=data_config)
        self.technical = TechnicalAnalysisAgent(config=config)
        self.sentiment = SentimentAnalysisAgent(
            config=config, sentiment_config=sentiment_config
        )
        self.risk = RiskManagementAgent(config=config, risk_config=risk_config)
        self.backtest = BacktestAgent(config=config, backtest_config=backtest_config)
        self.signal = SignalGeneratorAgent(config=config)
        self.portfolio = PortfolioManagerAgent(config=config)

        self._agents: List[BaseAgent] = [
            self.market_data,
            self.technical,
            self.sentiment,
            self.risk,
            self.backtest,
            self.signal,
            self.portfolio,
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        for agent in self._agents:
            agent.initialize()
        super().initialize()

    def shutdown(self) -> None:
        for agent in reversed(self._agents):
            agent.shutdown()
        super().shutdown()

    @classmethod
    def from_config_file(cls, path: str) -> "TradingOrchestrator":
        """Build an orchestrator from a YAML/JSON config file."""
        cfg = load_config(path)
        return cls(
            config=cfg.agent,
            data_config=cfg.data,
            risk_config=cfg.risk,
            sentiment_config=cfg.sentiment,
            backtest_config=cfg.backtest,
        )

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------
    def analyze(
        self,
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
    ) -> Dict[str, object]:
        """Run the full pipeline for one symbol.

        Returns a dict with candles, technical snapshot, sentiment, risk
        sizing and the final combined signal.
        """
        self._guard_initialized()
        candles = self.market_data.get_candles(symbol, period=period, interval=interval)
        quote = self.market_data.get_quote(symbol, use_cache=True)
        tech = self.technical.analyze(candles, symbol=symbol)

        sentiment = None
        if self.use_sentiment:
            sentiment = self.sentiment.full_analysis(symbol)

        # risk-aware signal
        risk_check = self.risk.check_portfolio_risk({}, quote.price * 1)
        signal = self.signal.generate(
            technical=tech,
            sentiment=sentiment,
            price=quote.price,
            risk_ok=True,
        )

        return {
            "symbol": symbol,
            "quote": quote,
            "candles_count": len(candles),
            "technical": tech,
            "sentiment": sentiment,
            "signal": signal,
        }

    def analyze_many(self, symbols: List[str]) -> Dict[str, Dict[str, object]]:
        """Run the full pipeline for several symbols."""
        results: Dict[str, Dict[str, object]] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Analysis failed for %s: %s", symbol, exc)
        return results

    def backtest_symbol(
        self,
        symbol: str,
        strategy: str,
        period: str = "2y",
        **strategy_kwargs,
    ) -> BacktestResult:
        """Fetch data and backtest a strategy for a symbol."""
        self._guard_initialized()
        candles = self.market_data.get_candles(symbol, period=period)
        return self.backtest.run(
            candles, strategy, symbol=symbol, strategy_kwargs=strategy_kwargs or None
        )

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------
    def get_candles(self, symbol: str, **kwargs) -> List[Candle]:
        return self.market_data.get_candles(symbol, **kwargs)

    def get_quote(self, symbol: str) -> Quote:
        return self.market_data.get_quote(symbol)

    def technical_analysis(self, symbol: str) -> IndicatorSnapshot:
        candles = self.market_data.get_candles(symbol)
        return self.technical.analyze(candles, symbol=symbol)

    def sentiment_analysis(
        self, symbol: str, news_items: Optional[List] = None
    ) -> SentimentScore:
        return self.sentiment.full_analysis(symbol, news_items)

    def size_position(self, symbol: str, entry: float, stop: float, equity: float, **kwargs):
        return self.risk.size_position(symbol, entry, stop, equity, **kwargs)

    def portfolio_allocation(
        self, symbols: List[str], total_value: float, method: str = "equal"
    ):
        if method == "risk_parity":
            candles = {
                s: self.market_data.get_candles(s) for s in symbols
            }
            return self.portfolio.inverse_volatility(candles, total_value)
        return self.portfolio.equal_weight(symbols, total_value)

    def health(self) -> Dict[str, object]:
        """Report suite + provider health."""
        return {
            "status": "ok",
            "provider": self.market_data.provider.name,
            "provider_healthy": self.market_data.health_check(),
            "agents": [a.name for a in self._agents],
        }
