"""
Trading AI Agents — a production-grade multi-agent trading system.

A suite of specialized, industry-level agents that collaborate to deliver
market data, technical analysis, sentiment scoring, risk management,
backtesting, trade signals and portfolio management.

Author: Ali Raza (Raza077-coder)
License: MIT
"""

__version__ = "1.0.0"
__all__ = [
    "MarketDataAgent",
    "TechnicalAnalysisAgent",
    "SentimentAnalysisAgent",
    "RiskManagementAgent",
    "BacktestAgent",
    "SignalGeneratorAgent",
    "PortfolioManagerAgent",
    "TradingOrchestrator",
]

from trading_agents.data.market_data_agent import MarketDataAgent
from trading_agents.analysis.technical_agent import TechnicalAnalysisAgent
from trading_agents.sentiment.sentiment_agent import SentimentAnalysisAgent
from trading_agents.risk.risk_agent import RiskManagementAgent
from trading_agents.backtest.engine import BacktestAgent
from trading_agents.signals.signal_agent import SignalGeneratorAgent
from trading_agents.portfolio.portfolio_agent import PortfolioManagerAgent
from trading_agents.orchestrator import TradingOrchestrator
