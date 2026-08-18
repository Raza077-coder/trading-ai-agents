"""End-to-end smoke tests for the trading agent suite (offline, synthetic data).

Run with:
    TA_DATA_PROVIDER=synthetic python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("TA_DATA_PROVIDER", "synthetic")

import pytest

from trading_agents.analysis import indicators as ta
from trading_agents.analysis.technical_agent import TechnicalAnalysisAgent
from trading_agents.backtest.engine import BacktestAgent
from trading_agents.core.models import SignalAction
from trading_agents.data.market_data_agent import MarketDataAgent
from trading_agents.portfolio.portfolio_agent import PortfolioManagerAgent
from trading_agents.risk.risk_agent import RiskManagementAgent
from trading_agents.sentiment.sentiment_agent import SentimentAnalysisAgent
from trading_agents.signals.signal_agent import SignalGeneratorAgent
from trading_agents.orchestrator import TradingOrchestrator

SYMBOL = "AAPL"


@pytest.fixture(scope="module")
def candles():
    with MarketDataAgent() as agent:
        return agent.get_candles(SYMBOL, period="2y")


def test_indicators_rsi_bounds(candles):
    closes = [c.close for c in candles]
    rsi_series = ta.rsi(closes, 14)
    valid = [v for v in rsi_series if not math.isnan(v)]
    assert len(valid) > 100
    assert all(0 <= v <= 100 for v in valid)


def test_indicators_macd_aligned(candles):
    closes = [c.close for c in candles]
    macd_line, signal_line, hist = ta.macd(closes)
    assert len(macd_line) == len(closes)
    assert len(signal_line) == len(closes)
    assert len(hist) == len(closes)


def test_market_data_agent(candles):
    assert len(candles) > 100
    assert all(c.low <= c.close <= c.high for c in candles)


def test_technical_agent(candles):
    with TechnicalAnalysisAgent() as agent:
        snap = agent.analyze(candles, symbol=SYMBOL)
    assert snap.symbol == SYMBOL
    for key in ("rsi", "macd", "sma_short", "bb_upper", "atr"):
        assert key in snap.indicators
    assert snap.summary


def test_sentiment_agent():
    with SentimentAnalysisAgent() as agent:
        bull = agent.analyze_text("Apple beats earnings, stock surges higher")
        bear = agent.analyze_text("Company misses targets, plunges on fraud probe")
    assert bull.score > 0
    assert bear.score < 0
    assert bull.label.value == "bullish"
    assert bear.label.value == "bearish"


def test_risk_agent(candles):
    with RiskManagementAgent() as agent:
        pos = agent.size_position(
            symbol=SYMBOL, entry=180.0, stop=171.0, equity=100_000
        )
        var = agent.estimate_var(candles, notional=10_000)
    assert pos.quantity > 0
    assert pos.stop_loss < pos.take_profit
    assert var["var_amount"] >= 0


def test_backtest_agent(candles):
    with BacktestAgent() as agent:
        result = agent.run(candles, "sma_cross", symbol=SYMBOL)
    assert result.num_trades > 0
    assert result.final_equity > 0
    assert result.sharpe_ratio != 0 or result.total_return_pct != 0


def test_signal_generator(candles):
    with TechnicalAnalysisAgent() as tech, SentimentAnalysisAgent() as sent, SignalGeneratorAgent() as sig:
        snap = tech.analyze(candles, symbol=SYMBOL)
        score = sent.analyze_text("strong growth and momentum", symbol=SYMBOL)
        signal = sig.generate(snap, score, price=candles[-1].close)
    assert signal.symbol == SYMBOL
    assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
    assert 0.0 <= signal.confidence <= 1.0


def test_portfolio_agent():
    with PortfolioManagerAgent() as agent:
        allocs = agent.equal_weight(["AAPL", "MSFT", "GOOG"], 300_000)
        assert abs(sum(a.weight for a in allocs.values()) - 1.0) < 1e-9
        plan = agent.rebalance(allocs, {})  # no quotes → no trades
    assert plan.total_turnover == 0


def test_orchestrator_full_pipeline():
    with TradingOrchestrator() as orch:
        report = orch.analyze(SYMBOL)
        health = orch.health()
    assert report["symbol"] == SYMBOL
    assert report["signal"].action in (
        SignalAction.BUY,
        SignalAction.SELL,
        SignalAction.HOLD,
    )
    assert health["status"] == "ok"
    assert len(health["agents"]) == 7
