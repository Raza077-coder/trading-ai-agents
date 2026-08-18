"""Trading AI Agents — FastAPI web service.

Exposes every agent in the suite as a REST API. Also ships a Mangum bridge
so the same app runs as a Vercel serverless function (see vercel.json).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from trading_agents.orchestrator import TradingOrchestrator

# ---------------------------------------------------------------------------
# App + shared orchestrator
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Trading AI Agents API",
    description=(
        "Production-grade multi-agent trading system — market data, technical "
        "analysis, sentiment, risk, backtesting, signals and portfolio management."
    ),
    version="1.0.0",
)

_orchestrator: Optional[TradingOrchestrator] = None


def get_orchestrator() -> TradingOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TradingOrchestrator()
        _orchestrator.initialize()
    return _orchestrator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to score")
    symbol: str = Field("GENERIC", description="Associated symbol")


class RiskRequest(BaseModel):
    symbol: str
    entry: float
    stop: float
    equity: float
    risk_per_trade_pct: Optional[float] = None


class BacktestRequest(BaseModel):
    symbol: str
    strategy: str = "sma_cross"
    period: str = "2y"


class RebalanceRequest(BaseModel):
    symbols: List[str]
    total_value: float
    method: str = "equal"
    tolerance_pct: float = 5.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": "Trading AI Agents",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, object]:
    return get_orchestrator().health()


@app.get("/quote/{symbol}")
def quote(symbol: str) -> Dict[str, object]:
    try:
        return get_orchestrator().get_quote(symbol).__dict__
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/candles/{symbol}")
def candles(
    symbol: str,
    period: str = Query("1y"),
    interval: str = Query("1d"),
    limit: Optional[int] = Query(None, ge=1, le=5000),
) -> List[Dict[str, object]]:
    try:
        bars = get_orchestrator().get_candles(symbol, period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    out = [
        {
            "timestamp": c.timestamp.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in (bars[-limit:] if limit else bars)
    ]
    return out


@app.get("/technical/{symbol}")
def technical(symbol: str) -> Dict[str, object]:
    try:
        return get_orchestrator().technical_analysis(symbol).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/sentiment")
def score_sentiment(req: SentimentRequest) -> Dict[str, object]:
    score = get_orchestrator().sentiment.analyze_text(req.text, symbol=req.symbol)
    return score.__dict__


@app.get("/sentiment/{symbol}")
def sentiment_symbol(symbol: str) -> Dict[str, object]:
    try:
        score = get_orchestrator().sentiment_analysis(symbol)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return score.__dict__


@app.post("/risk/position")
def risk_position(req: RiskRequest) -> Dict[str, object]:
    try:
        pos = get_orchestrator().size_position(
            req.symbol,
            entry=req.entry,
            stop=req.stop,
            equity=req.equity,
            risk_per_trade_pct=req.risk_per_trade_pct,
        )
        return pos.__dict__
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, object]:
    try:
        result = get_orchestrator().backtest_symbol(
            req.symbol, req.strategy, period=req.period
        )
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/portfolio/rebalance")
def portfolio_rebalance(req: RebalanceRequest) -> Dict[str, object]:
    try:
        orch = get_orchestrator()
        allocs = orch.portfolio_allocation(
            [s.upper() for s in req.symbols], req.total_value, method=req.method
        )
        quotes = {s: orch.get_quote(s) for s in req.symbols}
        plan = orch.portfolio.rebalance(allocs, quotes, tolerance_pct=req.tolerance_pct)
        return {
            "allocations": {
                s: a.to_dict() for s, a in allocs.items()
            },
            "rebalance": plan.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/analysis/{symbol}")
def full_analysis(symbol: str) -> Dict[str, object]:
    """Full pipeline: technical + sentiment + combined signal."""
    try:
        report = get_orchestrator().analyze(symbol)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "symbol": report["symbol"],
        "quote": report["quote"].__dict__,
        "technical": report["technical"].to_dict(),
        "sentiment": (
            report["sentiment"].__dict__ if report["sentiment"] is not None else None
        ),
        "signal": report["signal"].to_dict(),
    }


# ---------------------------------------------------------------------------
# Vercel serverless bridge
# ---------------------------------------------------------------------------

try:
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
except ImportError:  # pragma: no cover
    handler = None  # local dev only
