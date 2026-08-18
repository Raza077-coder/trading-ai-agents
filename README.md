<div align="center">

# 📈 Trading AI Agents

**Production-grade multi-agent trading system — 7 specialized agents working together**
Market Data · Technical Analysis · Sentiment · Risk Management · Backtesting · Signal Generation · Portfolio Management

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Tests](https://img.shields.io/badge/tests-9%2F9%20passing-brightgreen)

</div>

---

## 🧠 Overview

**Trading AI Agents** is an industry-grade, multi-agent trading framework. Each agent is a
self-contained, production-quality component with typed interfaces, structured logging,
configuration via environment variables, and graceful error handling. Agents are composed by a
**Trading Orchestrator** that runs the full pipeline — from raw market data to a final
BUY / SELL / HOLD recommendation with confidence scoring and risk controls.

The suite works out of the box in **offline/demo mode** (deterministic synthetic data provider)
and switches to **live data** (yfinance or Alpha Vantage) with a single environment variable.

---

## 🏗️ Architecture

```
                        ┌─────────────────────────┐
                        │   Trading Orchestrator  │
                        │   (pipeline composer)   │
                        └────────────┬────────────┘
                                     │
        ┌──────────┬──────────┬──────┴───────┬──────────┬───────────┬──────────┐
        ▼          ▼          ▼              ▼          ▼           ▼          ▼
 ┌────────────┐ ┌─────────┐ ┌───────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
 │   1.       │ │  2.     │ │   3.      │ │  4.     │ │   5.     │ │  6.    │ │   7.     │
 │ Market     │→│Technical│→│ Sentiment │ │ Risk    │ │ Backtest │→│ Signal │ │ Portfolio│
 │ Data       │ │Analysis │ │ Analysis  │ │ Mgmt    │ │ Engine   │ │ Gen    │ │ Manager  │
 └─────┬──────┘ └────┬────┘ └─────┬─────┘ └────┬────┘ └────┬─────┘ └───┬────┘ └────┬─────┘
       │             │            │            │          │           │          │
       ▼             ▼            ▼            ▼          ▼           ▼          ▼
   yfinance /   RSI · MACD ·  lexicon +    position   SMA/RSI/    weighted   equal-weight /
   AlphaVant. /  SMA/EMA ·     news API    sizing ·    MACD/BB     votes →    risk-parity /
   synthetic    Bollinger ·               stop-loss · strategies   BUY/SELL/  rebalance
   providers    ATR · Stoch               VaR                     HOLD       engine
```

**Data flow (end-to-end):** `MarketDataAgent` fetches candles → `TechnicalAnalysisAgent`
computes indicators → `SentimentAnalysisAgent` scores news → `RiskManagementAgent` sizes the
position & checks limits → `SignalGeneratorAgent` fuses everything into a weighted
BUY / SELL / HOLD signal → `PortfolioManagerAgent` allocates & rebalances across assets.
`BacktestAgent` validates any strategy on historical data before it goes live.

---

## 🤖 The 7 Agents

| # | Agent | Purpose | Key Inputs | Outputs |
|---|-------|---------|-----------|---------|
| 1 | **Market Data Agent** | Fetches OHLCV candles & live quotes from pluggable providers (yfinance, Alpha Vantage, synthetic) with caching + retries | symbol, period, interval | `List[Candle]`, `Quote` |
| 2 | **Technical Analysis Agent** | Computes RSI, MACD, SMA/EMA, Bollinger Bands, ATR, Stochastic; rule-based signals | candles | `IndicatorSnapshot` (indicators + signals + summary) |
| 3 | **Sentiment Analysis Agent** | Finance-aware lexicon scoring with negation/intensity handling; optional NewsAPI fetch | headlines / text | `SentimentScore` (-1 … +1, label, confidence) |
| 4 | **Risk Management Agent** | Fixed-fractional & ATR position sizing, stop-loss/take-profit, portfolio risk checks, historical VaR | entry, stop, equity, candles | `PositionSize`, risk report, VaR |
| 5 | **Strategy Backtesting Agent** | Simulates long-only strategies with commission + slippage; Sharpe, Sortino, drawdown, win rate, profit factor | candles, strategy name | `BacktestResult` (full metrics) |
| 6 | **Trade Signal Generator** | Weighted vote fusion of trend, momentum, mean-reversion, volume, sentiment → action + confidence | technical snapshot, sentiment | `Signal` (BUY/SELL/HOLD) |
| 7 | **Portfolio Manager Agent** | Equal-weight & inverse-volatility (risk-parity) allocation, drift-based rebalancing, portfolio stats | symbols, quotes, candles | allocations, `RebalancePlan`, stats |

Plus **TradingOrchestrator** — one-stop entry point that composes all 7 into a single
`analyze(symbol)` pipeline, and a **FastAPI web service** exposing every agent via REST.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/Raza077-coder/trading-ai-agents.git
cd trading-ai-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the CLI (offline demo mode — no API keys needed)

```bash
# Full analysis pipeline for AAPL (synthetic data)
TA_DATA_PROVIDER=synthetic python -m trading_agents.cli analyze AAPL

# Live data (default provider)
python -m trading_agents.cli analyze AAPL

# Live quote
python -m trading_agents.cli quote AAPL

# Backtest the SMA-cross strategy
python -m trading_agents.cli backtest AAPL --strategy sma_cross

# Sentiment on a headline
python -m trading_agents.cli sentiment "Apple beats earnings, stock surges"

# Position sizing + VaR
python -m trading_agents.cli risk AAPL --equity 100000

# Risk-parity allocation + rebalance plan
python -m trading_agents.cli portfolio AAPL,MSFT,GOOG --value 100000
```

### 3. Run the API locally

```bash
uvicorn api.main:app --reload
# Swagger docs → http://localhost:8000/docs
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Suite health + provider check |
| GET | `/quote/{symbol}` | Live quote |
| GET | `/candles/{symbol}?period=&interval=&limit=` | OHLCV bars |
| GET | `/analysis/{symbol}` | **Full pipeline** (technical + sentiment + signal) |
| GET | `/technical/{symbol}` | Indicator snapshot only |
| POST | `/sentiment` | Score a text (`{"text": "...", "symbol": "AAPL"}`) |
| GET | `/sentiment/{symbol}` | Aggregate news sentiment for a symbol |
| POST | `/risk/position` | Position sizing (`symbol, entry, stop, equity, risk_pct`) |
| POST | `/backtest` | Run a strategy backtest |
| POST | `/portfolio/rebalance` | Allocation + rebalance plan |

Example:

```bash
curl -s "http://localhost:8000/analysis/AAPL" | python -m json.tool
```

---

## ⚙️ Configuration

All configuration is **environment-driven** (prefix `TA_`) with sane defaults —
no config file required. A `config.example.yaml` + `.env.example` are included.

| Variable | Default | Description |
|----------|---------|-------------|
| `TA_DATA_PROVIDER` | `yfinance` | `yfinance` \| `alpha_vantage` \| `synthetic` |
| `TA_ALPHAVANTAGE_KEY` | — | Required for Alpha Vantage provider |
| `TA_MAX_POSITION_PCT` | `0.10` | Max position size as % of equity |
| `TA_MAX_PORTFOLIO_RISK_PCT` | `0.02` | Max risk per trade as % of equity |
| `TA_ATR_MULT_STOP` / `TA_ATR_MULT_TARGET` | `2.0` / `3.0` | ATR-based stop/target multiples |
| `TA_BACKTEST_CAPITAL` | `100000` | Default backtest capital |
| `TA_COMMISSION_PCT` / `TA_SLIPPAGE_PCT` | `0.001` / `0.0005` | Execution costs |
| `NEWS_API_KEY` | — | Optional — enables live news sentiment |
| `TA_LOG_LEVEL` | `INFO` | Logging verbosity |

```bash
cp .env.example .env   # then edit
set -a && source .env && set +a
```

---

## 🧪 Testing

```bash
TA_DATA_PROVIDER=synthetic pytest tests/ -v
# 9 passed — covers every agent, indicators, and the full orchestrator pipeline
```

---

## ☁️ Deployment

### Vercel

A `vercel.json` + FastAPI wrapper (`api/main.py`, with a Mangum serverless bridge) are included:

```bash
vercel --prod
```

> **Note:** live deployment was prepared (config + wrapper) but not executed from this
> workspace — no Vercel deployment credential was available at build time. Push the repo to
> your GitHub and import it in the Vercel dashboard, or run `vercel --prod` locally.

### Any ASGI host

```bash
pip install uvicorn && uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker (recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📁 Project Structure

```
trading-ai-agents/
├── api/
│   └── main.py                 # FastAPI web service (+ Vercel serverless bridge)
├── trading_agents/
│   ├── core/                   # BaseAgent, config, exceptions, models, logging
│   ├── data/                   # Market Data Agent + providers
│   ├── analysis/               # Technical Analysis Agent + indicator math
│   ├── sentiment/              # Sentiment Analysis Agent
│   ├── risk/                   # Risk Management Agent
│   ├── backtest/               # Backtesting engine + strategies
│   ├── signals/                # Trade Signal Generator
│   ├── portfolio/              # Portfolio Manager Agent
│   ├── orchestrator.py         # Trading Orchestrator (pipeline)
│   └── cli.py                  # Command-line interface
├── tests/
│   └── test_suite.py           # 9 end-to-end tests
├── requirements.txt
├── pyproject.toml
├── config.example.yaml
├── .env.example
└── vercel.json
```

---

## ⚠️ Disclaimer

This software is provided **for educational and research purposes only**. It is **not**
financial advice. Trading involves substantial risk of loss. Always do your own research and
consult a licensed financial advisor before making investment decisions. The authors are not
responsible for any financial losses incurred through use of this software.

---

## 📄 License

MIT © [Ali Raza](https://github.com/Raza077-coder)
