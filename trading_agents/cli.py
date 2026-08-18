"""Command-line interface for the trading agent suite.

Examples:
    python -m trading_agents.cli analyze AAPL
    python -m trading_agents.cli quote AAPL
    python -m trading_agents.cli backtest AAPL --strategy sma_cross
    python -m trading_agents.cli sentiment "Apple beats earnings"
    python -m trading_agents.cli risk AAPL --equity 100000
    python -m trading_agents.cli portfolio AAPL,MSFT,GOOG --value 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from trading_agents.core.logging_setup import setup_logging
from trading_agents.orchestrator import TradingOrchestrator


def _json(obj) -> str:
    """Pretty-print an object that has to_dict() or is JSON-serializable."""
    if hasattr(obj, "to_dict"):
        return json.dumps(obj.to_dict(), indent=2, default=str)
    return json.dumps(obj, indent=2, default=str)


def cmd_analyze(orch: TradingOrchestrator, args) -> int:
    report = orch.analyze(args.symbol, period=args.period, interval=args.interval)
    print(f"\n{'='*60}")
    print(f"  {report['symbol']} @ {report['quote'].price:.2f} "
          f"({report['quote'].change_pct:+.2f}%)")
    print(f"{'='*60}")
    print("\n--- TECHNICAL ---")
    print(_json(report["technical"]))
    if report["sentiment"] is not None:
        print("\n--- SENTIMENT ---")
        print(_json(report["sentiment"]))
    print("\n--- SIGNAL ---")
    print(_json(report["signal"]))
    return 0


def cmd_quote(orch: TradingOrchestrator, args) -> int:
    quote = orch.get_quote(args.symbol)
    print(_json(quote))
    return 0


def cmd_backtest(orch: TradingOrchestrator, args) -> int:
    result = orch.backtest_symbol(
        args.symbol,
        args.strategy,
        period=args.period,
    )
    print(_json(result))
    return 0


def cmd_sentiment(orch: TradingOrchestrator, args) -> int:
    score = orch.sentiment.analyze_text(args.text, symbol=args.symbol)
    print(_json(score))
    return 0


def cmd_risk(orch: TradingOrchestrator, args) -> int:
    quote = orch.get_quote(args.symbol)
    stop = orch.risk.suggested_stop(orch.get_candles(args.symbol))
    pos = orch.size_position(
        args.symbol, entry=quote.price, stop=stop, equity=args.equity
    )
    print(_json(pos))
    candles = orch.get_candles(args.symbol)
    var = orch.risk.estimate_var(candles, pos.notional)
    print("\n--- VAR ESTIMATE (95%, 1d) ---")
    print(_json(var))
    return 0


def cmd_portfolio(orch: TradingOrchestrator, args) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    allocs = orch.portfolio_allocation(
        symbols, args.value, method=args.method
    )
    quotes = {s: orch.get_quote(s) for s in symbols if s}
    plan = orch.portfolio.rebalance(allocs, quotes)
    print("\n--- ALLOCATION ---")
    for sym, alloc in allocs.items():
        print(f"  {sym}: {alloc.weight*100:.2f}%  (${alloc.value:,.0f})")
    print("\n--- REBALANCE PLAN ---")
    print(_json(plan))
    return 0


def cmd_health(orch: TradingOrchestrator, args) -> int:
    print(_json(orch.health()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-agents",
        description="Trading AI Agents — multi-agent trading suite",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--config", help="Path to YAML/JSON config file")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Full pipeline for a symbol")
    p.add_argument("symbol")
    p.add_argument("--period", default=None)
    p.add_argument("--interval", default=None)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("quote", help="Live quote")
    p.add_argument("symbol")
    p.set_defaults(func=cmd_quote)

    p = sub.add_parser("backtest", help="Backtest a strategy")
    p.add_argument("symbol")
    p.add_argument("--strategy", default="sma_cross",
                   choices=["sma_cross", "rsi_mean_reversion", "macd", "bollinger"])
    p.add_argument("--period", default="2y")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("sentiment", help="Score text sentiment")
    p.add_argument("text")
    p.add_argument("--symbol", default="GENERIC")
    p.set_defaults(func=cmd_sentiment)

    p = sub.add_parser("risk", help="Position sizing + VaR")
    p.add_argument("symbol")
    p.add_argument("--equity", type=float, default=100_000)
    p.set_defaults(func=cmd_risk)

    p = sub.add_parser("portfolio", help="Allocation + rebalance plan")
    p.add_argument("symbols", help="Comma-separated symbols, e.g. AAPL,MSFT,GOOG")
    p.add_argument("--value", type=float, default=100_000)
    p.add_argument("--method", default="equal", choices=["equal", "risk_parity"])
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("health", help="Suite health check")
    p.set_defaults(func=cmd_health)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    orch = TradingOrchestrator.from_config_file(args.config) if args.config else TradingOrchestrator()
    try:
        orch.initialize()
        return args.func(orch, args)
    finally:
        orch.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
