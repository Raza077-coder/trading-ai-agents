"""Portfolio Manager Agent — allocation, rebalancing, portfolio analytics.

Implements:

* **Equal-weight allocation** — distribute capital evenly across assets.
* **Risk-parity style** — weight by inverse volatility (clamped).
* **Rebalancing** — detect drift beyond a tolerance band and compute target
  trades to restore the target weights.
* **Portfolio analytics** — returns, volatility, Sharpe of the whole book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig
from trading_agents.core.models import Candle, Quote


@dataclass
class Allocation:
    """A target or current allocation for a symbol."""

    symbol: str
    weight: float  # 0.0 … 1.0
    value: float = 0.0
    target_value: float = 0.0
    drift_pct: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "symbol": self.symbol,
            "weight": round(self.weight, 4),
            "value": round(self.value, 2),
            "target_value": round(self.target_value, 2),
            "drift_pct": round(self.drift_pct, 2),
        }


@dataclass
class RebalancePlan:
    """A set of trades to move the portfolio back to target weights."""

    trades: List[Dict[str, object]] = field(default_factory=list)
    total_turnover: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "trades": self.trades,
            "total_turnover": round(self.total_turnover, 2),
            "summary": self.summary,
        }


class PortfolioManagerAgent(BaseAgent):
    """Manage multi-asset portfolios: allocate, rebalance, analyze.

    Example:
        >>> with PortfolioManagerAgent() as agent:
        ...     alloc = agent.equal_weight(["AAPL", "MSFT", "GOOG"], 300_000)
        ...     plan = agent.rebalance(alloc, quotes)
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__("portfolio_manager", config)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def equal_weight(
        self, symbols: List[str], total_value: float
    ) -> Dict[str, Allocation]:
        """Distribute capital equally across the given symbols."""
        self._guard_initialized()
        if not symbols:
            raise ValueError("symbols must not be empty")
        weight = 1.0 / len(symbols)
        per = total_value * weight
        return {
            s: Allocation(
                symbol=s,
                weight=weight,
                value=per,
                target_value=per,
            )
            for s in symbols
        }

    def inverse_volatility(
        self,
        candles_by_symbol: Dict[str, List[Candle]],
        total_value: float,
        lookback: int = 60,
    ) -> Dict[str, Allocation]:
        """Weight each asset inversely to its recent volatility (risk parity)."""
        self._guard_initialized()
        vols: Dict[str, float] = {}
        for symbol, candles in candles_by_symbol.items():
            closes = [c.close for c in candles[-lookback:]]
            if len(closes) < 10:
                vols[symbol] = 1.0
                continue
            rets = [
                closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            vols[symbol] = max(var ** 0.5, 1e-6)
        inv = {s: 1.0 / v for s, v in vols.items()}
        total_inv = sum(inv.values())
        allocs: Dict[str, Allocation] = {}
        for symbol, inv_vol in inv.items():
            weight = inv_vol / total_inv
            allocs[symbol] = Allocation(
                symbol=symbol,
                weight=weight,
                value=total_value * weight,
                target_value=total_value * weight,
            )
        return allocs

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------
    def rebalance(
        self,
        allocations: Dict[str, Allocation],
        quotes: Dict[str, Quote],
        tolerance_pct: float = 5.0,
    ) -> RebalancePlan:
        """Compute trades to rebalance a drifted portfolio.

        Args:
            allocations: Current allocations (values refreshed from quotes).
            quotes: Latest prices per symbol.
            tolerance_pct: Drift threshold (%) that triggers a rebalance trade.

        Returns:
            A :class:`RebalancePlan` with per-symbol trade instructions.
        """
        self._guard_initialized()
        total_value = sum(a.value for a in allocations.values())
        if total_value <= 0:
            raise ValueError("Portfolio value must be positive")

        trades: List[Dict[str, object]] = []
        turnover = 0.0
        for symbol, alloc in allocations.items():
            quote = quotes.get(symbol)
            if quote is None or quote.price <= 0:
                continue
            current_weight = alloc.value / total_value
            target_value = alloc.weight * total_value
            drift = (current_weight - alloc.weight) * 100.0
            alloc.target_value = target_value
            alloc.drift_pct = round(drift, 2)

            if abs(drift) < tolerance_pct:
                continue
            delta_value = target_value - alloc.value
            quantity = delta_value / quote.price
            side = "BUY" if delta_value > 0 else "SELL"
            trades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": round(abs(quantity), 4),
                    "value": round(abs(delta_value), 2),
                    "current_weight_pct": round(current_weight * 100, 2),
                    "target_weight_pct": round(alloc.weight * 100, 2),
                    "drift_pct": round(drift, 2),
                }
            )
            turnover += abs(delta_value)

        plan = RebalancePlan(
            trades=trades,
            total_turnover=turnover,
            summary=(
                f"{len(trades)} rebalance trade(s), "
                f"turnover ${turnover:,.0f} ({turnover/total_value*100:.1f}% of portfolio)"
                if trades
                else "Portfolio within tolerance — no rebalance needed"
            ),
        )
        return plan

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def portfolio_stats(
        self,
        candles_by_symbol: Dict[str, List[Candle]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Estimate portfolio return, volatility and Sharpe from components."""
        self._guard_initialized()
        import math

        symbols = list(candles_by_symbol.keys())
        closes_map: Dict[str, List[float]] = {}
        for sym, candles in candles_by_symbol.items():
            closes_map[sym] = [c.close for c in candles]
        min_len = min(len(v) for v in closes_map.values())
        if min_len < 20:
            raise ValueError("Need ≥20 bars per asset")

        # asset returns aligned to the shortest series
        rets_map: Dict[str, List[float]] = {}
        for sym in symbols:
            closes = closes_map[sym][-min_len:]
            rets_map[sym] = [
                closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]

        n = len(next(iter(rets_map.values())))
        port_returns: List[float] = []
        for i in range(n):
            port_returns.append(
                sum(weights.get(sym, 0.0) * rets_map[sym][i] for sym in symbols)
            )

        mean = sum(port_returns) / len(port_returns)
        var = sum((r - mean) ** 2 for r in port_returns) / (len(port_returns) - 1)
        vol = math.sqrt(var)
        sharpe = (mean - 0.04 / 252.0) / vol * math.sqrt(252) if vol > 0 else 0.0
        cumulative = 1.0
        for r in port_returns:
            cumulative *= 1.0 + r
        return {
            "expected_annual_return_pct": round((cumulative ** (252 / n) - 1) * 100, 2),
            "annualized_volatility_pct": round(vol * math.sqrt(252) * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "num_assets": len(symbols),
        }
