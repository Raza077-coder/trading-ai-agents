"""Backtest engine — runs a strategy over historical candles and reports metrics.

Simulates long-only trades with commission + slippage, tracks an equity curve,
and computes the classic performance battery: total/annualized return,
max drawdown, Sharpe & Sortino ratios, win rate, profit factor.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

from trading_agents.backtest.strategies import Strategy, build_strategy
from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig, BacktestConfig
from trading_agents.core.exceptions import BacktestError, InsufficientDataError
from trading_agents.core.models import BacktestResult, Candle, OrderSide, Trade
from trading_agents.core.models import safe_div


class BacktestAgent(BaseAgent):
    """Backtest a strategy on historical candles.

    Example:
        >>> with BacktestAgent() as agent:
        ...     result = agent.run(candles, strategy_name="sma_cross")
        ...     print(result.total_return_pct, result.sharpe_ratio)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ) -> None:
        super().__init__("backtest", config)
        self.bt_config = backtest_config or BacktestConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        candles: List[Candle],
        strategy_name: str,
        symbol: Optional[str] = None,
        strategy_kwargs: Optional[dict] = None,
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """Run a backtest for a named strategy.

        Args:
            candles: OHLCV bars (oldest → newest).
            strategy_name: Strategy key (``sma_cross``, ``rsi_mean_reversion``,
                ``macd``, ``bollinger``).
            symbol: Symbol label for the result.
            strategy_kwargs: Optional params passed to the strategy.
            initial_capital: Override default capital.

        Raises:
            InsufficientDataError: If too few bars.
            BacktestError: On strategy failures.
        """
        self._guard_initialized()
        if not candles or len(candles) < 100:
            raise InsufficientDataError(
                f"Backtest needs ≥100 candles, got {len(candles or [])}"
            )
        symbol = symbol or "UNKNOWN"
        capital = initial_capital or self.bt_config.initial_capital
        try:
            strategy = build_strategy(strategy_name, **(strategy_kwargs or {}))
        except ValueError as exc:
            raise BacktestError(str(exc)) from exc

        self.logger.info(
            "Backtesting %s on %s (%d bars, $%.0f)",
            strategy.name,
            symbol,
            len(candles),
            capital,
        )
        signals = strategy.generate_signals(candles)
        if len(signals) != len(candles):
            raise BacktestError("Strategy returned mismatched signal length")

        return self._simulate(candles, signals, strategy.name, symbol, capital)

    def run_strategy(
        self,
        candles: List[Candle],
        strategy: Strategy,
        symbol: Optional[str] = None,
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """Backtest with a pre-instantiated :class:`Strategy` object."""
        capital = initial_capital or self.bt_config.initial_capital
        signals = strategy.generate_signals(candles)
        return self._simulate(
            candles, signals, strategy.name, symbol or "UNKNOWN", capital
        )

    def compare(
        self,
        candles: List[Candle],
        strategy_names: List[str],
        symbol: Optional[str] = None,
    ) -> List[BacktestResult]:
        """Run several strategies and return results sorted by Sharpe ratio."""
        results = [
            self.run(candles, name, symbol=symbol) for name in strategy_names
        ]
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _simulate(
        self,
        candles: List[Candle],
        signals: List[int],
        strategy_name: str,
        symbol: str,
        capital: float,
    ) -> BacktestResult:
        equity = capital
        cash = capital
        position_qty = 0.0
        entry_price = 0.0
        entry_time: Optional[datetime] = None
        trades: List[Trade] = []
        equity_curve: List[float] = []
        open_trade: Optional[Trade] = None

        commission = self.bt_config.commission_pct
        slippage = self.bt_config.slippage_pct

        for i, candle in enumerate(candles):
            signal = signals[i]
            price = candle.close
            # --- entry ---
            if signal == 1 and position_qty == 0:
                fill = price * (1 + slippage)
                qty = (cash * 0.98) / fill  # keep a small cash buffer
                cost = qty * fill
                fee = cost * commission
                if cost + fee <= cash:
                    cash -= cost + fee
                    position_qty = qty
                    entry_price = fill
                    entry_time = candle.timestamp
                    open_trade = Trade(
                        symbol=symbol,
                        side=OrderSide.LONG,
                        entry_time=entry_time,
                        entry_price=round(entry_price, 4),
                        quantity=round(qty, 4),
                    )
            # --- exit ---
            elif signal == -1 and position_qty > 0 and open_trade is not None:
                fill = price * (1 - slippage)
                proceeds = position_qty * fill
                fee = proceeds * commission
                cash += proceeds - fee
                pnl = proceeds - fee - (open_trade.entry_price * open_trade.quantity)
                pnl_pct = safe_div(pnl, open_trade.entry_price * open_trade.quantity) * 100
                open_trade = Trade(
                    symbol=open_trade.symbol,
                    side=open_trade.side,
                    entry_time=open_trade.entry_time,
                    entry_price=open_trade.entry_price,
                    exit_time=candle.timestamp,
                    exit_price=round(fill, 4),
                    quantity=open_trade.quantity,
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    exit_reason="signal",
                )
                trades.append(open_trade)
                position_qty = 0.0
                open_trade = None

            equity = cash + position_qty * price
            equity_curve.append(equity)

        # force-close any open position at last price
        if position_qty > 0 and open_trade is not None:
            last = candles[-1]
            fill = last.close * (1 - slippage)
            proceeds = position_qty * fill
            fee = proceeds * commission
            cash += proceeds - fee
            pnl = proceeds - fee - (open_trade.entry_price * open_trade.quantity)
            pnl_pct = safe_div(pnl, open_trade.entry_price * open_trade.quantity) * 100
            trades.append(
                Trade(
                    symbol=open_trade.symbol,
                    side=open_trade.side,
                    entry_time=open_trade.entry_time,
                    entry_price=open_trade.entry_price,
                    exit_time=last.timestamp,
                    exit_price=round(fill, 4),
                    quantity=open_trade.quantity,
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    exit_reason="end_of_data",
                )
            )
            position_qty = 0.0

        final_equity = cash + position_qty * candles[-1].close
        return self._metrics(
            candles=candles,
            trades=trades,
            equity_curve=equity_curve,
            strategy_name=strategy_name,
            symbol=symbol,
            initial_capital=capital,
            final_equity=final_equity,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _metrics(
        self,
        candles: List[Candle],
        trades: List[Trade],
        equity_curve: List[float],
        strategy_name: str,
        symbol: str,
        initial_capital: float,
        final_equity: float,
    ) -> BacktestResult:
        total_return = safe_div(final_equity - initial_capital, initial_capital) * 100

        # annualization from trading days
        n_days = max(1, len(equity_curve))
        years = n_days / 252.0
        if final_equity > 0 and initial_capital > 0 and years > 0:
            annualized = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized = 0.0

        # drawdown
        peak = -float("inf")
        max_dd = 0.0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (eq - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        max_dd_pct = max_dd * 100

        # returns series for Sharpe/Sortino
        rets = [
            (equity_curve[i] / equity_curve[i - 1] - 1.0)
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] > 0
        ]
        sharpe = self._sharpe(rets)
        sortino = self._sortino(rets)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = safe_div(gross_win, gross_loss)
        win_rate = safe_div(len(wins), len(trades)) * 100 if trades else 0.0
        avg_trade = (
            sum(t.pnl_pct for t in trades) / len(trades) if trades else 0.0
        )

        # benchmark (buy & hold)
        if len(candles) >= 2:
            bench = safe_div(
                candles[-1].close - candles[0].close, candles[0].close
            ) * 100
        else:
            bench = None

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            initial_capital=round(initial_capital, 2),
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annualized, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 3),
            num_trades=len(trades),
            num_wins=len(wins),
            num_losses=len(losses),
            avg_trade_pct=round(avg_trade, 2),
            benchmark_return_pct=round(bench, 2) if bench is not None else None,
            equity_curve=[round(e, 2) for e in equity_curve],
            trades=trades,
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
        )

    @staticmethod
    def _sharpe(rets: List[float], risk_free: float = 0.04) -> float:
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        daily_rf = risk_free / 252.0
        return (mean - daily_rf) / std * math.sqrt(252)

    @staticmethod
    def _sortino(rets: List[float], risk_free: float = 0.04) -> float:
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        downside = [r for r in rets if r < 0]
        if not downside:
            return 0.0
        dd_var = sum(r ** 2 for r in downside) / len(downside)
        dd_std = math.sqrt(dd_var)
        if dd_std == 0:
            return 0.0
        daily_rf = risk_free / 252.0
        return (mean - daily_rf) / dd_std * math.sqrt(252)
