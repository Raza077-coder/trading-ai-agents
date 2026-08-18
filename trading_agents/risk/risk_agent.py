"""Risk Management Agent — position sizing, stop-loss, portfolio risk controls.

Implements the two most widely used sizing frameworks:

* **Fixed-fractional (risk-per-trade)**: risk a fixed % of equity per trade,
  sized via ``risk_amount / (entry - stop)``.
* **ATR-based**: stop distance derived from volatility, then risk-sized.

Also exposes portfolio-level risk checks (concentration, max risk exposure)
and VaR-style loss estimation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from trading_agents.core.base import BaseAgent
from trading_agents.core.config import AgentConfig, RiskConfig
from trading_agents.core.exceptions import RiskLimitError
from trading_agents.core.models import Candle, PositionSize, SignalAction


class RiskManagementAgent(BaseAgent):
    """Compute position sizes and enforce risk limits.

    Example:
        >>> with RiskManagementAgent() as agent:
        ...     pos = agent.size_position(
        ...         symbol="AAPL", entry=180.0, stop=171.0, equity=100_000
        ...     )
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        risk_config: Optional[RiskConfig] = None,
    ) -> None:
        super().__init__("risk_management", config)
        self.risk_config = risk_config or RiskConfig()

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def size_position(
        self,
        symbol: str,
        entry: float,
        stop: float,
        equity: float,
        risk_per_trade_pct: Optional[float] = None,
        action: SignalAction = SignalAction.BUY,
        max_position_pct: Optional[float] = None,
    ) -> PositionSize:
        """Size a long position using fixed-fractional risk.

        Args:
            symbol: Ticker.
            entry: Planned entry price.
            stop: Stop-loss price.
            equity: Current account equity.
            risk_per_trade_pct: Fraction of equity risked (default from config).
            action: BUY (long) or SELL (short).
            max_position_pct: Hard cap on position notional as % of equity.

        Raises:
            RiskLimitError: If risk limits would be breached or inputs invalid.
        """
        self._guard_initialized()
        risk_pct = risk_per_trade_pct or self.risk_config.max_portfolio_risk_pct
        max_pct = max_position_pct or self.risk_config.max_position_pct

        if entry <= 0 or stop <= 0:
            raise RiskLimitError("Entry and stop must be positive")
        if stop >= entry and action == SignalAction.BUY:
            raise RiskLimitError("For a long, stop must be below entry")
        if stop <= entry and action == SignalAction.SELL:
            raise RiskLimitError("For a short, stop must be above entry")

        risk_amount = equity * risk_pct
        if action == SignalAction.BUY:
            stop_distance = entry - stop
            direction = 1.0
        else:
            stop_distance = stop - entry
            direction = -1.0

        if stop_distance <= 0:
            raise RiskLimitError("Stop distance must be positive")

        notional = risk_amount / (stop_distance / entry)  # risk-normalized notional
        # cap by max position % of equity
        max_notional = equity * max_pct
        if notional > max_notional:
            notional = max_notional
            self.logger.info(
                "Position capped by max_position_pct (%.0f%% of equity)", max_pct * 100
            )

        quantity = notional / entry
        take_profit = entry + direction * (stop_distance * self._rr_target())

        return PositionSize(
            symbol=symbol,
            quantity=round(quantity, 4),
            notional=round(notional, 2),
            risk_amount=round(risk_amount, 2),
            stop_loss=round(stop, 2),
            take_profit=round(take_profit, 2),
            risk_reward_ratio=round(self._rr_target(), 2),
            rationale=(
                f"Risk {risk_pct*100:.1f}% of equity (${risk_amount:,.0f}); "
                f"stop {stop_distance/entry*100:.1f}% from entry"
            ),
        )

    def size_position_atr(
        self,
        symbol: str,
        entry: float,
        atr_value: float,
        equity: float,
        risk_per_trade_pct: Optional[float] = None,
        direction: str = "long",
    ) -> PositionSize:
        """Size a position with the stop derived from ATR volatility."""
        mult_stop = self.risk_config.atr_multiplier_stop
        mult_target = self.risk_config.atr_multiplier_target
        if direction == "short":
            stop = entry + mult_stop * atr_value
            action = SignalAction.SELL
        else:
            stop = entry - mult_stop * atr_value
            action = SignalAction.BUY
        pos = self.size_position(
            symbol=symbol,
            entry=entry,
            stop=stop,
            equity=equity,
            risk_per_trade_pct=risk_per_trade_pct,
            action=action,
        )
        return PositionSize(
            symbol=pos.symbol,
            quantity=pos.quantity,
            notional=pos.notional,
            risk_amount=pos.risk_amount,
            stop_loss=pos.stop_loss,
            take_profit=round(entry + (mult_target * atr_value) * (1 if direction == "long" else -1), 2),
            risk_reward_ratio=round(mult_target / mult_stop, 2),
            rationale=(
                f"ATR-based stop: {mult_stop:.1f}×ATR (${mult_stop*atr_value:.2f}); "
                f"target {mult_target:.1f}×ATR"
            ),
        )

    # ------------------------------------------------------------------
    # Portfolio-level risk
    # ------------------------------------------------------------------
    def check_portfolio_risk(
        self,
        positions: Dict[str, PositionSize],
        equity: float,
    ) -> Dict[str, object]:
        """Validate a set of positions against concentration/risk limits.

        Returns a dict with ``ok``, ``issues``, and portfolio stats.
        """
        issues: List[str] = []
        total_risk = sum(p.risk_amount for p in positions.values())
        total_notional = sum(p.notional for p in positions.values())

        if total_notional / equity > 1.0:
            issues.append("Total notional exceeds account equity (leverage > 1)")

        for symbol, pos in positions.items():
            pct = pos.notional / equity
            if pct > self.risk_config.max_position_pct:
                issues.append(
                    f"{symbol}: position {pct*100:.1f}% exceeds "
                    f"max {self.risk_config.max_position_pct*100:.1f}%"
                )
            if pos.risk_amount / equity > self.risk_config.max_portfolio_risk_pct:
                issues.append(
                    f"{symbol}: trade risk {pos.risk_amount/equity*100:.1f}% exceeds "
                    f"max {self.risk_config.max_portfolio_risk_pct*100:.1f}%"
                )

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "total_risk_amount": round(total_risk, 2),
            "total_notional": round(total_notional, 2),
            "total_risk_pct": round(total_risk / equity * 100, 2) if equity else 0.0,
            "gross_exposure_pct": round(total_notional / equity * 100, 2) if equity else 0.0,
        }

    def estimate_var(
        self,
        candles: List[Candle],
        notional: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> Dict[str, float]:
        """Historical VaR estimate for a position.

        Returns dict with ``var_amount`` and ``var_pct``.
        """
        from trading_agents.analysis.indicators import last_valid

        closes = [c.close for c in candles]
        if len(closes) < 30:
            raise RiskLimitError("Not enough history for VaR (need ≥30 bars)")
        returns = [
            (closes[i] / closes[i - 1] - 1.0)
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if not returns:
            raise RiskLimitError("No valid returns for VaR")
        # horizon scaling (sqrt rule)
        scale = (horizon_days ** 0.5)
        sorted_ret = sorted(returns)
        idx = min(len(sorted_ret) - 1, int((1 - confidence) * len(sorted_ret)))
        var_pct = abs(sorted_ret[idx]) * scale
        return {
            "var_amount": round(notional * var_pct, 2),
            "var_pct": round(var_pct * 100, 3),
        }

    def suggested_stop(self, candles: List[Candle], direction: str = "long") -> float:
        """Volatility-based stop price using ATR."""
        from trading_agents.analysis.indicators import atr, last_valid

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        atr_series = atr(highs, lows, closes, 14)
        atr_now = last_valid(atr_series)
        if atr_now is None:
            raise RiskLimitError("Could not compute ATR for stop")
        price = closes[-1]
        mult = self.risk_config.atr_multiplier_stop
        return round(price - mult * atr_now, 2) if direction == "long" else round(price + mult * atr_now, 2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rr_target(self) -> float:
        """Default risk-reward target from config (target/stop multiples)."""
        return self.risk_config.atr_multiplier_target / self.risk_config.atr_multiplier_stop
