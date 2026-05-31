"""RunStrategy use case — orchestrates the full signal → risk → order pipeline.

This is the central application-layer orchestrator.  On each closed candle it:
  1. Asks the signal port for candidate signals.
  2. Runs every signal through the risk engine (evaluate_risk use case).
  3. Submits approved orders via the order port.
  4. Emits metrics and notifications.

It depends only on ports (interfaces), never on concrete adapters.
"""

from __future__ import annotations

import logging
from typing import List

from application.ports.input.i_market_data_port import IMarketDataPort
from application.ports.input.i_signal_port import ISignalPort
from application.ports.output.i_metrics_port import IMetricsPort
from application.ports.output.i_notify_port import INotifyPort
from application.ports.output.i_order_port import IOrderPort
from application.use_cases.evaluate_risk import EvaluateRisk
from domain.entities.market_snapshot import MarketSnapshot
from domain.entities.order import Order, OrderSide, OrderType
from domain.entities.signal import Direction, Signal
from domain.portfolio.portfolio_state import PortfolioState
from domain.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class RunStrategy:
    """Application service that runs the end-to-end trading loop for one strategy.

    Args:
        signal_port:      Strategy that generates signals from candles.
        order_port:       Execution venue adapter.
        notify_port:      Notification adapter.
        metrics_port:     Metrics adapter.
        evaluate_risk:    Pre-wired EvaluateRisk use case.
        portfolio:        Shared mutable portfolio state.
        circuit_breaker:  Shared circuit breaker instance.
    """

    def __init__(
        self,
        signal_port: ISignalPort,
        order_port: IOrderPort,
        notify_port: INotifyPort,
        metrics_port: IMetricsPort,
        evaluate_risk: EvaluateRisk,
        portfolio: PortfolioState,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._signal_port = signal_port
        self._order_port = order_port
        self._notify_port = notify_port
        self._metrics_port = metrics_port
        self._evaluate_risk = evaluate_risk
        self._portfolio = portfolio
        self._circuit_breaker = circuit_breaker

    async def on_candle(self, snapshot: MarketSnapshot) -> List[Order]:
        """Entry point called by the market-data loop on every closed candle.

        Returns the list of Orders that were actually submitted this tick.
        """
        if self._circuit_breaker.is_tripped:
            logger.warning(
                "Circuit breaker tripped — skipping candle %s %s",
                snapshot.symbol,
                snapshot.timestamp,
            )
            return []

        signals: List[Signal] = await self._signal_port.on_candle(snapshot)
        if not signals:
            return []

        submitted_orders: List[Order] = []
        for signal in signals:
            await self._metrics_port.record_signal(
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                confidence=signal.confidence,
            )
            await self._notify_port.send_signal_alert(signal)

            verdict = self._evaluate_risk.run(
                signal=signal,
                portfolio=self._portfolio,
            )

            if not verdict.approved:
                logger.info(
                    "Signal rejected for %s: %s",
                    signal.symbol,
                    "; ".join(verdict.reasons),
                )
                continue

            order = self._build_order(signal, verdict.adjusted_qty)
            submitted = await self._order_port.submit_order(order)
            submitted_orders.append(submitted)

            await self._notify_port.send_order_update(submitted)
            logger.info(
                "Order submitted: %s %s qty=%.6f",
                submitted.side.value,
                submitted.symbol,
                submitted.quantity,
            )

        await self._metrics_port.record_portfolio_snapshot(
            nav=self._portfolio.nav,
            cash=self._portfolio.cash,
            unrealized_pnl=self._portfolio.unrealized_pnl,
            open_positions=self._portfolio.open_position_count,
        )
        return submitted_orders

    @staticmethod
    def _build_order(signal: Signal, quantity: float) -> Order:
        """Translate a Signal into an Order domain object."""
        side = OrderSide.BUY if signal.direction is Direction.LONG else OrderSide.SELL
        order_type = (
            OrderType.LIMIT if signal.entry_price is not None else OrderType.MARKET
        )
        return Order(
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=signal.entry_price,
            signal_id=None,
        )