"""RecordTrade use case — persists a completed trade and updates portfolio state.

Called when the execution adapter confirms that a position has been fully closed.
It creates a Trade entity, updates the PortfolioState, notifies the operator,
pushes metrics, and optionally triggers the CircuitBreaker if the trade was a loss.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol

from application.ports.output.i_metrics_port import IMetricsPort
from application.ports.output.i_notify_port import INotifyPort
from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.trade import Trade
from domain.portfolio.portfolio_state import PortfolioState
from domain.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class ITradeRepository(Protocol):
    """Minimal persistence protocol for completed trades."""

    async def save(self, trade: Trade) -> None:
        """Persist a Trade to the data store."""
        ...


class RecordTrade:
    """Application service that finalises a closed trade.

    Args:
        trade_repo:      Persistence adapter (SQLite, etc.) implementing ITradeRepository.
        portfolio:       Shared mutable portfolio state.
        circuit_breaker: Shared circuit breaker to register losses against.
        notify_port:     Notification adapter for trade summaries.
        metrics_port:    Metrics adapter for trade telemetry.
    """

    def __init__(
        self,
        trade_repo: ITradeRepository,
        portfolio: PortfolioState,
        circuit_breaker: CircuitBreaker,
        notify_port: INotifyPort,
        metrics_port: IMetricsPort,
    ) -> None:
        self._trade_repo = trade_repo
        self._portfolio = portfolio
        self._circuit_breaker = circuit_breaker
        self._notify_port = notify_port
        self._metrics_port = metrics_port

    async def execute(
        self,
        position: Position,
        exit_order: Order,
        exit_reason: str,
        entry_reason: str,
    ) -> Trade:
        """Close position, create Trade record, persist, and emit events.

        Args:
            position:     The Position being closed.
            exit_order:   The filled exit Order (must be in FILLED status).
            exit_reason:  Why the position was closed (e.g. ``"stop_loss"``, ``"take_profit"``).
            entry_reason: The reason from the original Signal.

        Returns:
            The persisted Trade entity.
        """
        exit_price = exit_order.avg_fill_price or exit_order.price or 0.0
        trade = Trade(
            symbol=position.symbol,
            direction=position.direction,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            strategy_id=position.strategy_id,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            entry_at=position.opened_at,
            exit_at=datetime.now(UTC),
            entry_fee=0.0,
            exit_fee=exit_order.fee,
        )

        # Update portfolio: close position, credit/debit cash
        position.update_price(exit_price)
        self._portfolio.close_position(position.position_id, exit_price)

        # Update circuit breaker if this trade was a loss
        if trade.net_pnl < 0:
            was_tripped = self._circuit_breaker.record_loss(
                loss=abs(trade.net_pnl), nav=self._portfolio.nav
            )
            if was_tripped:
                logger.critical(
                    "Circuit breaker tripped after trade %s: %s",
                    trade.trade_id,
                    self._circuit_breaker.trip_reason,
                )
                await self._notify_port.send_circuit_breaker_alert(
                    self._circuit_breaker.trip_reason
                )
        else:
            self._circuit_breaker.record_gain(trade.net_pnl)

        # Persist
        await self._trade_repo.save(trade)

        # Notify and emit metrics
        await self._notify_port.send_trade_summary(trade)
        await self._metrics_port.record_trade(
            symbol=trade.symbol,
            direction=trade.direction.value,
            pnl=trade.net_pnl,
            duration_seconds=trade.duration_seconds,
            strategy_id=trade.strategy_id,
        )

        logger.info(
            "Trade recorded: %s %s net_pnl=%.4f return=%.2f%%",
            trade.symbol,
            trade.direction.value,
            trade.net_pnl,
            trade.return_pct * 100,
        )
        return trade
