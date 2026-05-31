"""IOrderPort — output port interface for submitting orders to an exchange.

The application layer calls this port; execution adapters (paper trade,
live broker) implement it.  This decoupling allows swapping between paper
trading and live trading without touching any business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from domain.entities.order import Order


class IOrderPort(ABC):
    """Driven port: the application submits orders through this interface."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Send an order to the execution venue and return the updated Order.

        The returned order will have its exchange_id and status updated.

        Args:
            order: The Order to submit (status should be PENDING on entry).

        Returns:
            The same Order with status updated to SUBMITTED or REJECTED.
        """

    @abstractmethod
    async def cancel_order(self, order_id: UUID) -> Order:
        """Request cancellation of an open order.

        Args:
            order_id: The internal UUID of the order to cancel.

        Returns:
            Updated Order with CANCELLED status.
        """

    @abstractmethod
    async def get_order(self, order_id: UUID) -> Order | None:
        """Retrieve the current state of an order by its internal ID.

        Returns None if the order is not found at the execution venue.
        """

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Return all currently open (unfilled) orders, optionally filtered by symbol."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> list[Order]:
        """Cancel all open orders, optionally filtered by symbol.

        Returns the list of cancelled orders.
        """
