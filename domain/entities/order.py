"""Order entity — represents a request to buy or sell on an exchange.

Orders are created by the application layer after a signal passes risk checks.
They travel outward through the IOrderPort and return as filled/rejected
OrderStatus updates.  This file contains zero infrastructure imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4


class OrderType(Enum):
    """Execution semantics for an order."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(Enum):
    """Lifecycle state of an order."""

    PENDING = "pending"  # created locally, not yet sent
    SUBMITTED = "submitted"  # sent to exchange, awaiting ack
    OPEN = "open"  # acknowledged, resting on the book
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Buy or sell."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Mutable order entity that tracks its own lifecycle.

    Attributes:
        symbol:          Trading pair, e.g. ``"BTC/USDT"``.
        side:            BUY or SELL.
        order_type:      Execution type (market, limit, …).
        quantity:        Base-asset quantity to trade.
        price:           Limit price; None for market orders.
        stop_price:      Trigger price for stop orders.
        status:          Current lifecycle state.
        signal_id:       UUID of the Signal that originated this order.
        exchange_id:     Exchange-assigned order ID once submitted.
        filled_qty:      How much has been executed so far.
        avg_fill_price:  Volume-weighted average fill price.
        fee:             Total fee charged by the exchange.
        created_at:      UTC creation timestamp.
        updated_at:      UTC timestamp of last status change.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    signal_id: UUID | None = None
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: UUID = field(default_factory=uuid4)
    exchange_id: str | None = None
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fee: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def fill(self, qty: float, price: float, fee: float = 0.0) -> None:
        """Record a (partial) fill event."""
        if qty <= 0:
            raise ValueError("Fill quantity must be positive")
        total_cost = (self.avg_fill_price or 0) * self.filled_qty + price * qty
        self.filled_qty += qty
        self.avg_fill_price = total_cost / self.filled_qty
        self.fee += fee
        self.updated_at = datetime.now(UTC)
        if self.filled_qty >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Transition to cancelled state."""
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now(UTC)

    def reject(self) -> None:
        """Transition to rejected state."""
        self.status = OrderStatus.REJECTED
        self.updated_at = datetime.now(UTC)

    @property
    def is_terminal(self) -> bool:
        """True when no further state changes are expected."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.quantity - self.filled_qty)
