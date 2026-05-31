"""Position entity — represents an open exposure in a specific symbol.

A Position is created when a fill arrives and destroyed (or reduced) when an
offsetting fill arrives.  It is a domain concept: no exchange or framework
knowledge lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

_UTC = timezone.utc
from typing import Optional
from uuid import UUID, uuid4

from domain.entities.signal import Direction


@dataclass
class Position:
    """Tracks a single open market position.

    Attributes:
        position_id:    Unique identifier.
        symbol:         Trading pair, e.g. ``"BTC/USDT"``.
        direction:      LONG or SHORT.
        quantity:       Open base-asset quantity (always positive).
        entry_price:    Volume-weighted average entry price.
        current_price:  Most recent market price (updated on each tick).
        stop_loss:      Active stop-loss price.
        take_profit:    Active take-profit price.
        strategy_id:    Strategy that originated this position.
        opened_at:      UTC time the position was first entered.
        updated_at:     UTC time the position was last modified.
    """

    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_id: str
    current_price: float = field(init=False)
    position_id: UUID = field(default_factory=uuid4)
    opened_at: datetime = field(default_factory=lambda: datetime.now(_UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(_UTC))

    def __post_init__(self) -> None:
        self.current_price = self.entry_price

    def update_price(self, price: float) -> None:
        """Refresh current_price on each market tick."""
        self.current_price = price
        self.updated_at = datetime.now(_UTC)

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market PnL at the current price."""
        multiplier = 1.0 if self.direction is Direction.LONG else -1.0
        return multiplier * (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL as a fraction of entry cost."""
        cost = self.entry_price * self.quantity
        return self.unrealized_pnl / cost if cost else 0.0

    @property
    def notional_value(self) -> float:
        """Current market value of the position (always positive; direction-agnostic)."""
        return self.current_price * self.quantity

    @property
    def nav_contribution(self) -> float:
        """Direction-aware capital contribution to portfolio NAV.

        Equals the margin posted at entry (entry_price × qty) plus unrealized PnL.
        For longs this equals notional_value; for shorts it moves inversely with
        the market price, correctly reducing NAV when the price rises against a
        short position.

        Formula: entry_price × qty + unrealized_pnl
          LONG  @100, current=120 → 100 + 20  = 120  (same as notional_value)
          SHORT @100, current=80  → 100 + 20  = 120  (short profit)
          SHORT @100, current=120 → 100 + (−20) = 80  (short loss)
        """
        return self.entry_price * self.quantity + self.unrealized_pnl

    @property
    def is_stopped_out(self) -> bool:
        """True when current price has breached the stop-loss level."""
        if self.direction is Direction.LONG:
            return self.current_price <= self.stop_loss
        return self.current_price >= self.stop_loss

    @property
    def is_target_reached(self) -> bool:
        """True when current price has reached the take-profit level."""
        if self.direction is Direction.LONG:
            return self.current_price >= self.take_profit
        return self.current_price <= self.take_profit

    def reduce(self, qty: float) -> None:
        """Partially close the position by qty."""
        if qty > self.quantity:
            raise ValueError(f"Cannot reduce by {qty}; position size is {self.quantity}")
        self.quantity -= qty
        self.updated_at = datetime.now(_UTC)