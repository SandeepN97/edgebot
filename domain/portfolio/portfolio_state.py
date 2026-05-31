"""PortfolioState — tracks open positions, available cash, and net asset value.

This is a mutable domain aggregate.  It is the single source of truth for the
current state of the portfolio within one process.  The infrastructure layer is
responsible for hydrating it from the database on startup and persisting changes
after each trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from domain.entities.position import Position
from domain.entities.signal import Direction


class PortfolioState:
    """Mutable aggregate tracking cash, open positions, and NAV.

    Args:
        initial_cash:  Starting cash balance in quote currency (e.g. USDT).
    """

    def __init__(self, initial_cash: float) -> None:
        if initial_cash < 0:
            raise ValueError("initial_cash cannot be negative")
        self._cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self._positions: Dict[UUID, Position] = {}
        self._created_at: datetime = datetime.utcnow()
        self._updated_at: datetime = datetime.utcnow()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def open_positions(self) -> List[Position]:
        return list(self._positions.values())

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    @property
    def positions_by_symbol(self) -> Dict[str, List[Position]]:
        result: Dict[str, List[Position]] = {}
        for pos in self._positions.values():
            result.setdefault(pos.symbol, []).append(pos)
        return result

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def open_positions_notional(self) -> float:
        return sum(p.notional_value for p in self._positions.values())

    @property
    def nav(self) -> float:
        """Net asset value = cash + mark-to-market value of all open positions."""
        return self._cash + self.open_positions_notional

    @property
    def total_return_pct(self) -> float:
        """Portfolio return since inception."""
        return (self.nav - self._initial_cash) / self._initial_cash if self._initial_cash else 0.0

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """Register a newly opened position and debit cash."""
        cost = position.entry_price * position.quantity
        if cost > self._cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}"
            )
        self._positions[position.position_id] = position
        self._cash -= cost
        self._touch()

    def close_position(self, position_id: UUID, exit_price: float) -> float:
        """Remove a position and credit cash at exit_price. Returns realised PnL."""
        pos = self._positions.pop(position_id, None)
        if pos is None:
            raise KeyError(f"Position {position_id} not found")
        proceeds = exit_price * pos.quantity
        self._cash += proceeds
        realised_pnl = pos.unrealized_pnl  # already updated to exit_price
        self._touch()
        return realised_pnl

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Refresh mark-to-market prices for all open positions."""
        for pos in self._positions.values():
            if pos.symbol in prices:
                pos.update_price(prices[pos.symbol])
        self._touch()

    def get_position(self, position_id: UUID) -> Optional[Position]:
        return self._positions.get(position_id)

    def has_position(self, symbol: str, direction: Direction) -> bool:
        return any(
            p.symbol == symbol and p.direction == direction
            for p in self._positions.values()
        )

    def credit_cash(self, amount: float) -> None:
        """Add cash (e.g. from a realised gain or deposit)."""
        self._cash += amount
        self._touch()

    def debit_cash(self, amount: float) -> None:
        """Remove cash (e.g. fee payment)."""
        if amount > self._cash:
            raise ValueError("Insufficient cash for debit")
        self._cash -= amount
        self._touch()

    def snapshot(self) -> dict:
        """Return a serialisable summary for logging or dashboards."""
        return {
            "cash": self._cash,
            "nav": self.nav,
            "open_positions": self.open_position_count,
            "unrealized_pnl": self.unrealized_pnl,
            "total_return_pct": self.total_return_pct,
            "updated_at": self._updated_at.isoformat(),
        }

    def _touch(self) -> None:
        self._updated_at = datetime.utcnow()