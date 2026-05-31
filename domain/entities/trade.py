"""Trade entity — a completed round-trip (entry + exit) with PnL calculation.

A Trade is created by the record_trade use case once a position is fully closed.
It is immutable and represents the historical record of a finished trade cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from domain.entities.signal import Direction


@dataclass(frozen=True)
class Trade:
    """Immutable record of a completed trade (entry → exit).

    Attributes:
        symbol:           Trading pair.
        direction:        LONG or SHORT (the entry side).
        quantity:         Base-asset quantity traded.
        entry_price:      Volume-weighted average entry fill price.
        exit_price:       Volume-weighted average exit fill price.
        entry_fee:        Fee paid on entry.
        exit_fee:         Fee paid on exit.
        strategy_id:      Strategy that generated the originating signal.
        entry_reason:     Why the trade was entered (from Signal.reason).
        exit_reason:      Why the trade was exited (stop, target, manual, …).
        entry_at:         UTC time position was opened.
        exit_at:          UTC time position was closed.
        trade_id:         Unique identifier.
    """

    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    exit_price: float
    strategy_id: str
    entry_reason: str
    exit_reason: str
    entry_at: datetime
    exit_at: datetime
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    trade_id: UUID = field(default_factory=uuid4)

    @property
    def gross_pnl(self) -> float:
        """PnL before fees."""
        multiplier = 1.0 if self.direction is Direction.LONG else -1.0
        return multiplier * (self.exit_price - self.entry_price) * self.quantity

    @property
    def net_pnl(self) -> float:
        """PnL after all fees."""
        return self.gross_pnl - self.entry_fee - self.exit_fee

    @property
    def total_fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def return_pct(self) -> float:
        """Net return as a fraction of entry cost (0.05 = 5%)."""
        cost = self.entry_price * self.quantity
        return self.net_pnl / cost if cost else 0.0

    @property
    def duration_seconds(self) -> float:
        """Time in position in seconds."""
        return (self.exit_at - self.entry_at).total_seconds()

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def risk_reward_achieved(self) -> float | None:
        """Actual R:R realised — gross_pnl / entry_fee (proxy for risk)."""
        risk = self.entry_price * self.quantity * 0.01  # 1% as surrogate
        return abs(self.gross_pnl) / risk if risk else None
