"""Signal entity — represents a trading intent produced by a strategy.

In Hexagonal Architecture this is a pure domain object: no I/O, no framework
imports.  Strategies produce Signals; the application layer consumes them and
routes them through the risk engine before any order is placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class Direction(Enum):
    """Directional bias of a trading signal."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # exit / close position


class Market(Enum):
    """Supported market types — drives adapter selection at the infrastructure layer."""

    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"
    FOREX = "forex"


@dataclass(frozen=True)
class Signal:
    """Immutable trading signal produced by a strategy.

    Attributes:
        symbol:       Trading pair or instrument, e.g. ``"BTC/USDT"``.
        direction:    LONG, SHORT, or FLAT (close existing position).
        market:       Market type the signal targets.
        confidence:   Model confidence in [0.0, 1.0]; used by the position sizer.
        entry_price:  Suggested entry price (None ⇒ market order).
        stop_loss:    Hard stop-loss price; mandatory for risk evaluation.
        take_profit:  Target profit price; mandatory for R:R evaluation.
        strategy_id:  Identifier of the originating strategy.
        reason:       Human-readable explanation of why the signal fired.
        timestamp:    UTC time the signal was generated.
        timeframe:    Candle timeframe that produced the signal, e.g. ``"4h"``.
        metadata:     Arbitrary strategy-specific key/value pairs.
    """

    symbol: str
    direction: Direction
    market: Market
    confidence: float
    stop_loss: float
    take_profit: float
    strategy_id: str
    reason: str
    entry_price: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    timeframe: str = "4h"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive")
        if self.take_profit <= 0:
            raise ValueError("take_profit must be positive")

    @property
    def risk_reward_ratio(self) -> float | None:
        """Compute raw R:R from entry → TP / entry → SL (requires entry_price)."""
        if self.entry_price is None:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk == 0:
            return None
        return reward / risk

    @property
    def is_entry(self) -> bool:
        """True when the signal intends to open a new position."""
        return self.direction in (Direction.LONG, Direction.SHORT)

    @property
    def is_exit(self) -> bool:
        """True when the signal intends to close an existing position."""
        return self.direction is Direction.FLAT
