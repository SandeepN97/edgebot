"""ISignalPort — input port interface for receiving trading signals.

Strategies implement this port so the application layer can pull signals
without knowing which strategy (EMA crossover, ML model, etc.) is running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from domain.entities.market_snapshot import MarketSnapshot
from domain.entities.signal import Signal


class ISignalPort(ABC):
    """Driving port: a strategy produces Signals from MarketSnapshots."""

    @abstractmethod
    async def on_candle(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Process a new candle and return zero or more signals.

        The application layer calls this on every closed candle.  Returning an
        empty list is valid — it means no trade setup is present.

        Args:
            snapshot: The latest closed OHLCV candle.

        Returns:
            List of Signal objects (may be empty).
        """

    @abstractmethod
    def warm_up(self, history: list[MarketSnapshot]) -> None:
        """Pre-load historical candles to initialise indicators.

        Must be called before the first on_candle() call.  The strategy
        uses this history to compute initial EMA values, etc.

        Args:
            history: List of past snapshots, oldest first.
        """

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy instance."""

    @property
    @abstractmethod
    def required_warmup_bars(self) -> int:
        """Minimum number of historical candles needed before signals are reliable."""
