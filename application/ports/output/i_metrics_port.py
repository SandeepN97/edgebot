"""IMetricsPort — output port interface for emitting operational metrics.

Infrastructure adapters (InfluxDB, Prometheus, StatsD) implement this so the
application can record performance data without importing any metrics library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class IMetricsPort(ABC):
    """Driven port: the application pushes metrics through this interface."""

    @abstractmethod
    async def record_trade(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        duration_seconds: float,
        strategy_id: str,
    ) -> None:
        """Emit metrics for a completed trade.

        Args:
            symbol:           Trading pair.
            direction:        ``"long"`` or ``"short"``.
            pnl:              Net PnL of the trade.
            duration_seconds: Time in position.
            strategy_id:      Originating strategy.
        """

    @abstractmethod
    async def record_signal(self, symbol: str, strategy_id: str, confidence: float) -> None:
        """Emit a counter and confidence gauge for each generated signal."""

    @abstractmethod
    async def record_portfolio_snapshot(
        self,
        nav: float,
        cash: float,
        unrealized_pnl: float,
        open_positions: int,
    ) -> None:
        """Push a periodic portfolio snapshot for dashboarding."""

    @abstractmethod
    async def record_order_latency(
        self, symbol: str, latency_ms: float, order_type: str
    ) -> None:
        """Track round-trip order submission latency in milliseconds."""

    @abstractmethod
    async def increment_counter(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment an arbitrary named counter with optional tags."""

    @abstractmethod
    async def set_gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set an arbitrary named gauge to a specific value."""