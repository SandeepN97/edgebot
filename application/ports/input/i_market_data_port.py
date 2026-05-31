"""IMarketDataPort — input port interface for receiving market data.

Strategies and use cases depend on this abstraction.  Infrastructure adapters
(Binance WebSocket, CSV replayer, etc.) implement it so the domain never
imports any exchange library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from domain.entities.market_snapshot import MarketSnapshot


class IMarketDataPort(ABC):
    """Driving port: market-data adapters push snapshots into the application."""

    @abstractmethod
    async def subscribe(self, symbol: str, timeframe: str) -> None:
        """Start streaming candles for the given symbol/timeframe pair.

        Args:
            symbol:    Trading pair, e.g. ``"BTC/USDT"``.
            timeframe: Candle period, e.g. ``"4h"``.
        """

    @abstractmethod
    async def unsubscribe(self, symbol: str, timeframe: str) -> None:
        """Stop streaming for the given symbol/timeframe pair."""

    @abstractmethod
    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        """Async generator that yields MarketSnapshot objects as they arrive."""

    @abstractmethod
    async def fetch_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: Optional[int] = None,
    ) -> List[MarketSnapshot]:
        """Fetch historical OHLCV candles for indicator warm-up.

        Args:
            symbol:    Trading pair.
            timeframe: Candle period.
            limit:     Maximum number of candles to return.
            since:     Unix timestamp in milliseconds; fetch from this point.

        Returns:
            List of MarketSnapshot ordered oldest-first.
        """

    @abstractmethod
    async def get_latest(self, symbol: str, timeframe: str) -> Optional[MarketSnapshot]:
        """Return the most recent closed candle for a symbol/timeframe pair."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the underlying connection (WebSocket / REST session)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly close the underlying connection."""