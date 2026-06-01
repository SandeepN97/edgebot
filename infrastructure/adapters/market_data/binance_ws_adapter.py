"""BinanceWsAdapter — streams live OHLCV candles from Binance via CCXT WebSocket.

Implements IMarketDataPort using the ccxt.pro (ccxt async WebSocket) library.
Each closed candle is converted to a MarketSnapshot and yielded to callers.

Reconnection: up to MAX_RECONNECT_RETRIES consecutive failures per subscription,
with exponential backoff (1s → 2s → 4s → 8s → 16s).  After all retries are
exhausted the subscription is removed and a CRITICAL log is emitted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from application.ports.input.i_market_data_port import IMarketDataPort
from domain.entities.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)

MAX_RECONNECT_RETRIES = 5
_BACKOFF_BASE = 1.0
_BACKOFF_CAP = 30.0

ROTATION_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT",
    "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
]


class BinanceWsAdapter(IMarketDataPort):
    """CCXT-based WebSocket adapter for Binance spot candle streaming.

    Args:
        api_key:     Binance API key (read-only permissions sufficient).
        api_secret:  Binance API secret.
        testnet:     If True, connects to Binance testnet endpoints.
        exchange_id: CCXT exchange identifier (``"binance"`` or ``"binanceusdm"``).
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        exchange_id: str = "binance",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._exchange_id = exchange_id
        self._exchange = None
        self._subscriptions: set[tuple[str, str]] = set()
        self._snapshot_queue: asyncio.Queue[MarketSnapshot] = asyncio.Queue()

    async def connect(self) -> None:
        """Initialise the CCXT exchange instance."""
        try:
            import ccxt.pro as ccxtpro  # type: ignore[import]
        except ImportError:
            raise ImportError("ccxt[pro] is required: pip install ccxt[async]")

        exchange_class = getattr(ccxtpro, self._exchange_id)
        self._exchange = exchange_class(
            {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if self._testnet:
            self._exchange.set_sandbox_mode(True)
        await self._exchange.load_markets()
        logger.info("Connected to %s (testnet=%s)", self._exchange_id, self._testnet)

    async def disconnect(self) -> None:
        """Close WebSocket connections and release resources."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
            logger.info("Disconnected from %s", self._exchange_id)

    async def subscribe(self, symbol: str, timeframe: str) -> None:
        """Register a symbol/timeframe pair and start streaming in background."""
        key = (symbol, timeframe)
        if key in self._subscriptions:
            return
        self._subscriptions.add(key)
        asyncio.create_task(self._stream_candles(symbol, timeframe))
        logger.info("Subscribed to %s %s", symbol, timeframe)

    async def subscribe_rotation_universe(self, timeframe: str = "1d") -> None:
        """Subscribe to all 10 rotation universe symbols at once."""
        for sym in ROTATION_SYMBOLS:
            await self.subscribe(sym, timeframe)

    async def unsubscribe(self, symbol: str, timeframe: str) -> None:
        self._subscriptions.discard((symbol, timeframe))
        logger.info("Unsubscribed from %s %s", symbol, timeframe)

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        """Yield snapshots from the internal queue as they arrive."""
        while True:
            snapshot = await self._snapshot_queue.get()
            yield snapshot

    async def fetch_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: int | None = None,
    ) -> list[MarketSnapshot]:
        """Fetch historical OHLCV bars for indicator warm-up."""
        self._require_connected()
        raw = await self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit, since=since
        )
        return [self._candle_to_snapshot(symbol, timeframe, row, closed=True) for row in raw]

    async def get_latest(self, symbol: str, timeframe: str) -> MarketSnapshot | None:
        """Return the most recent closed candle."""
        history = await self.fetch_history(symbol, timeframe, limit=1)
        return history[-1] if history else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _stream_candles(self, symbol: str, timeframe: str) -> None:
        """Background task: watch candles and push closed ones to the queue.

        Retries up to MAX_RECONNECT_RETRIES times with exponential backoff.
        Removes the subscription after all retries are exhausted.
        """
        self._require_connected()
        prev_ts: int | None = None
        consecutive_errors = 0
        backoff = _BACKOFF_BASE

        while (symbol, timeframe) in self._subscriptions:
            try:
                candles = await self._exchange.watch_ohlcv(symbol, timeframe)
                consecutive_errors = 0
                backoff = _BACKOFF_BASE
                for row in candles:
                    ts = row[0]
                    if ts != prev_ts:
                        snapshot = self._candle_to_snapshot(symbol, timeframe, row, closed=True)
                        await self._snapshot_queue.put(snapshot)
                        prev_ts = ts
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= MAX_RECONNECT_RETRIES:
                    logger.critical(
                        "Stream %s %s failed %d consecutive times — subscription removed. "
                        "Last error: %s",
                        symbol, timeframe, MAX_RECONNECT_RETRIES, exc,
                    )
                    self._subscriptions.discard((symbol, timeframe))
                    return
                logger.warning(
                    "Stream error %s %s (attempt %d/%d): %s — retrying in %.0fs",
                    symbol, timeframe, consecutive_errors, MAX_RECONNECT_RETRIES, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP)

    @staticmethod
    def _candle_to_snapshot(
        symbol: str,
        timeframe: str,
        row: list,
        closed: bool = True,
    ) -> MarketSnapshot:
        ts = datetime.fromtimestamp(row[0] / 1000, tz=UTC).replace(tzinfo=None)
        return MarketSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            is_closed=closed,
        )

    def _require_connected(self) -> None:
        if self._exchange is None:
            raise RuntimeError("Call connect() before using the adapter")
