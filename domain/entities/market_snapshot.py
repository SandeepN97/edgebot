"""MarketSnapshot entity — a single OHLCV candle enriched with spread and volume data.

This is the primary data transfer object that market-data adapters produce and
strategies consume.  It is a pure domain value object with no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class MarketSnapshot:
    """OHLCV candle plus microstructure data for one symbol/timeframe pair.

    Attributes:
        symbol:        Trading pair, e.g. ``"BTC/USDT"``.
        timeframe:     Candle period string, e.g. ``"4h"``, ``"1m"``.
        timestamp:     UTC open time of this candle.
        open:          Opening price.
        high:          Highest price in the candle.
        low:           Lowest price in the candle.
        close:         Closing price.
        volume:        Base-asset volume traded in the candle.
        quote_volume:  Quote-asset volume (price × qty) in the candle.
        bid:           Current best bid (real-time, None for historical).
        ask:           Current best ask (real-time, None for historical).
        trades:        Number of individual trades in the candle.
        is_closed:     True when the candle has closed (not a partial candle).
    """

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    trades: int = 0
    is_closed: bool = True

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")

    @property
    def spread(self) -> Optional[float]:
        """Absolute bid–ask spread; None when tick data is unavailable."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Spread as a fraction of mid-price."""
        if self.spread is None:
            return None
        mid = (self.bid + self.ask) / 2  # type: ignore[operator]
        return self.spread / mid if mid else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid-market price when both sides of the book are available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def typical_price(self) -> float:
        """(H + L + C) / 3 — common indicator input."""
        return (self.high + self.low + self.close) / 3

    @property
    def candle_range(self) -> float:
        """Full candle range (high − low)."""
        return self.high - self.low

    @property
    def body_size(self) -> float:
        """Absolute candle body size (|close − open|)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open