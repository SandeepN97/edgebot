"""EmaCrossoverStrategy — EMA9 × EMA21 crossover on 4h candles.

Signal conditions (all must be true):
  • EMA9 crosses above EMA21  → LONG signal
  • EMA9 crosses below EMA21 → SHORT signal
  • RSI(14) is in [40, 70] (avoids entering in extreme overbought/oversold)
  • Volume of the signal bar > 1.2× the 20-bar average volume

A reason string is always attached to every emitted Signal.

Implements ISignalPort so the application layer needs no Freqtrade dependency.
The strategy can be adapted to Freqtrade's populate_entry_trend() interface by
calling on_candle() from within populate_*.
"""

from __future__ import annotations

import logging
from collections import deque

from application.ports.input.i_signal_port import ISignalPort
from domain.entities.market_snapshot import MarketSnapshot
from domain.entities.signal import Direction, Market, Signal

logger = logging.getLogger(__name__)

_STRATEGY_ID = "ema_crossover_4h"
_EMA_FAST = 9
_EMA_SLOW = 21
_RSI_PERIOD = 14
_RSI_LOW = 40.0
_RSI_HIGH = 70.0
_VOLUME_MA_PERIOD = 20
_VOLUME_MULTIPLIER = 1.2
_ATR_PERIOD = 14
_ATR_SL_MULTIPLIER = 1.5  # stop-loss = entry ± ATR × 1.5
_ATR_TP_MULTIPLIER = 3.0  # take-profit = entry ± ATR × 3.0


class EmaCrossoverStrategy(ISignalPort):
    """EMA9/EMA21 crossover strategy with RSI and volume filters.

    Args:
        symbol:    Trading pair this instance watches (e.g. ``"BTC/USDT"``).
        market:    Market type for emitted signals (default SPOT).
        timeframe: Expected candle timeframe (default ``"4h"``).
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        market: Market = Market.SPOT,
        timeframe: str = "4h",
    ) -> None:
        self._symbol = symbol
        self._market = market
        self._timeframe = timeframe
        self._closes: deque[float] = deque(maxlen=max(_EMA_SLOW, _RSI_PERIOD) + 50)
        self._volumes: deque[float] = deque(maxlen=_VOLUME_MA_PERIOD + 5)
        self._highs: deque[float] = deque(maxlen=_ATR_PERIOD + 5)
        self._lows: deque[float] = deque(maxlen=_ATR_PERIOD + 5)
        self._prev_ema_fast: float | None = None
        self._prev_ema_slow: float | None = None
        self._warmed_up: bool = False

    # ------------------------------------------------------------------
    # ISignalPort implementation
    # ------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        return _STRATEGY_ID

    @property
    def required_warmup_bars(self) -> int:
        return _EMA_SLOW + _RSI_PERIOD + _VOLUME_MA_PERIOD

    def warm_up(self, history: list[MarketSnapshot]) -> None:
        """Pre-load historical candles to seed all indicators."""
        for snap in history:
            self._push_candle(snap)
        self._warmed_up = True
        logger.info(
            "%s warmed up with %d bars for %s",
            _STRATEGY_ID,
            len(history),
            self._symbol,
        )

    async def on_candle(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Evaluate a closed candle and return 0 or 1 signals."""
        if snapshot.symbol != self._symbol or not snapshot.is_closed:
            return []

        self._push_candle(snapshot)

        if not self._warmed_up or len(self._closes) < self.required_warmup_bars:
            return []

        closes = list(self._closes)
        ema_fast = _ema(closes, _EMA_FAST)
        ema_slow = _ema(closes, _EMA_SLOW)
        rsi = _rsi(closes, _RSI_PERIOD)
        atr = _atr(list(self._highs), list(self._lows), closes, _ATR_PERIOD)
        avg_volume = sum(list(self._volumes)[-_VOLUME_MA_PERIOD:]) / _VOLUME_MA_PERIOD
        current_volume = snapshot.volume

        signal = None

        if self._prev_ema_fast is not None and self._prev_ema_slow is not None:
            bull_cross = self._prev_ema_fast <= self._prev_ema_slow and ema_fast > ema_slow
            bear_cross = self._prev_ema_fast >= self._prev_ema_slow and ema_fast < ema_slow
            volume_ok = current_volume > avg_volume * _VOLUME_MULTIPLIER
            rsi_ok = _RSI_LOW <= rsi <= _RSI_HIGH

            if bull_cross and volume_ok and rsi_ok:
                entry = snapshot.close
                sl = entry - atr * _ATR_SL_MULTIPLIER
                tp = entry + atr * _ATR_TP_MULTIPLIER
                reason = (
                    f"EMA{_EMA_FAST} crossed above EMA{_EMA_SLOW}; "
                    f"RSI={rsi:.1f} in [{_RSI_LOW},{_RSI_HIGH}]; "
                    f"vol={current_volume:.0f} > {_VOLUME_MULTIPLIER}×avg"
                )
                signal = self._make_signal(Direction.LONG, entry, sl, tp, rsi, reason)

            elif bear_cross and volume_ok and rsi_ok:
                entry = snapshot.close
                sl = entry + atr * _ATR_SL_MULTIPLIER
                tp = entry - atr * _ATR_TP_MULTIPLIER
                reason = (
                    f"EMA{_EMA_FAST} crossed below EMA{_EMA_SLOW}; "
                    f"RSI={rsi:.1f} in [{_RSI_LOW},{_RSI_HIGH}]; "
                    f"vol={current_volume:.0f} > {_VOLUME_MULTIPLIER}×avg"
                )
                signal = self._make_signal(Direction.SHORT, entry, sl, tp, rsi, reason)

        self._prev_ema_fast = ema_fast
        self._prev_ema_slow = ema_slow

        return [signal] if signal is not None else []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _push_candle(self, snap: MarketSnapshot) -> None:
        self._closes.append(snap.close)
        self._volumes.append(snap.volume)
        self._highs.append(snap.high)
        self._lows.append(snap.low)

    def _make_signal(
        self,
        direction: Direction,
        entry: float,
        sl: float,
        tp: float,
        rsi: float,
        reason: str,
    ) -> Signal:
        return Signal(
            symbol=self._symbol,
            direction=direction,
            market=self._market,
            confidence=self._confidence(rsi),
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            strategy_id=_STRATEGY_ID,
            reason=reason,
            timeframe=self._timeframe,
        )

    @staticmethod
    def _confidence(rsi: float) -> float:
        """Map RSI distance from extremes to a [0.5, 0.9] confidence score."""
        centre_distance = abs(rsi - 55.0)
        normalized = max(0.0, 1.0 - centre_distance / 30.0)
        return 0.5 + 0.4 * normalized


# ------------------------------------------------------------------
# Pure indicator functions (no class state)
# ------------------------------------------------------------------


def _ema(closes: list[float], period: int) -> float:
    """Exponential moving average of the last `period` closes."""
    if len(closes) < period:
        return sum(closes) / len(closes)
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int = 14) -> float:
    """Wilder RSI from a list of closing prices."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [abs(d) for d in deltas if d < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0.0
    avg_loss = sum(losses[-period:]) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Average True Range (Wilder smoothing)."""
    if len(highs) < 2:
        return (highs[-1] - lows[-1]) if highs else 0.0
    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / min(len(true_ranges), period)
