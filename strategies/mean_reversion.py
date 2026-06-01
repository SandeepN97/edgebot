"""MeanReversionStrategy — Bollinger Band oversold/overbought bounce on 4h candles.

Signal conditions (all must be true):
  LONG:  close < BB_lower(20, 2σ)  AND  RSI(14) < 35  (oversold bounce)
  SHORT: close > BB_upper(20, 2σ)  AND  RSI(14) > 65  (overbought fade)

Exit targets:
  Stop loss:   entry ± 2.0 × ATR(14)
  Take profit: Bollinger midline (20-period SMA) — dynamic mean-reversion target

strategy_id = "mean_reversion_4h"

Designed to pair with EmaCrossoverStrategy via StrategyRouter: this strategy
fires only in RANGING regimes (ADX < 20); EMA crossover fires in TRENDING.
"""

from __future__ import annotations

import logging
import math
from collections import deque

from application.ports.input.i_signal_port import ISignalPort
from domain.entities.market_snapshot import MarketSnapshot
from domain.entities.signal import Direction, Market, Signal

logger = logging.getLogger(__name__)

_STRATEGY_ID = "mean_reversion_4h"
_BB_PERIOD = 20
_BB_NSTD = 2.0
_RSI_PERIOD = 14
_RSI_LONG_MAX = 35.0   # oversold threshold — LONG only below this
_RSI_SHORT_MIN = 65.0  # overbought threshold — SHORT only above this
_ATR_PERIOD = 14
_ATR_SL_MULT = 2.0


class MeanReversionStrategy(ISignalPort):
    """BB-bounce mean-reversion strategy.

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
        maxlen = _BB_PERIOD + _RSI_PERIOD + 10
        self._closes: deque[float] = deque(maxlen=maxlen)
        self._highs: deque[float] = deque(maxlen=_ATR_PERIOD + 5)
        self._lows: deque[float] = deque(maxlen=_ATR_PERIOD + 5)
        self._warmed_up: bool = False

    # ------------------------------------------------------------------
    # ISignalPort
    # ------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        return _STRATEGY_ID

    @property
    def required_warmup_bars(self) -> int:
        return _BB_PERIOD + _RSI_PERIOD + 5

    def warm_up(self, history: list[MarketSnapshot]) -> None:
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
        if snapshot.symbol != self._symbol or not snapshot.is_closed:
            return []

        self._push_candle(snapshot)

        if not self._warmed_up or len(self._closes) < self.required_warmup_bars:
            return []

        closes = list(self._closes)
        rsi = _rsi(closes, _RSI_PERIOD)
        atr = _atr(list(self._highs), list(self._lows), closes, _ATR_PERIOD)
        bb_upper, bb_mid, bb_lower = _bollinger(closes, _BB_PERIOD, _BB_NSTD)

        current_close = snapshot.close
        signal: Signal | None = None

        if current_close < bb_lower and rsi < _RSI_LONG_MAX:
            entry = current_close
            sl = entry - atr * _ATR_SL_MULT
            tp = bb_mid
            if sl > 0 and tp > entry:
                reason = (
                    f"Close {entry:.2f} < BB_lower {bb_lower:.2f}; "
                    f"RSI={rsi:.1f} < {_RSI_LONG_MAX}; "
                    f"TP=BB_mid {tp:.2f}"
                )
                signal = self._make_signal(Direction.LONG, entry, sl, tp, rsi, reason)

        elif current_close > bb_upper and rsi > _RSI_SHORT_MIN:
            entry = current_close
            sl = entry + atr * _ATR_SL_MULT
            tp = bb_mid
            if tp > 0 and tp < entry:
                reason = (
                    f"Close {entry:.2f} > BB_upper {bb_upper:.2f}; "
                    f"RSI={rsi:.1f} > {_RSI_SHORT_MIN}; "
                    f"TP=BB_mid {tp:.2f}"
                )
                signal = self._make_signal(Direction.SHORT, entry, sl, tp, rsi, reason)

        return [signal] if signal is not None else []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _push_candle(self, snap: MarketSnapshot) -> None:
        self._closes.append(snap.close)
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
            confidence=_confidence(rsi, direction),
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            strategy_id=_STRATEGY_ID,
            reason=reason,
            timeframe=self._timeframe,
        )


# ------------------------------------------------------------------
# Pure indicator functions
# ------------------------------------------------------------------


def _rsi(closes: list[float], period: int = _RSI_PERIOD) -> float:
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


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = _ATR_PERIOD) -> float:
    if len(highs) < 2:
        return (highs[-1] - lows[-1]) if highs else 0.0
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(highs))
    ]
    return sum(trs[-period:]) / min(len(trs), period)


def _bollinger(
    closes: list[float], period: int = _BB_PERIOD, nstd: float = _BB_NSTD
) -> tuple[float, float, float]:
    """Return (upper, mid, lower) Bollinger Bands."""
    window = closes[-period:]
    mid = sum(window) / len(window)
    variance = sum((c - mid) ** 2 for c in window) / max(len(window) - 1, 1)
    std = math.sqrt(variance)
    return mid + nstd * std, mid, mid - nstd * std


def _confidence(rsi: float, direction: Direction) -> float:
    """Map RSI distance from the entry threshold to [0.5, 0.9]."""
    if direction == Direction.LONG:
        dist = max(0.0, _RSI_LONG_MAX - rsi)   # further oversold = more confident
        normalized = min(dist / _RSI_LONG_MAX, 1.0)
    else:
        dist = max(0.0, rsi - _RSI_SHORT_MIN)
        normalized = min(dist / (100.0 - _RSI_SHORT_MIN), 1.0)
    return 0.5 + 0.4 * normalized