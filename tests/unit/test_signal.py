"""Unit tests for the Signal domain entity."""

from __future__ import annotations

import pytest
from datetime import datetime

from domain.entities.signal import Direction, Market, Signal


def _valid_signal(**overrides) -> Signal:
    defaults = dict(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        market=Market.SPOT,
        confidence=0.75,
        entry_price=30_000.0,
        stop_loss=29_000.0,
        take_profit=32_000.0,
        strategy_id="ema_crossover",
        reason="EMA9 crossed EMA21 with volume confirmation",
    )
    defaults.update(overrides)
    return Signal(**defaults)


class TestSignalCreation:
    def test_creates_valid_signal(self) -> None:
        s = _valid_signal()
        assert s.symbol == "BTC/USDT"
        assert s.direction is Direction.LONG
        assert s.market is Market.SPOT
        assert s.confidence == 0.75

    def test_frozen_dataclass(self) -> None:
        s = _valid_signal()
        with pytest.raises(Exception):  # FrozenInstanceError
            s.symbol = "ETH/USDT"  # type: ignore[misc]

    def test_default_timestamp_is_utc(self) -> None:
        s = _valid_signal()
        assert isinstance(s.timestamp, datetime)

    def test_default_timeframe_is_4h(self) -> None:
        s = _valid_signal()
        assert s.timeframe == "4h"


class TestSignalValidation:
    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _valid_signal(confidence=1.1)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _valid_signal(confidence=-0.1)

    def test_zero_confidence_is_valid(self) -> None:
        s = _valid_signal(confidence=0.0)
        assert s.confidence == 0.0

    def test_confidence_exactly_one_is_valid(self) -> None:
        s = _valid_signal(confidence=1.0)
        assert s.confidence == 1.0

    def test_negative_stop_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss"):
            _valid_signal(stop_loss=-1.0)

    def test_zero_stop_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss"):
            _valid_signal(stop_loss=0.0)

    def test_negative_take_profit_raises(self) -> None:
        with pytest.raises(ValueError, match="take_profit"):
            _valid_signal(take_profit=-1.0)


class TestRiskRewardRatio:
    def test_rr_calculated_correctly_long(self) -> None:
        # entry=100, SL=95 (risk=5), TP=110 (reward=10) → R:R=2.0
        s = _valid_signal(entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        assert s.risk_reward_ratio == pytest.approx(2.0)

    def test_rr_calculated_correctly_short(self) -> None:
        # entry=100, SL=105 (risk=5), TP=90 (reward=10) → R:R=2.0
        s = _valid_signal(
            direction=Direction.SHORT,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
        )
        assert s.risk_reward_ratio == pytest.approx(2.0)

    def test_rr_is_none_without_entry_price(self) -> None:
        s = _valid_signal(entry_price=None)
        assert s.risk_reward_ratio is None

    def test_rr_is_none_when_sl_equals_entry(self) -> None:
        s = _valid_signal(entry_price=100.0, stop_loss=100.0, take_profit=110.0)
        assert s.risk_reward_ratio is None


class TestIsEntryIsExit:
    def test_long_is_entry(self) -> None:
        assert _valid_signal(direction=Direction.LONG).is_entry

    def test_short_is_entry(self) -> None:
        assert _valid_signal(direction=Direction.SHORT, stop_loss=31_000.0, take_profit=28_000.0).is_entry

    def test_flat_is_exit(self) -> None:
        s = Signal(
            symbol="BTC/USDT",
            direction=Direction.FLAT,
            market=Market.SPOT,
            confidence=0.5,
            stop_loss=1.0,
            take_profit=1.0,
            strategy_id="test",
            reason="close position",
        )
        assert s.is_exit
        assert not s.is_entry

    def test_long_is_not_exit(self) -> None:
        assert not _valid_signal(direction=Direction.LONG).is_exit


class TestDirectionEnum:
    def test_direction_values(self) -> None:
        assert Direction.LONG.value == "long"
        assert Direction.SHORT.value == "short"
        assert Direction.FLAT.value == "flat"


class TestMarketEnum:
    def test_market_values(self) -> None:
        assert Market.SPOT.value == "spot"
        assert Market.FUTURES.value == "futures"
        assert Market.MARGIN.value == "margin"


class TestSignalMetadata:
    def test_default_metadata_is_empty_dict(self) -> None:
        s = _valid_signal()
        assert s.metadata == {}

    def test_custom_metadata_stored(self) -> None:
        s = _valid_signal(metadata={"ml_score": 0.91, "regime": "trending"})
        assert s.metadata["ml_score"] == 0.91
        assert s.metadata["regime"] == "trending"