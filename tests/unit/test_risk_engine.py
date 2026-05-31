"""Unit tests for RiskEngine — verifies every hardcoded risk rule in isolation."""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import uuid4

from domain.entities.signal import Signal, Direction, Market
from domain.entities.position import Position
from domain.risk.risk_engine import (
    RiskEngine,
    MAX_POSITION_SIZE_PCT,
    DAILY_LOSS_LIMIT_PCT,
    MAX_OPEN_POSITIONS,
    MIN_RISK_REWARD_RATIO,
    FEE_BUFFER_PCT,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine()


def _make_signal(
    entry: float = 100.0,
    sl: float = 95.0,
    tp: float = 110.0,
    direction: Direction = Direction.LONG,
    symbol: str = "BTC/USDT",
    confidence: float = 0.8,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        market=Market.SPOT,
        confidence=confidence,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        strategy_id="test",
        reason="unit test signal",
    )


def _make_position(symbol: str = "BTC/USDT", direction: Direction = Direction.LONG) -> Position:
    return Position(
        symbol=symbol,
        direction=direction,
        quantity=0.1,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        strategy_id="test",
    )


# ------------------------------------------------------------------
# Happy-path tests
# ------------------------------------------------------------------

class TestRiskEngineApproved:
    def test_valid_signal_is_approved(self, engine: RiskEngine) -> None:
        signal = _make_signal()
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[])
        assert verdict.approved
        assert verdict.reasons == []
        assert verdict.adjusted_qty is not None
        assert verdict.adjusted_qty > 0

    def test_position_size_respects_max_pct(self, engine: RiskEngine) -> None:
        nav = 10_000.0
        signal = _make_signal(entry=100.0)
        verdict = engine.evaluate(signal, nav=nav, daily_loss=0, open_positions=[])
        effective_pct = MAX_POSITION_SIZE_PCT - FEE_BUFFER_PCT
        expected_qty = (nav * effective_pct) / 100.0
        assert abs(verdict.adjusted_qty - expected_qty) < 1e-9

    def test_two_open_positions_still_approved(self, engine: RiskEngine) -> None:
        positions = [
            _make_position("ETH/USDT", Direction.LONG),
            _make_position("SOL/USDT", Direction.LONG),
        ]
        signal = _make_signal(symbol="BTC/USDT")
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=positions)
        assert verdict.approved


# ------------------------------------------------------------------
# Daily loss limit tests
# ------------------------------------------------------------------

class TestDailyLossLimit:
    def test_below_daily_loss_limit_passes(self, engine: RiskEngine) -> None:
        nav = 10_000.0
        daily_loss = nav * DAILY_LOSS_LIMIT_PCT * 0.99  # just under limit
        verdict = engine.evaluate(_make_signal(), nav=nav, daily_loss=daily_loss, open_positions=[])
        assert verdict.approved

    def test_at_daily_loss_limit_is_rejected(self, engine: RiskEngine) -> None:
        nav = 10_000.0
        daily_loss = nav * DAILY_LOSS_LIMIT_PCT  # exactly at limit
        verdict = engine.evaluate(_make_signal(), nav=nav, daily_loss=daily_loss, open_positions=[])
        assert not verdict.approved
        assert any("Daily loss" in r for r in verdict.reasons)

    def test_above_daily_loss_limit_is_rejected(self, engine: RiskEngine) -> None:
        nav = 10_000.0
        daily_loss = nav * 0.10  # 10% loss, well above 5% limit
        verdict = engine.evaluate(_make_signal(), nav=nav, daily_loss=daily_loss, open_positions=[])
        assert not verdict.approved

    def test_zero_nav_is_rejected(self, engine: RiskEngine) -> None:
        verdict = engine.evaluate(_make_signal(), nav=0, daily_loss=0, open_positions=[])
        assert not verdict.approved


# ------------------------------------------------------------------
# Max open positions tests
# ------------------------------------------------------------------

class TestMaxOpenPositions:
    def test_max_positions_exactly_at_limit_rejected(self, engine: RiskEngine) -> None:
        positions = [
            _make_position(f"SYM{i}/USDT") for i in range(MAX_OPEN_POSITIONS)
        ]
        verdict = engine.evaluate(_make_signal(), nav=10_000, daily_loss=0, open_positions=positions)
        assert not verdict.approved
        assert any("Max open positions" in r for r in verdict.reasons)

    def test_one_below_max_positions_approved(self, engine: RiskEngine) -> None:
        positions = [
            _make_position(f"SYM{i}/USDT") for i in range(MAX_OPEN_POSITIONS - 1)
        ]
        verdict = engine.evaluate(_make_signal(), nav=10_000, daily_loss=0, open_positions=positions)
        assert verdict.approved


# ------------------------------------------------------------------
# Risk-reward ratio tests
# ------------------------------------------------------------------

class TestRiskRewardRatio:
    def test_rr_exactly_at_minimum_approved(self, engine: RiskEngine) -> None:
        # R:R = 2.0: entry=100, SL=95 (risk=5), TP=110 (reward=10)
        signal = _make_signal(entry=100.0, sl=95.0, tp=110.0)
        assert signal.risk_reward_ratio == pytest.approx(2.0)
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[])
        assert verdict.approved

    def test_rr_below_minimum_rejected(self, engine: RiskEngine) -> None:
        # R:R = 1.0: entry=100, SL=95 (risk=5), TP=105 (reward=5)
        signal = _make_signal(entry=100.0, sl=95.0, tp=105.0)
        assert signal.risk_reward_ratio == pytest.approx(1.0)
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[])
        assert not verdict.approved
        assert any("R:R" in r for r in verdict.reasons)

    def test_no_entry_price_cannot_compute_rr(self, engine: RiskEngine) -> None:
        signal = Signal(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            market=Market.SPOT,
            confidence=0.8,
            entry_price=None,
            stop_loss=95.0,
            take_profit=110.0,
            strategy_id="test",
            reason="no entry price",
        )
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[])
        assert not verdict.approved


# ------------------------------------------------------------------
# Duplicate position guard
# ------------------------------------------------------------------

class TestDuplicatePositionGuard:
    def test_duplicate_symbol_direction_rejected(self, engine: RiskEngine) -> None:
        existing = _make_position("BTC/USDT", Direction.LONG)
        signal = _make_signal(symbol="BTC/USDT", direction=Direction.LONG)
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[existing])
        assert not verdict.approved
        assert any("Duplicate" in r for r in verdict.reasons)

    def test_same_symbol_opposite_direction_approved(self, engine: RiskEngine) -> None:
        existing = _make_position("BTC/USDT", Direction.SHORT)
        signal = _make_signal(symbol="BTC/USDT", direction=Direction.LONG)
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[existing])
        # Will be approved on duplicate check (may fail other checks)
        assert not any("Duplicate" in r for r in verdict.reasons)


# ------------------------------------------------------------------
# Custom risk limits
# ------------------------------------------------------------------

class TestCustomLimits:
    def test_custom_max_positions(self) -> None:
        engine = RiskEngine(max_open_positions=1)
        positions = [_make_position("ETH/USDT")]
        verdict = engine.evaluate(_make_signal(), nav=10_000, daily_loss=0, open_positions=positions)
        assert not verdict.approved

    def test_custom_rr_ratio(self) -> None:
        engine = RiskEngine(min_risk_reward_ratio=3.0)
        # R:R = 2.0 — passes default but fails custom 3.0
        signal = _make_signal(entry=100.0, sl=95.0, tp=110.0)
        verdict = engine.evaluate(signal, nav=10_000, daily_loss=0, open_positions=[])
        assert not verdict.approved

    def test_multiple_violations_all_reported(self, engine: RiskEngine) -> None:
        nav = 10_000.0
        # Trigger: daily loss + max positions + bad R:R
        positions = [_make_position(f"SYM{i}/USDT") for i in range(MAX_OPEN_POSITIONS)]
        bad_signal = _make_signal(entry=100.0, sl=99.0, tp=101.0)  # R:R=1
        daily_loss = nav * DAILY_LOSS_LIMIT_PCT
        verdict = engine.evaluate(bad_signal, nav=nav, daily_loss=daily_loss, open_positions=positions)
        assert not verdict.approved
        assert len(verdict.reasons) >= 2