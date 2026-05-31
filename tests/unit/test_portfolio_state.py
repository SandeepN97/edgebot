"""Unit tests for PortfolioState — long and short round-trips with NAV assertions."""

from __future__ import annotations

from uuid import uuid4

import pytest

from domain.entities.position import Position
from domain.entities.signal import Direction
from domain.portfolio.portfolio_state import PortfolioState

INITIAL_CASH = 10_000.0


def _long(symbol: str = "BTC/USDT", entry: float = 100.0, qty: float = 1.0) -> Position:
    return Position(
        symbol=symbol,
        direction=Direction.LONG,
        quantity=qty,
        entry_price=entry,
        stop_loss=entry * 0.95,
        take_profit=entry * 1.10,
        strategy_id="test",
    )


def _short(symbol: str = "BTC/USDT", entry: float = 100.0, qty: float = 1.0) -> Position:
    return Position(
        symbol=symbol,
        direction=Direction.SHORT,
        quantity=qty,
        entry_price=entry,
        stop_loss=entry * 1.05,
        take_profit=entry * 0.90,
        strategy_id="test",
    )


# ------------------------------------------------------------------
# NAV at entry (before any price move)
# ------------------------------------------------------------------


class TestNavAtEntry:
    def test_nav_equals_initial_cash_before_any_position(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        assert port.nav == pytest.approx(INITIAL_CASH)

    def test_long_nav_unchanged_at_entry_price(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _long(entry=100.0, qty=1.0)
        port.add_position(pos)
        # cash = 9900, nav_contribution = 100 + 0 = 100, nav = 10000
        assert port.cash == pytest.approx(9_900.0)
        assert port.nav == pytest.approx(INITIAL_CASH)

    def test_short_nav_unchanged_at_entry_price(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _short(entry=100.0, qty=1.0)
        port.add_position(pos)
        # cash = 9900, nav_contribution = 100 + 0 = 100, nav = 10000
        assert port.cash == pytest.approx(9_900.0)
        assert port.nav == pytest.approx(INITIAL_CASH)


# ------------------------------------------------------------------
# Mark-to-market NAV during open position
# ------------------------------------------------------------------


class TestNavMarkToMarket:
    def test_long_nav_increases_when_price_rises(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _long(entry=100.0, qty=1.0)
        port.add_position(pos)
        port.update_prices({"BTC/USDT": 120.0})
        # nav_contribution = 100 + (120-100)*1 = 120; cash = 9900
        assert port.nav == pytest.approx(10_020.0)

    def test_long_nav_decreases_when_price_falls(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _long(entry=100.0, qty=1.0)
        port.add_position(pos)
        port.update_prices({"BTC/USDT": 80.0})
        assert port.nav == pytest.approx(9_980.0)

    def test_short_nav_increases_when_price_falls(self) -> None:
        """Short profits when price drops — NAV should rise."""
        port = PortfolioState(INITIAL_CASH)
        pos = _short(entry=100.0, qty=1.0)
        port.add_position(pos)
        port.update_prices({"BTC/USDT": 80.0})
        # unrealized_pnl = (100-80)*1 = 20; nav_contribution = 100+20 = 120
        assert port.nav == pytest.approx(10_020.0)

    def test_short_nav_decreases_when_price_rises(self) -> None:
        """Short loses when price rises — NAV should fall."""
        port = PortfolioState(INITIAL_CASH)
        pos = _short(entry=100.0, qty=1.0)
        port.add_position(pos)
        port.update_prices({"BTC/USDT": 120.0})
        # unrealized_pnl = (100-120)*1 = -20; nav_contribution = 100-20 = 80
        assert port.nav == pytest.approx(9_980.0)


# ------------------------------------------------------------------
# LONG round-trip
# ------------------------------------------------------------------


class TestLongRoundTrip:
    def test_long_winning_trade_nav_equals_initial_plus_pnl(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _long(entry=100.0, qty=1.0)
        port.add_position(pos)
        pnl = port.close_position(pos.position_id, exit_price=120.0)

        assert pnl == pytest.approx(20.0)
        assert port.cash == pytest.approx(INITIAL_CASH + 20.0)
        assert port.nav == pytest.approx(INITIAL_CASH + 20.0)
        assert port.open_position_count == 0

    def test_long_losing_trade_nav_equals_initial_minus_loss(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _long(entry=100.0, qty=1.0)
        port.add_position(pos)
        pnl = port.close_position(pos.position_id, exit_price=85.0)

        assert pnl == pytest.approx(-15.0)
        assert port.cash == pytest.approx(INITIAL_CASH - 15.0)
        assert port.nav == pytest.approx(INITIAL_CASH - 15.0)

    def test_long_breakeven_trade_nav_unchanged(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _long(entry=100.0, qty=1.0)
        port.add_position(pos)
        pnl = port.close_position(pos.position_id, exit_price=100.0)

        assert pnl == pytest.approx(0.0)
        assert port.nav == pytest.approx(INITIAL_CASH)


# ------------------------------------------------------------------
# SHORT round-trip
# ------------------------------------------------------------------


class TestShortRoundTrip:
    def test_short_winning_trade_nav_equals_initial_plus_pnl(self) -> None:
        """Short wins when price drops; NAV should be initial + profit."""
        port = PortfolioState(INITIAL_CASH)
        pos = _short(entry=100.0, qty=1.0)
        port.add_position(pos)
        pnl = port.close_position(pos.position_id, exit_price=80.0)

        assert pnl == pytest.approx(20.0)
        assert port.cash == pytest.approx(INITIAL_CASH + 20.0)
        assert port.nav == pytest.approx(INITIAL_CASH + 20.0)
        assert port.open_position_count == 0

    def test_short_losing_trade_nav_equals_initial_minus_loss(self) -> None:
        """Short loses when price rises; NAV should be initial − loss."""
        port = PortfolioState(INITIAL_CASH)
        pos = _short(entry=100.0, qty=1.0)
        port.add_position(pos)
        pnl = port.close_position(pos.position_id, exit_price=115.0)

        assert pnl == pytest.approx(-15.0)
        assert port.cash == pytest.approx(INITIAL_CASH - 15.0)
        assert port.nav == pytest.approx(INITIAL_CASH - 15.0)

    def test_short_breakeven_trade_nav_unchanged(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        pos = _short(entry=100.0, qty=1.0)
        port.add_position(pos)
        pnl = port.close_position(pos.position_id, exit_price=100.0)

        assert pnl == pytest.approx(0.0)
        assert port.nav == pytest.approx(INITIAL_CASH)


# ------------------------------------------------------------------
# Mixed portfolio (long + short simultaneously)
# ------------------------------------------------------------------


class TestMixedPortfolio:
    def test_long_and_short_nav_sums_correctly(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        lp = _long("BTC/USDT", entry=100.0, qty=1.0)
        sp = _short("ETH/USDT", entry=50.0, qty=2.0)
        port.add_position(lp)
        port.add_position(sp)

        # cash = 10000 - 100 - 100 = 9800
        assert port.cash == pytest.approx(9_800.0)

        # BTC up 10 → long nav_contribution = 110
        # ETH up 10 → short nav_contribution = 50*2 + (50-60)*2 = 100 - 20 = 80
        port.update_prices({"BTC/USDT": 110.0, "ETH/USDT": 60.0})
        expected_nav = 9_800.0 + 110.0 + 80.0
        assert port.nav == pytest.approx(expected_nav)


# ------------------------------------------------------------------
# Guard rails
# ------------------------------------------------------------------


class TestGuardRails:
    def test_add_position_raises_on_insufficient_cash(self) -> None:
        port = PortfolioState(50.0)
        pos = _long(entry=100.0, qty=1.0)  # costs 100, only 50 cash
        with pytest.raises(ValueError, match="Insufficient cash"):
            port.add_position(pos)

    def test_close_nonexistent_position_raises(self) -> None:
        port = PortfolioState(INITIAL_CASH)
        with pytest.raises(KeyError):
            port.close_position(uuid4(), exit_price=100.0)

    def test_negative_initial_cash_raises(self) -> None:
        with pytest.raises(ValueError):
            PortfolioState(-1.0)
