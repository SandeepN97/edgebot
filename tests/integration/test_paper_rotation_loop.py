"""Integration tests for the paper-trading rotation loop.

Tests cover:
  1. Rotation domain: correct basket from synthetic price data
  2. Per-symbol slippage tiers applied in fill_rotation_rebalance
  3. Portfolio NAV conservation through a rebalance
  4. TradeJournal: save/load round-trip
  5. Telegram alert queued on rebalance (mocked)
"""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from domain.strategy.rotation_strategy import LOOKBACK, compute_decision
from infrastructure.adapters.execution.paper_trade_adapter import (
    PaperTradeAdapter,
    _DEFAULT_ILLIQUID_SLIPPAGE,
    _DEFAULT_LIQUID_SLIPPAGE,
    _ROTATION_COMMISSION,
)
from infrastructure.adapters.notify.telegram_adapter import TelegramAdapter
from infrastructure.observability.trade_journal import RebalanceEntry, TradeJournal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_prices(
    n_days: int,
    start: float,
    daily_return: float,
    noise: float = 0.003,
    seed: int = 42,
) -> list[float]:
    """Generate a price series with a mean daily return plus small Gaussian noise.

    Noise is required so daily_return_std > 0 and momentum scores can be computed.
    """
    rng = np.random.default_rng(seed)
    prices = [start]
    for _ in range(n_days - 1):
        r = daily_return + rng.normal(0.0, noise)
        prices.append(prices[-1] * (1.0 + r))
    return prices


def _make_flat_prices(n_days: int, price: float, noise: float = 0.002, seed: int = 0) -> list[float]:
    """Flat price with tiny noise so std > 0 but mean return ≈ 0."""
    rng = np.random.default_rng(seed)
    prices = [price]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1.0 + rng.normal(0.0, noise)))
    return prices


def _synthetic_price_history() -> dict[str, list[float]]:
    """
    SOL/XRP: strongly trending up — high positive momentum → should be selected
    BTC/ETH/BNB: near-flat — low scores
    Others:  declining — negative momentum

    With LOOKBACK=90 and +1.5%/day mean, SOL and XRP have the highest
    risk-adjusted momentum and should be the top-2.
    """
    n = LOOKBACK + 10   # enough history for a full momentum window
    return {
        "BTC/USDT":  _make_flat_prices(n, 60_000.0, seed=1),
        "ETH/USDT":  _make_flat_prices(n, 3_000.0, seed=2),
        "BNB/USDT":  _make_flat_prices(n, 400.0, seed=3),
        "SOL/USDT":  _make_trending_prices(n, 100.0, +0.015, seed=10),
        "XRP/USDT":  _make_trending_prices(n, 0.5, +0.012, seed=11),
        "ADA/USDT":  _make_trending_prices(n, 0.4, -0.005, seed=20),
        "DOGE/USDT": _make_trending_prices(n, 0.08, -0.003, seed=21),
        "AVAX/USDT": _make_trending_prices(n, 35.0, -0.004, seed=22),
        "LINK/USDT": _make_trending_prices(n, 14.0, -0.006, seed=23),
        "DOT/USDT":  _make_trending_prices(n, 6.0, -0.007, seed=24),
    }


# ---------------------------------------------------------------------------
# 1. Rotation domain — correct basket
# ---------------------------------------------------------------------------

class TestRotationDecision:
    def test_selects_top_two_positive_momentum(self) -> None:
        history = _synthetic_price_history()
        decision = compute_decision(history)
        assert not decision.in_cash
        assert set(decision.target.keys()) == {"SOL/USDT", "XRP/USDT"}

    def test_equal_weight_fractions(self) -> None:
        history = _synthetic_price_history()
        decision = compute_decision(history)
        for frac in decision.target.values():
            assert frac == pytest.approx(0.5, rel=1e-6)

    def test_cash_filter_when_all_negative(self) -> None:
        n = LOOKBACK + 5
        history = {
            sym: _make_trending_prices(n, 100.0, -0.01)
            for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        }
        decision = compute_decision(history)
        assert decision.in_cash
        assert decision.target == {}

    def test_insufficient_history_excludes_symbol(self) -> None:
        history = {
            "SOL/USDT": _make_trending_prices(LOOKBACK - 5, 100.0, +0.01),  # too short
            "XRP/USDT": _make_trending_prices(LOOKBACK + 10, 0.5, +0.015),
        }
        decision = compute_decision(history)
        # SOL has insufficient history so it's excluded — scores dict won't contain it
        assert "SOL/USDT" not in decision.scores
        # XRP has enough data; if score > 0 it should be selected
        if not decision.in_cash:
            assert "XRP/USDT" in decision.target


# ---------------------------------------------------------------------------
# 2. Per-symbol slippage tiers
# ---------------------------------------------------------------------------

class TestPaperRotationSlippage:
    def _adapter(self) -> PaperTradeAdapter:
        return PaperTradeAdapter(
            slippage_map={"BTC/USDT": 0.001, "ETH/USDT": 0.001},
            illiquid_slippage=0.003,
        )

    def test_btc_gets_liquid_slippage(self) -> None:
        adapter = self._adapter()
        assert adapter._get_slippage("BTC/USDT") == pytest.approx(_DEFAULT_LIQUID_SLIPPAGE)

    def test_eth_gets_liquid_slippage(self) -> None:
        adapter = self._adapter()
        assert adapter._get_slippage("ETH/USDT") == pytest.approx(_DEFAULT_LIQUID_SLIPPAGE)

    def test_sol_gets_illiquid_slippage(self) -> None:
        adapter = self._adapter()
        assert adapter._get_slippage("SOL/USDT") == pytest.approx(_DEFAULT_ILLIQUID_SLIPPAGE)

    def test_dot_gets_illiquid_slippage(self) -> None:
        adapter = self._adapter()
        assert adapter._get_slippage("DOT/USDT") == pytest.approx(_DEFAULT_ILLIQUID_SLIPPAGE)

    def test_custom_map_overrides_default(self) -> None:
        adapter = PaperTradeAdapter(slippage_map={"SOL/USDT": 0.002})
        assert adapter._get_slippage("SOL/USDT") == pytest.approx(0.002)

    def test_slippage_applied_to_buys(self) -> None:
        """Buying SOL at 100 should fill above 100 (buy slippage raises price)."""
        adapter = self._adapter()
        prices = {"SOL/USDT": 100.0}
        # Start with cash, buy SOL
        new_holdings, new_cash, slippage = adapter.fill_rotation_rebalance(
            holdings={},
            cash=100.0,
            target={"SOL/USDT": 1.0},
            prices=prices,
        )
        # Slippage cost should be approx 0.3% of the spend
        spend = 100.0  # all cash goes to SOL
        expected_slip = spend * _DEFAULT_ILLIQUID_SLIPPAGE / (1.0 + _DEFAULT_ILLIQUID_SLIPPAGE)
        assert slippage.get("SOL/USDT", 0.0) == pytest.approx(expected_slip, rel=0.05)

    def test_slippage_applied_to_sells(self) -> None:
        """Selling BTC should incur liquid slippage."""
        adapter = self._adapter()
        prices = {"BTC/USDT": 60_000.0}
        new_holdings, new_cash, slippage = adapter.fill_rotation_rebalance(
            holdings={"BTC/USDT": 0.001},
            cash=0.0,
            target={},   # sell everything
            prices=prices,
        )
        gross = 0.001 * 60_000.0
        expected_slip = gross * _DEFAULT_LIQUID_SLIPPAGE
        assert slippage.get("BTC/USDT", 0.0) == pytest.approx(expected_slip, rel=0.05)


# ---------------------------------------------------------------------------
# 3. Portfolio NAV conservation
# ---------------------------------------------------------------------------

class TestPaperRotationPortfolio:
    def _adapter(self) -> PaperTradeAdapter:
        return PaperTradeAdapter(
            slippage_map={"BTC/USDT": 0.001, "ETH/USDT": 0.001},
            illiquid_slippage=0.003,
        )

    def test_nav_reduced_by_costs_after_buy(self) -> None:
        adapter = self._adapter()
        starting_cash = 100.0
        prices = {"SOL/USDT": 100.0, "XRP/USDT": 0.50}

        new_holdings, new_cash, slippage = adapter.fill_rotation_rebalance(
            holdings={},
            cash=starting_cash,
            target={"SOL/USDT": 0.5, "XRP/USDT": 0.5},
            prices=prices,
        )
        nav = new_cash + sum(
            new_holdings.get(s, 0.0) * prices[s] for s in new_holdings
        )
        total_slip = sum(slippage.values())
        # NAV = starting_cash − fees − slippage (fees baked into shares received)
        assert nav < starting_cash
        assert nav == pytest.approx(starting_cash - total_slip, rel=0.01)

    def test_rotation_basket_changes(self) -> None:
        """Start holding BTC, rotate to SOL+XRP."""
        adapter = self._adapter()
        prices = {"BTC/USDT": 60_000.0, "SOL/USDT": 100.0, "XRP/USDT": 0.5}
        initial_holdings = {"BTC/USDT": 0.001}   # $60 in BTC
        initial_cash = 40.0

        new_holdings, new_cash, _ = adapter.fill_rotation_rebalance(
            holdings=initial_holdings,
            cash=initial_cash,
            target={"SOL/USDT": 0.5, "XRP/USDT": 0.5},
            prices=prices,
        )
        assert "BTC/USDT" not in new_holdings
        assert "SOL/USDT" in new_holdings
        assert "XRP/USDT" in new_holdings

    def test_cash_filter_liquidates_all(self) -> None:
        adapter = self._adapter()
        prices = {"BTC/USDT": 60_000.0, "SOL/USDT": 100.0}
        holdings = {"BTC/USDT": 0.001, "SOL/USDT": 0.5}
        cash = 0.0

        new_holdings, new_cash, _ = adapter.fill_rotation_rebalance(
            holdings=holdings,
            cash=cash,
            target={},   # cash filter
            prices=prices,
        )
        assert new_holdings == {}
        assert new_cash > 0

    def test_small_partial_adjustments_skipped(self) -> None:
        """Partial rebalancing adjustments below MIN_TRADE_USD are skipped."""
        adapter = self._adapter()
        # Hold $49.95 SOL targeting $50.00 — diff = $0.05 < MIN_TRADE_USD
        prices = {"SOL/USDT": 100.0, "XRP/USDT": 0.5}
        holdings = {"SOL/USDT": 0.4995}  # worth $49.95
        cash = 50.05  # total NAV = $100

        new_holdings, new_cash, slippage = adapter.fill_rotation_rebalance(
            holdings=holdings,
            cash=cash,
            target={"SOL/USDT": 0.5},   # target $50 — diff only $0.05
            prices=prices,
        )
        # Diff is below MIN_TRADE_USD — SOL holdings should not change materially
        assert new_holdings.get("SOL/USDT", 0.0) == pytest.approx(
            holdings["SOL/USDT"], rel=0.001
        )


# ---------------------------------------------------------------------------
# 4. TradeJournal persistence
# ---------------------------------------------------------------------------

class TestTradeJournal:
    @pytest.fixture
    async def journal(self, tmp_path) -> TradeJournal:
        j = TradeJournal(db_path=str(tmp_path / "test.db"))
        await j.connect()
        yield j
        await j.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_load_portfolio(self, journal: TradeJournal) -> None:
        holdings = {"SOL/USDT": 0.5, "XRP/USDT": 100.0}
        cash = 23.45
        days = 3

        await journal.save_portfolio(holdings, cash, days)
        loaded_holdings, loaded_cash, loaded_days = await journal.load_portfolio()

        assert loaded_holdings == holdings
        assert loaded_cash == pytest.approx(cash)
        assert loaded_days == days

    @pytest.mark.asyncio
    async def test_load_empty_portfolio_returns_defaults(self, journal: TradeJournal) -> None:
        holdings, cash, days = await journal.load_portfolio()
        assert holdings == {}
        assert cash == pytest.approx(0.0)
        assert days == 0

    @pytest.mark.asyncio
    async def test_record_and_retrieve_rebalance(self, journal: TradeJournal) -> None:
        entry = RebalanceEntry(
            rebalance_date=date(2024, 3, 15),
            basket_before=["BTC/USDT"],
            basket_after=["SOL/USDT", "XRP/USDT"],
            momentum_scores={"SOL/USDT": 12.5, "XRP/USDT": 9.3, "BTC/USDT": -1.2},
            portfolio_value=103.45,
            realized_pnl=3.45,
            slippage_paid={"SOL/USDT": 0.015, "XRP/USDT": 0.012},
            in_cash=False,
        )
        await journal.record_rebalance(entry)
        loaded = await journal.last_rebalance()

        assert loaded is not None
        assert loaded.rebalance_date == date(2024, 3, 15)
        assert loaded.basket_before == ["BTC/USDT"]
        assert loaded.basket_after == ["SOL/USDT", "XRP/USDT"]
        assert loaded.portfolio_value == pytest.approx(103.45)
        assert loaded.realized_pnl == pytest.approx(3.45)

    @pytest.mark.asyncio
    async def test_upsert_overwrites_portfolio(self, journal: TradeJournal) -> None:
        await journal.save_portfolio({"BTC/USDT": 0.01}, 50.0, 2)
        await journal.save_portfolio({"SOL/USDT": 1.0}, 25.0, 4)
        holdings, cash, days = await journal.load_portfolio()
        assert "SOL/USDT" in holdings
        assert "BTC/USDT" not in holdings
        assert cash == pytest.approx(25.0)
        assert days == 4


# ---------------------------------------------------------------------------
# 5. Telegram queued on rebalance (mocked)
# ---------------------------------------------------------------------------

class TestTelegramRotationAlerts:
    @pytest.mark.asyncio
    async def test_rebalance_alert_calls_send_message(self) -> None:
        adapter = TelegramAdapter(token="fake", default_chat_id="123")
        adapter.send_message = AsyncMock()

        await adapter.send_rebalance_alert(
            rebalance_date=date(2024, 3, 15),
            basket_before=["BTC/USDT"],
            basket_after=["SOL/USDT", "XRP/USDT"],
            scores={"SOL/USDT": 12.5, "XRP/USDT": 9.3},
            portfolio_value=103.45,
            week_pnl=3.45,
        )

        adapter.send_message.assert_awaited_once()
        call_args = adapter.send_message.call_args[0]
        msg = call_args[0]
        assert "SOL" in msg
        assert "XRP" in msg
        assert "103.45" in msg

    @pytest.mark.asyncio
    async def test_cash_filter_alert_calls_send_message(self) -> None:
        adapter = TelegramAdapter(token="fake", default_chat_id="123")
        adapter.send_message = AsyncMock()

        await adapter.send_cash_filter_alert(
            rebalance_date=date(2024, 3, 15),
            portfolio_value=95.0,
            scores={"BTC/USDT": -2.1, "ETH/USDT": -1.8},
        )

        adapter.send_message.assert_awaited_once()
        call_args = adapter.send_message.call_args[0]
        msg = call_args[0]
        assert "cash" in msg.lower() or "Cash" in msg

    @pytest.mark.asyncio
    async def test_rebalance_shows_entered_and_exited(self) -> None:
        adapter = TelegramAdapter(token="fake", default_chat_id="123")
        adapter.send_message = AsyncMock()

        await adapter.send_rebalance_alert(
            rebalance_date=date(2024, 3, 15),
            basket_before=["BTC/USDT", "ETH/USDT"],
            basket_after=["SOL/USDT", "XRP/USDT"],
            scores={},
            portfolio_value=100.0,
            week_pnl=-2.0,
        )

        msg = adapter.send_message.call_args[0][0]
        # BTC and ETH exited, SOL and XRP entered
        assert "SOL" in msg
        assert "XRP" in msg
        assert "BTC" in msg
        assert "ETH" in msg
