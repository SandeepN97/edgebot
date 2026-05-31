"""Integration tests for PaperTradeAdapter — simulates fills without real exchange."""

from __future__ import annotations

import asyncio
import pytest
from uuid import uuid4

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from infrastructure.adapters.execution.paper_trade_adapter import PaperTradeAdapter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def adapter() -> PaperTradeAdapter:
    return PaperTradeAdapter(
        slippage_pct=0.001,
        maker_fee_pct=0.001,
        taker_fee_pct=0.001,
        fill_delay_ms=0,
    )


def _market_order(symbol: str = "BTC/USDT", side: OrderSide = OrderSide.BUY, qty: float = 0.1) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
    )


def _limit_order(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    qty: float = 0.1,
    price: float = 30_000.0,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty,
        price=price,
    )


# ------------------------------------------------------------------
# Market order tests
# ------------------------------------------------------------------

class TestMarketOrders:
    @pytest.mark.asyncio
    async def test_market_buy_fills_immediately(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order(side=OrderSide.BUY)
        result = await adapter.submit_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_market_buy_applies_positive_slippage(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order(side=OrderSide.BUY)
        await adapter.submit_order(order)
        assert order.avg_fill_price > 30_000.0  # buy fills above price

    @pytest.mark.asyncio
    async def test_market_sell_applies_negative_slippage(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order(side=OrderSide.SELL)
        await adapter.submit_order(order)
        assert order.avg_fill_price < 30_000.0  # sell fills below price

    @pytest.mark.asyncio
    async def test_market_order_charges_taker_fee(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order(qty=1.0)
        await adapter.submit_order(order)
        assert order.fee > 0

    @pytest.mark.asyncio
    async def test_market_order_rejected_without_price(self, adapter: PaperTradeAdapter) -> None:
        # No price set for symbol — should reject
        order = _market_order(symbol="UNKNOWN/USDT")
        result = await adapter.submit_order(order)
        assert result.status == OrderStatus.REJECTED


# ------------------------------------------------------------------
# Limit order tests
# ------------------------------------------------------------------

class TestLimitOrders:
    @pytest.mark.asyncio
    async def test_limit_buy_fills_when_price_drops(self, adapter: PaperTradeAdapter) -> None:
        order = _limit_order(side=OrderSide.BUY, price=30_000.0)
        await adapter.submit_order(order)
        assert order.status == OrderStatus.SUBMITTED

        # Price moves to limit level
        adapter.update_price("BTC/USDT", 29_999.0)
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(30_000.0)

    @pytest.mark.asyncio
    async def test_limit_buy_does_not_fill_above_price(self, adapter: PaperTradeAdapter) -> None:
        order = _limit_order(side=OrderSide.BUY, price=30_000.0)
        await adapter.submit_order(order)
        adapter.update_price("BTC/USDT", 31_000.0)  # price above limit — no fill
        assert order.status != OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_limit_sell_fills_when_price_rises(self, adapter: PaperTradeAdapter) -> None:
        order = _limit_order(side=OrderSide.SELL, price=31_000.0)
        await adapter.submit_order(order)
        adapter.update_price("BTC/USDT", 31_001.0)
        assert order.status == OrderStatus.FILLED


# ------------------------------------------------------------------
# Order management tests
# ------------------------------------------------------------------

class TestOrderManagement:
    @pytest.mark.asyncio
    async def test_cancel_open_limit_order(self, adapter: PaperTradeAdapter) -> None:
        order = _limit_order(price=28_000.0)
        await adapter.submit_order(order)
        cancelled = await adapter.cancel_order(order.order_id)
        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cannot_cancel_filled_order(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order()
        await adapter.submit_order(order)
        assert order.status == OrderStatus.FILLED
        with pytest.raises(ValueError):
            await adapter.cancel_order(order.order_id)

    @pytest.mark.asyncio
    async def test_get_order_returns_correct_order(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order()
        await adapter.submit_order(order)
        retrieved = await adapter.get_order(order.order_id)
        assert retrieved is not None
        assert retrieved.order_id == order.order_id

    @pytest.mark.asyncio
    async def test_get_order_returns_none_for_unknown(self, adapter: PaperTradeAdapter) -> None:
        result = await adapter.get_order(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_open_orders_excludes_filled(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        market_order = _market_order()
        limit_order = _limit_order(price=25_000.0)
        await adapter.submit_order(market_order)
        await adapter.submit_order(limit_order)
        open_orders = await adapter.get_open_orders()
        assert market_order not in open_orders
        assert limit_order in open_orders

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, adapter: PaperTradeAdapter) -> None:
        orders = [_limit_order(price=25_000.0 - i * 100) for i in range(3)]
        for o in orders:
            await adapter.submit_order(o)
        cancelled = await adapter.cancel_all_orders("BTC/USDT")
        assert len(cancelled) == 3
        open_after = await adapter.get_open_orders("BTC/USDT")
        assert len(open_after) == 0

    @pytest.mark.asyncio
    async def test_get_open_orders_filtered_by_symbol(self, adapter: PaperTradeAdapter) -> None:
        btc_order = _limit_order(symbol="BTC/USDT", price=25_000.0)
        eth_order = _limit_order(symbol="ETH/USDT", price=1_800.0)
        await adapter.submit_order(btc_order)
        await adapter.submit_order(eth_order)
        btc_open = await adapter.get_open_orders("BTC/USDT")
        assert btc_order in btc_open
        assert eth_order not in btc_open


# ------------------------------------------------------------------
# Exchange ID assignment
# ------------------------------------------------------------------

class TestExchangeId:
    @pytest.mark.asyncio
    async def test_submitted_order_gets_paper_exchange_id(self, adapter: PaperTradeAdapter) -> None:
        adapter.update_price("BTC/USDT", 30_000.0)
        order = _market_order()
        await adapter.submit_order(order)
        assert order.exchange_id is not None
        assert order.exchange_id.startswith("paper-")