"""PaperTradeAdapter — simulates order fills without touching a real exchange.

Implements IOrderPort for paper-trading mode.  Fills are simulated at the
current market price with a configurable slippage and fee model.  All state
is in-memory; use the SQLite adapter alongside this one for persistence.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from application.ports.output.i_order_port import IOrderPort
from domain.entities.order import Order, OrderStatus

logger = logging.getLogger(__name__)


class PaperTradeAdapter(IOrderPort):
    """In-process order execution simulator.

    Args:
        slippage_pct:   Fraction of price added as adverse slippage on fills.
        maker_fee_pct:  Maker fee rate (limit orders).
        taker_fee_pct:  Taker fee rate (market orders).
        fill_delay_ms:  Simulated latency before a fill event fires.
    """

    def __init__(
        self,
        slippage_pct: float = 0.0005,
        maker_fee_pct: float = 0.001,
        taker_fee_pct: float = 0.001,
        fill_delay_ms: int = 50,
    ) -> None:
        self.slippage_pct = slippage_pct
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.fill_delay_ms = fill_delay_ms
        self._orders: Dict[UUID, Order] = {}
        self._current_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # IOrderPort implementation
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Simulate immediate fill for market orders; queue limit orders."""
        order.exchange_id = f"paper-{order.order_id}"
        order.status = OrderStatus.SUBMITTED
        self._orders[order.order_id] = order

        if order.order_type.value == "market":
            await asyncio.sleep(self.fill_delay_ms / 1000)
            await self._fill_at_market(order)
        else:
            # Limit orders are stored and filled on next price_update() call
            logger.debug("Limit order queued: %s", order.order_id)

        return order

    async def cancel_order(self, order_id: UUID) -> Order:
        order = self._get_or_raise(order_id)
        if order.is_terminal:
            raise ValueError(f"Cannot cancel terminal order {order_id}")
        order.cancel()
        logger.info("Paper order cancelled: %s", order_id)
        return order

    async def get_order(self, order_id: UUID) -> Optional[Order]:
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return [
            o for o in self._orders.values()
            if not o.is_terminal and (symbol is None or o.symbol == symbol)
        ]

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        cancelled = []
        for order in await self.get_open_orders(symbol):
            await self.cancel_order(order.order_id)
            cancelled.append(order)
        return cancelled

    # ------------------------------------------------------------------
    # Paper-trade specific helpers
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Called by the market-data loop to push the latest price in."""
        self._current_prices[symbol] = price
        self._try_fill_limit_orders(symbol, price)

    async def _fill_at_market(self, order: Order) -> None:
        price = self._current_prices.get(order.symbol)
        if price is None:
            logger.warning("No price available for %s — rejecting paper order", order.symbol)
            order.reject()
            return
        fill_price = self._apply_slippage(price, order.side.value)
        fee = fill_price * order.quantity * self.taker_fee_pct
        order.fill(qty=order.quantity, price=fill_price, fee=fee)
        logger.info(
            "Paper fill: %s %s qty=%.6f @%.4f fee=%.4f",
            order.side.value,
            order.symbol,
            order.quantity,
            fill_price,
            fee,
        )

    def _try_fill_limit_orders(self, symbol: str, price: float) -> None:
        for order in list(self._orders.values()):
            if order.symbol != symbol or order.is_terminal:
                continue
            if order.price is None:
                continue
            triggered = (
                (order.side.value == "buy" and price <= order.price)
                or (order.side.value == "sell" and price >= order.price)
            )
            if triggered:
                fee = order.price * order.quantity * self.maker_fee_pct
                order.fill(qty=order.quantity, price=order.price, fee=fee)
                logger.info(
                    "Paper limit fill: %s %s @%.4f",
                    order.side.value,
                    order.symbol,
                    order.price,
                )

    def _apply_slippage(self, price: float, side: str) -> float:
        """Adverse slippage: buy fills above price, sell fills below."""
        factor = 1 + self.slippage_pct if side == "buy" else 1 - self.slippage_pct
        return price * factor

    def _get_or_raise(self, order_id: UUID) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found in paper adapter")
        return order