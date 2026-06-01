"""PaperTradeAdapter — simulates order fills without touching a real exchange.

Implements IOrderPort for paper-trading mode.  Fills are simulated at the
current market price with a configurable slippage and fee model.  All state
is in-memory; use the SQLite adapter alongside this one for persistence.

Rotation rebalancing:
  fill_rotation_rebalance() executes a full portfolio rotation in one call,
  applying per-symbol slippage tiers (liquid vs illiquid coins) so paper
  results reflect real execution cost differences across the universe.
"""

from __future__ import annotations

import logging
from uuid import UUID

from application.ports.output.i_order_port import IOrderPort
from domain.entities.order import Order, OrderStatus

import asyncio

logger = logging.getLogger(__name__)

# Slippage tiers: liquid large-caps pay less than smaller-cap alts
_LIQUID_SYMBOLS = {"BTC/USDT", "ETH/USDT"}
_DEFAULT_LIQUID_SLIPPAGE = 0.001      # 0.1%
_DEFAULT_ILLIQUID_SLIPPAGE = 0.003    # 0.3%

_ROTATION_COMMISSION = 0.001   # 0.1% per side (matches backtest)
_MIN_TRADE_USD = 0.50          # skip legs smaller than this


class PaperTradeAdapter(IOrderPort):
    """In-process order execution simulator.

    Args:
        slippage_pct:       Default slippage for single-order fills (IOrderPort path).
        maker_fee_pct:      Maker fee rate.
        taker_fee_pct:      Taker fee rate.
        fill_delay_ms:      Simulated latency before a fill event fires.
        slippage_map:       Per-symbol slippage overrides for rotation rebalancing.
                            Keys that are absent fall back to liquid/illiquid defaults.
        illiquid_slippage:  Slippage rate for symbols not in slippage_map and not
                            in the liquid-symbol set.
    """

    def __init__(
        self,
        slippage_pct: float = 0.0005,
        maker_fee_pct: float = 0.001,
        taker_fee_pct: float = 0.001,
        fill_delay_ms: int = 50,
        slippage_map: dict[str, float] | None = None,
        illiquid_slippage: float = _DEFAULT_ILLIQUID_SLIPPAGE,
    ) -> None:
        self.slippage_pct = slippage_pct
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.fill_delay_ms = fill_delay_ms
        self._slippage_map: dict[str, float] = slippage_map or {}
        self._illiquid_slippage = illiquid_slippage
        self._orders: dict[UUID, Order] = {}
        self._current_prices: dict[str, float] = {}

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
            logger.debug("Limit order queued: %s", order.order_id)

        return order

    async def cancel_order(self, order_id: UUID) -> Order:
        order = self._get_or_raise(order_id)
        if order.is_terminal:
            raise ValueError(f"Cannot cancel terminal order {order_id}")
        order.cancel()
        logger.info("Paper order cancelled: %s", order_id)
        return order

    async def get_order(self, order_id: UUID) -> Order | None:
        return self._orders.get(order_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if not o.is_terminal and (symbol is None or o.symbol == symbol)
        ]

    async def cancel_all_orders(self, symbol: str | None = None) -> list[Order]:
        cancelled = []
        for order in await self.get_open_orders(symbol):
            await self.cancel_order(order.order_id)
            cancelled.append(order)
        return cancelled

    # ------------------------------------------------------------------
    # Rotation rebalancing
    # ------------------------------------------------------------------

    def fill_rotation_rebalance(
        self,
        holdings: dict[str, float],
        cash: float,
        target: dict[str, float],
        prices: dict[str, float],
    ) -> tuple[dict[str, float], float, dict[str, float]]:
        """Simulate a full rotation rebalance fill at close + per-symbol slippage.

        Args:
            holdings: Current holdings {symbol: shares}.
            cash:     Current cash balance (USDT).
            target:   Target fractions {symbol: fraction}.  Empty = move to cash.
            prices:   Current close prices {symbol: price}.

        Returns:
            (new_holdings, new_cash, slippage_paid_by_symbol)
            slippage_paid values are in USDT.
        """
        holdings = dict(holdings)
        slippage_paid: dict[str, float] = {}

        def _nav() -> float:
            return cash + sum(
                holdings.get(s, 0.0) * prices[s]
                for s in holdings
                if s in prices and prices[s] > 0
            )

        # 1. Sell coins exiting the portfolio
        for sym in [s for s in list(holdings.keys()) if s not in target]:
            p = prices.get(sym, 0.0)
            shares = holdings.pop(sym, 0.0)
            if p <= 0 or shares <= 0:
                continue
            slip = self._get_slippage(sym)
            fill_price = p * (1.0 - slip)
            proceeds = shares * fill_price * (1.0 - _ROTATION_COMMISSION)
            cash += proceeds
            slippage_paid[sym] = slippage_paid.get(sym, 0.0) + shares * p * slip

        # 2. Adjust coins remaining in / entering the portfolio
        port_val = _nav()
        for sym, frac in target.items():
            p = prices.get(sym, 0.0)
            if p <= 0:
                continue
            target_usd = port_val * frac
            current_usd = holdings.get(sym, 0.0) * p
            diff = target_usd - current_usd

            if diff > _MIN_TRADE_USD:
                spend = min(diff, cash)
                if spend < _MIN_TRADE_USD:
                    continue
                slip = self._get_slippage(sym)
                fill_price = p * (1.0 + slip)
                shares_bought = spend * (1.0 - _ROTATION_COMMISSION) / fill_price
                holdings[sym] = holdings.get(sym, 0.0) + shares_bought
                cash -= spend
                slippage_paid[sym] = slippage_paid.get(sym, 0.0) + spend * slip / (1.0 + slip)

            elif diff < -_MIN_TRADE_USD:
                shares_to_sell = min(abs(diff) / p, holdings.get(sym, 0.0))
                if shares_to_sell * p < _MIN_TRADE_USD:
                    continue
                slip = self._get_slippage(sym)
                fill_price = p * (1.0 - slip)
                cash += shares_to_sell * fill_price * (1.0 - _ROTATION_COMMISSION)
                holdings[sym] = holdings.get(sym, 0.0) - shares_to_sell
                if holdings[sym] < 1e-12:
                    holdings.pop(sym, None)
                slippage_paid[sym] = slippage_paid.get(sym, 0.0) + shares_to_sell * p * slip

        return holdings, cash, slippage_paid

    # ------------------------------------------------------------------
    # Paper-trade specific helpers
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Called by the market-data loop to push the latest price in."""
        self._current_prices[symbol] = price
        self._try_fill_limit_orders(symbol, price)

    def _get_slippage(self, symbol: str) -> float:
        """Per-symbol slippage: explicit map > liquid default > illiquid default."""
        if symbol in self._slippage_map:
            return self._slippage_map[symbol]
        if symbol in _LIQUID_SYMBOLS:
            return _DEFAULT_LIQUID_SLIPPAGE
        return self._illiquid_slippage

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
            triggered = (order.side.value == "buy" and price <= order.price) or (
                order.side.value == "sell" and price >= order.price
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
