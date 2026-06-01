"""EdgeBot — paper-trading loop for cross-sectional rotation strategy.

Streams daily candles for all 10 universe coins via Binance WebSocket,
detects weekly rebalance points, executes paper fills, logs to SQLite,
and sends Telegram alerts.  Zero real orders are placed.

Required environment variables:
    BINANCE_API_KEY        Read-only Binance API key (WS auth)
    BINANCE_API_SECRET     Corresponding secret
    TELEGRAM_BOT_TOKEN     Telegram Bot API token
    TELEGRAM_CHAT_ID       Target chat/group for rebalance alerts
    DB_PATH                SQLite path (default: data/edgebot.db)
    INITIAL_CASH           Starting paper balance USDT (default: 100)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections import defaultdict
from datetime import date, datetime

from domain.strategy.rotation_strategy import (
    REBALANCE_EVERY,
    compute_decision,
)
from infrastructure.adapters.execution.paper_trade_adapter import PaperTradeAdapter
from infrastructure.adapters.market_data.binance_ws_adapter import (
    ROTATION_SYMBOLS,
    BinanceWsAdapter,
)
from infrastructure.adapters.notify.telegram_adapter import TelegramAdapter
from infrastructure.observability.trade_journal import RebalanceEntry, TradeJournal

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# History length kept per symbol (more than enough for LOOKBACK=90)
_HISTORY_MAXLEN = 150
# Minimum symbols with data before triggering a rebalance
_MIN_SYMBOLS = 8


def _build_adapters() -> tuple[BinanceWsAdapter, PaperTradeAdapter, TelegramAdapter, TradeJournal]:
    market_data = BinanceWsAdapter(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        exchange_id=os.getenv("BINANCE_EXCHANGE_ID", "binance"),
    )
    paper = PaperTradeAdapter(
        slippage_map={
            "BTC/USDT": 0.001,
            "ETH/USDT": 0.001,
        },
        illiquid_slippage=0.003,
    )
    telegram = TelegramAdapter(
        token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        default_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )
    journal = TradeJournal(db_path=os.getenv("DB_PATH", "data/edgebot.db"))
    return market_data, paper, telegram, journal


async def _warm_up_history(
    market_data: BinanceWsAdapter,
    symbols: list[str],
) -> dict[str, list[float]]:
    """Fetch 120 days of history per symbol via REST for momentum warm-up."""
    price_history: dict[str, list[float]] = {}
    logger.info("Warming up price history for %d symbols...", len(symbols))
    for sym in symbols:
        try:
            snaps = await market_data.fetch_history(sym, "1d", limit=_HISTORY_MAXLEN)
            price_history[sym] = [s.close for s in snaps]
            logger.info("  %s: %d bars", sym, len(price_history[sym]))
        except Exception as exc:
            logger.warning("History fetch failed for %s: %s — excluded from warm-up", sym, exc)
    return price_history


async def _do_rebalance(
    rebalance_date: date,
    price_history: dict[str, list[float]],
    current_prices: dict[str, float],
    holdings: dict[str, float],
    cash: float,
    prev_nav: float,
    paper: PaperTradeAdapter,
    journal: TradeJournal,
    telegram: TelegramAdapter,
) -> tuple[dict[str, float], float]:
    """Execute one rotation rebalance. Returns (new_holdings, new_cash)."""
    decision = compute_decision(price_history)
    basket_before = list(holdings.keys())

    if decision.in_cash:
        # Cash filter: liquidate all holdings
        target_prices = {s: current_prices.get(s, 0.0) for s in holdings}
        new_holdings, new_cash, slippage_paid = paper.fill_rotation_rebalance(
            holdings, cash, {}, target_prices
        )
        nav = new_cash + sum(
            new_holdings.get(s, 0.0) * current_prices.get(s, 0.0)
            for s in new_holdings
        )
        entry = RebalanceEntry(
            rebalance_date=rebalance_date,
            basket_before=basket_before,
            basket_after=[],
            momentum_scores=decision.scores,
            portfolio_value=nav,
            realized_pnl=nav - prev_nav,
            slippage_paid=slippage_paid,
            in_cash=True,
        )
        await journal.record_rebalance(entry)
        await telegram.send_cash_filter_alert(
            rebalance_date=rebalance_date,
            portfolio_value=nav,
            scores=decision.scores,
        )
        logger.info("[%s] CASH FILTER — NAV $%.2f", rebalance_date, nav)
        return new_holdings, new_cash

    # Normal rebalance
    target_prices = {s: current_prices.get(s, 0.0) for s in set(list(holdings) + list(decision.target))}
    new_holdings, new_cash, slippage_paid = paper.fill_rotation_rebalance(
        holdings, cash, decision.target, target_prices
    )
    nav = new_cash + sum(
        new_holdings.get(s, 0.0) * current_prices.get(s, 0.0)
        for s in new_holdings
    )
    week_pnl = nav - prev_nav

    entry = RebalanceEntry(
        rebalance_date=rebalance_date,
        basket_before=basket_before,
        basket_after=list(decision.target.keys()),
        momentum_scores=decision.scores,
        portfolio_value=nav,
        realized_pnl=week_pnl,
        slippage_paid=slippage_paid,
        in_cash=False,
    )
    await journal.record_rebalance(entry)
    await telegram.send_rebalance_alert(
        rebalance_date=rebalance_date,
        basket_before=basket_before,
        basket_after=list(decision.target.keys()),
        scores=decision.scores,
        portfolio_value=nav,
        week_pnl=week_pnl,
    )

    basket_str = ", ".join(s.split("/")[0] for s in decision.target)
    total_slip = sum(slippage_paid.values())
    logger.info(
        "[%s] REBALANCE → [%s]  NAV $%.2f  PnL %+.2f  slippage $%.4f",
        rebalance_date, basket_str, nav, week_pnl, total_slip,
    )
    return new_holdings, new_cash


async def run() -> None:
    """Main paper-trading loop."""
    market_data, paper, telegram, journal = _build_adapters()

    stop_event = asyncio.Event()

    def _handle_signal(*_) -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    logger.info("EdgeBot rotation paper-trader starting up")

    await journal.connect()
    await market_data.connect()

    # Load or initialise portfolio
    holdings, cash, days_since_rebalance = await journal.load_portfolio()
    initial_cash = float(os.getenv("INITIAL_CASH", "100"))
    if not holdings and cash == 0.0:
        cash = initial_cash
        logger.info("First run — initialising with $%.2f cash", cash)

    # Warm up price history
    price_history = await _warm_up_history(market_data, ROTATION_SYMBOLS)
    current_prices: dict[str, float] = {
        sym: prices[-1] for sym, prices in price_history.items() if prices
    }

    # Check if a rebalance is overdue from a previous run or first boot
    if days_since_rebalance >= REBALANCE_EVERY:
        nav = cash + sum(holdings.get(s, 0.0) * current_prices.get(s, 0.0) for s in holdings)
        logger.info("Overdue rebalance on startup (day %d)", days_since_rebalance)
        holdings, cash = await _do_rebalance(
            rebalance_date=date.today(),
            price_history=price_history,
            current_prices=current_prices,
            holdings=holdings,
            cash=cash,
            prev_nav=nav,
            paper=paper,
            journal=journal,
            telegram=telegram,
        )
        days_since_rebalance = 0
        await journal.save_portfolio(holdings, cash, days_since_rebalance)

    # Subscribe to daily candles for all 10 coins
    await market_data.subscribe_rotation_universe(timeframe="1d")
    logger.info("Subscribed to 1d candles for %d symbols", len(ROTATION_SYMBOLS))

    # Track which symbols have reported a close for the current day
    pending_day: dict[str, float] = {}    # symbol → close for current date
    current_date: date | None = None

    async def _process_completed_day(day: date, closes: dict[str, float]) -> None:
        nonlocal holdings, cash, days_since_rebalance, current_prices
        # Update price histories and current prices
        for sym, close in closes.items():
            price_history.setdefault(sym, []).append(close)
            if len(price_history[sym]) > _HISTORY_MAXLEN:
                price_history[sym] = price_history[sym][-_HISTORY_MAXLEN:]
            current_prices[sym] = close
            paper.update_price(sym, close)

        days_since_rebalance += 1
        logger.debug("Day %s complete — %d symbols, day count %d/%d",
                     day, len(closes), days_since_rebalance, REBALANCE_EVERY)

        if days_since_rebalance >= REBALANCE_EVERY:
            nav = cash + sum(
                holdings.get(s, 0.0) * current_prices.get(s, 0.0) for s in holdings
            )
            holdings, cash = await _do_rebalance(
                rebalance_date=day,
                price_history=price_history,
                current_prices=current_prices,
                holdings=holdings,
                cash=cash,
                prev_nav=nav,
                paper=paper,
                journal=journal,
                telegram=telegram,
            )
            days_since_rebalance = 0

        await journal.save_portfolio(holdings, cash, days_since_rebalance)

    # Main event loop
    stream_task = asyncio.create_task(_stream_snapshots(
        market_data, pending_day, current_date, _process_completed_day, stop_event
    ))

    await stop_event.wait()
    logger.info("Stopping — persisting portfolio state")
    stream_task.cancel()
    await journal.save_portfolio(holdings, cash, days_since_rebalance)
    await market_data.disconnect()
    await journal.disconnect()
    logger.info("EdgeBot stopped cleanly")


async def _stream_snapshots(
    market_data: BinanceWsAdapter,
    pending_day: dict[str, float],
    current_date: date | None,
    on_day_complete,
    stop_event: asyncio.Event,
) -> None:
    """Pull from the WS snapshot queue and dispatch completed days."""
    async for snapshot in market_data.stream():
        if stop_event.is_set():
            break
        if not snapshot.is_closed:
            continue

        snap_date = snapshot.timestamp.date()

        if current_date is None:
            current_date = snap_date

        if snap_date > current_date:
            # Previous day rolled over — if we have enough data, process it
            if len(pending_day) >= _MIN_SYMBOLS:
                await on_day_complete(current_date, dict(pending_day))
            pending_day.clear()
            current_date = snap_date

        pending_day[snapshot.symbol] = snapshot.close

        # All symbols reported for this date
        if len(pending_day) >= len(ROTATION_SYMBOLS):
            await on_day_complete(current_date, dict(pending_day))
            pending_day.clear()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
