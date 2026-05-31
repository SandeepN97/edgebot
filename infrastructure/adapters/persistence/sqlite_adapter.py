"""SQLiteAdapter — persists trades and orders to a local SQLite database.

Implements the ITradeRepository protocol used by the RecordTrade use case.
Uses aiosqlite for non-blocking I/O.  Schema is created automatically on
first connection.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID

from domain.entities.order import Order
from domain.entities.signal import Direction
from domain.entities.trade import Trade

logger = logging.getLogger(__name__)

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    quantity        REAL NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    entry_fee       REAL NOT NULL DEFAULT 0,
    exit_fee        REAL NOT NULL DEFAULT 0,
    strategy_id     TEXT NOT NULL,
    entry_reason    TEXT NOT NULL,
    exit_reason     TEXT NOT NULL,
    entry_at        TEXT NOT NULL,
    exit_at         TEXT NOT NULL,
    net_pnl         REAL,
    return_pct      REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    quantity        REAL NOT NULL,
    price           REAL,
    status          TEXT NOT NULL,
    exchange_id     TEXT,
    filled_qty      REAL NOT NULL DEFAULT 0,
    avg_fill_price  REAL,
    fee             REAL NOT NULL DEFAULT 0,
    signal_id       TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


class SQLiteAdapter:
    """Async SQLite persistence adapter for trades and orders.

    Args:
        db_path: File path for the SQLite database (e.g. ``"data/edgebot.db"``).
    """

    def __init__(self, db_path: str = "data/edgebot.db") -> None:
        self._db_path = db_path
        self._conn = None

    async def connect(self) -> None:
        """Open the database connection and create tables if absent."""
        try:
            import aiosqlite  # type: ignore[import]
        except ImportError:
            raise ImportError("aiosqlite is required: pip install aiosqlite")

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(_CREATE_TRADES_TABLE)
        await self._conn.execute(_CREATE_ORDERS_TABLE)
        await self._conn.commit()
        logger.info("SQLite connected: %s", self._db_path)

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Trade persistence
    # ------------------------------------------------------------------

    async def save(self, trade: Trade) -> None:
        """Persist a completed Trade. Implements ITradeRepository.save."""
        self._require_connected()
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO trades
            (trade_id, symbol, direction, quantity, entry_price, exit_price,
             entry_fee, exit_fee, strategy_id, entry_reason, exit_reason,
             entry_at, exit_at, net_pnl, return_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(trade.trade_id),
                trade.symbol,
                trade.direction.value,
                trade.quantity,
                trade.entry_price,
                trade.exit_price,
                trade.entry_fee,
                trade.exit_fee,
                trade.strategy_id,
                trade.entry_reason,
                trade.exit_reason,
                trade.entry_at.isoformat(),
                trade.exit_at.isoformat(),
                trade.net_pnl,
                trade.return_pct,
            ),
        )
        await self._conn.commit()
        logger.debug("Trade saved: %s", trade.trade_id)

    async def load_trades(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
        limit: int = 500,
    ) -> list[Trade]:
        """Load historical trades from the database."""
        self._require_connected()
        clauses = []
        params = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if strategy_id:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cursor = await self._conn.execute(
            f"SELECT * FROM trades {where} ORDER BY exit_at DESC LIMIT ?", params
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(row) for row in rows]

    # ------------------------------------------------------------------
    # Order persistence
    # ------------------------------------------------------------------

    async def save_order(self, order: Order) -> None:
        self._require_connected()
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO orders
            (order_id, symbol, side, order_type, quantity, price, status,
             exchange_id, filled_qty, avg_fill_price, fee, signal_id,
             created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(order.order_id),
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.quantity,
                order.price,
                order.status.value,
                order.exchange_id,
                order.filled_qty,
                order.avg_fill_price,
                order.fee,
                str(order.signal_id) if order.signal_id else None,
                order.created_at.isoformat(),
                order.updated_at.isoformat(),
            ),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_trade(row) -> Trade:
        return Trade(
            trade_id=UUID(row["trade_id"]),
            symbol=row["symbol"],
            direction=Direction(row["direction"]),
            quantity=row["quantity"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            entry_fee=row["entry_fee"],
            exit_fee=row["exit_fee"],
            strategy_id=row["strategy_id"],
            entry_reason=row["entry_reason"],
            exit_reason=row["exit_reason"],
            entry_at=datetime.fromisoformat(row["entry_at"]),
            exit_at=datetime.fromisoformat(row["exit_at"]),
        )

    def _require_connected(self) -> None:
        if self._conn is None:
            raise RuntimeError("Call connect() before using SQLiteAdapter")
