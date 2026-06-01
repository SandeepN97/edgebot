"""TradeJournal — persists rotation rebalance events and portfolio state to SQLite.

One row per rebalance in rebalance_journal for audit/review.
rotation_portfolio table stores current holdings so the bot survives restarts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DDL_JOURNAL = """
CREATE TABLE IF NOT EXISTS rebalance_journal (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    rebalance_date    TEXT    NOT NULL,
    basket_before     TEXT    NOT NULL,
    basket_after      TEXT    NOT NULL,
    momentum_scores   TEXT    NOT NULL,
    portfolio_value   REAL    NOT NULL,
    realized_pnl      REAL    NOT NULL,
    slippage_paid     TEXT    NOT NULL,
    in_cash           INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_PORTFOLIO = """
CREATE TABLE IF NOT EXISTS rotation_portfolio (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class RebalanceEntry:
    rebalance_date: date
    basket_before: list[str]
    basket_after: list[str]
    momentum_scores: dict[str, float]
    portfolio_value: float
    realized_pnl: float
    slippage_paid: dict[str, float]
    in_cash: bool = False


class TradeJournal:
    """Async SQLite journal for rotation rebalance events and portfolio state.

    Args:
        db_path: File path for the SQLite database.
    """

    def __init__(self, db_path: str = "data/edgebot.db") -> None:
        self._db_path = db_path
        self._conn = None

    async def connect(self) -> None:
        try:
            import aiosqlite  # type: ignore[import]
        except ImportError:
            raise ImportError("aiosqlite is required: pip install aiosqlite")

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(_DDL_JOURNAL)
        await self._conn.execute(_DDL_PORTFOLIO)
        await self._conn.commit()
        logger.info("TradeJournal connected: %s", self._db_path)

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def record_rebalance(self, entry: RebalanceEntry) -> None:
        """Persist one rebalance event to the journal."""
        self._require_connected()
        await self._conn.execute(
            """
            INSERT INTO rebalance_journal
            (rebalance_date, basket_before, basket_after, momentum_scores,
             portfolio_value, realized_pnl, slippage_paid, in_cash)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                str(entry.rebalance_date),
                json.dumps(entry.basket_before),
                json.dumps(entry.basket_after),
                json.dumps({k: round(v, 4) for k, v in entry.momentum_scores.items()}),
                entry.portfolio_value,
                entry.realized_pnl,
                json.dumps({k: round(v, 6) for k, v in entry.slippage_paid.items()}),
                int(entry.in_cash),
            ),
        )
        await self._conn.commit()
        logger.debug("Rebalance recorded: %s", entry.rebalance_date)

    async def load_portfolio(self) -> tuple[dict[str, float], float, int]:
        """Load persisted portfolio state.

        Returns:
            (holdings, cash, days_since_rebalance)
            holdings: {symbol: shares}
        """
        self._require_connected()
        cursor = await self._conn.execute(
            "SELECT key, value FROM rotation_portfolio"
        )
        rows = await cursor.fetchall()
        kv = {row["key"]: row["value"] for row in rows}

        holdings: dict[str, float] = json.loads(kv.get("holdings", "{}"))
        cash = float(kv.get("cash", "0.0"))
        days_since_rebalance = int(kv.get("days_since_rebalance", "0"))
        return holdings, cash, days_since_rebalance

    async def save_portfolio(
        self,
        holdings: dict[str, float],
        cash: float,
        days_since_rebalance: int,
    ) -> None:
        """Persist current portfolio state (upsert)."""
        self._require_connected()
        now = datetime.now(UTC).isoformat()
        rows = [
            ("holdings", json.dumps(holdings), now),
            ("cash", str(cash), now),
            ("days_since_rebalance", str(days_since_rebalance), now),
        ]
        await self._conn.executemany(
            """
            INSERT INTO rotation_portfolio (key, value, updated_at)
            VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            rows,
        )
        await self._conn.commit()

    async def last_rebalance(self) -> RebalanceEntry | None:
        """Return the most recent rebalance entry, or None."""
        self._require_connected()
        cursor = await self._conn.execute(
            "SELECT * FROM rebalance_journal ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        from datetime import date as _date
        return RebalanceEntry(
            rebalance_date=_date.fromisoformat(row["rebalance_date"]),
            basket_before=json.loads(row["basket_before"]),
            basket_after=json.loads(row["basket_after"]),
            momentum_scores=json.loads(row["momentum_scores"]),
            portfolio_value=row["portfolio_value"],
            realized_pnl=row["realized_pnl"],
            slippage_paid=json.loads(row["slippage_paid"]),
            in_cash=bool(row["in_cash"]),
        )

    def _require_connected(self) -> None:
        if self._conn is None:
            raise RuntimeError("Call connect() before using TradeJournal")
