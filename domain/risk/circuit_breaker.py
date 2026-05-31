"""CircuitBreaker — daily-loss kill switch that halts all trading when triggered.

This is a pure domain service.  It maintains only in-memory state; persistence
of the tripped flag across restarts is the responsibility of the infrastructure
layer (e.g. the SQLite adapter writing a `circuit_breaker_state` record).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List


@dataclass
class DailyLossRecord:
    """Single entry in the circuit-breaker loss ledger."""

    trade_date: date
    realised_loss: float  # positive = money lost
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CircuitBreaker:
    """Tracks cumulative daily loss and trips when the limit is exceeded.

    Once tripped the breaker remains open (halted) until manually reset or
    until midnight UTC rolls the trading day over.

    Args:
        daily_loss_limit_pct: Fraction of NAV that trips the breaker (default 5 %).
        auto_reset_daily:     If True, the breaker resets automatically at UTC midnight.
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 0.05,
        auto_reset_daily: bool = True,
    ) -> None:
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.auto_reset_daily = auto_reset_daily
        self._tripped: bool = False
        self._trip_reason: str = ""
        self._current_date: date = date.today()
        self._daily_loss: float = 0.0
        self._loss_log: List[DailyLossRecord] = []

    @property
    def is_tripped(self) -> bool:
        """True when trading is halted due to daily-loss limit breach."""
        if self.auto_reset_daily:
            today = date.today()
            if today != self._current_date:
                self._roll_day(today)
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    @property
    def daily_loss(self) -> float:
        """Cumulative realised loss for the current trading day."""
        return self._daily_loss

    def record_loss(self, loss: float, nav: float) -> bool:
        """Register a realised loss and trip the breaker if the limit is hit.

        Args:
            loss: Positive dollar amount lost on a trade.
            nav:  Current NAV used to compute the loss percentage.

        Returns:
            True if the breaker was just tripped by this call.
        """
        if loss <= 0:
            return False
        if self.auto_reset_daily:
            today = date.today()
            if today != self._current_date:
                self._roll_day(today)

        self._daily_loss += loss
        self._loss_log.append(
            DailyLossRecord(trade_date=self._current_date, realised_loss=loss)
        )

        if nav > 0 and self._daily_loss / nav >= self.daily_loss_limit_pct:
            self._trip(nav)
            return True
        return False

    def record_gain(self, gain: float) -> None:
        """Netting gains against daily loss (only reduces loss, never below 0)."""
        self._daily_loss = max(0.0, self._daily_loss - gain)

    def reset(self, reason: str = "manual reset") -> None:
        """Manually reset the breaker (e.g. after operator review)."""
        self._tripped = False
        self._trip_reason = ""
        self._daily_loss = 0.0

    def status_summary(self) -> dict:
        """Machine-readable snapshot for dashboards and alerts."""
        return {
            "tripped": self._tripped,
            "daily_loss": self._daily_loss,
            "limit_pct": self.daily_loss_limit_pct,
            "trip_reason": self._trip_reason,
            "trading_date": str(self._current_date),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trip(self, nav: float) -> None:
        pct = self._daily_loss / nav
        self._tripped = True
        self._trip_reason = (
            f"Daily loss {self._daily_loss:.2f} ({pct:.2%}) exceeded "
            f"limit of {self.daily_loss_limit_pct:.2%} of NAV {nav:.2f}"
        )

    def _roll_day(self, new_date: date) -> None:
        """Reset state for a new trading day."""
        self._current_date = new_date
        self._daily_loss = 0.0
        self._tripped = False
        self._trip_reason = ""