"""RiskEngine — enforces all pre-trade risk rules before an order is placed.

This is a pure domain service: it receives domain objects and returns a
RiskVerdict.  No exchange, database, or network calls are made here.  The
application layer calls evaluate() and only proceeds to order placement when
the verdict is approved.

Hard-coded constants (can be overridden via constructor for testing):
    MAX_POSITION_SIZE_PCT  = 0.02   (2 % of NAV per trade)
    DAILY_LOSS_LIMIT_PCT   = 0.05   (5 % of NAV daily drawdown kill-switch)
    MAX_OPEN_POSITIONS     = 3      (maximum concurrent open positions)
    MIN_RISK_REWARD_RATIO  = 2.0    (must risk 1 to make at least 2)
    FEE_BUFFER_PCT         = 0.002  (0.2 % reserved per side for fees)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from domain.entities.signal import Signal
from domain.entities.position import Position


MAX_POSITION_SIZE_PCT: float = 0.02
DAILY_LOSS_LIMIT_PCT: float = 0.05
MAX_OPEN_POSITIONS: int = 3
MIN_RISK_REWARD_RATIO: float = 2.0
FEE_BUFFER_PCT: float = 0.002


@dataclass(frozen=True)
class RiskVerdict:
    """Outcome of a risk evaluation.

    Attributes:
        approved:      True when all checks pass.
        reasons:       List of human-readable rule violations (empty if approved).
        adjusted_qty:  Suggested position size after capping; None if rejected.
    """

    approved: bool
    reasons: List[str]
    adjusted_qty: float | None = None


class RiskEngine:
    """Stateless domain service that validates a Signal against risk rules.

    All limits are configurable at construction time; the module-level
    constants serve as documented defaults.
    """

    def __init__(
        self,
        max_position_size_pct: float = MAX_POSITION_SIZE_PCT,
        daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT,
        max_open_positions: int = MAX_OPEN_POSITIONS,
        min_risk_reward_ratio: float = MIN_RISK_REWARD_RATIO,
        fee_buffer_pct: float = FEE_BUFFER_PCT,
    ) -> None:
        self.max_position_size_pct = max_position_size_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_open_positions = max_open_positions
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.fee_buffer_pct = fee_buffer_pct

    def evaluate(
        self,
        signal: Signal,
        nav: float,
        daily_loss: float,
        open_positions: List[Position],
    ) -> RiskVerdict:
        """Run all risk checks and return a RiskVerdict.

        Args:
            signal:          The candidate signal to evaluate.
            nav:             Current net asset value of the portfolio (cash + open positions).
            daily_loss:      Cumulative realised loss today (positive number = money lost).
            open_positions:  All currently open Position objects.

        Returns:
            RiskVerdict with approved=True only when every check passes.
        """
        violations: List[str] = []

        # 1. Daily loss circuit-breaker
        violations += self._check_daily_loss(daily_loss, nav)

        # 2. Max concurrent open positions
        violations += self._check_open_positions(open_positions)

        # 3. Minimum risk-reward ratio
        violations += self._check_risk_reward(signal)

        # 4. Duplicate position guard
        violations += self._check_duplicate(signal, open_positions)

        if violations:
            return RiskVerdict(approved=False, reasons=violations)

        # 5. Compute (and cap) position size
        raw_qty = self._compute_qty(signal, nav)
        return RiskVerdict(approved=True, reasons=[], adjusted_qty=raw_qty)

    # ------------------------------------------------------------------
    # Private check helpers — each returns a list of violation strings
    # ------------------------------------------------------------------

    def _check_daily_loss(self, daily_loss: float, nav: float) -> List[str]:
        if nav <= 0:
            return ["NAV must be positive"]
        if daily_loss / nav >= self.daily_loss_limit_pct:
            return [
                f"Daily loss limit breached: {daily_loss/nav:.2%} >= "
                f"{self.daily_loss_limit_pct:.2%} of NAV"
            ]
        return []

    def _check_open_positions(self, open_positions: List[Position]) -> List[str]:
        if len(open_positions) >= self.max_open_positions:
            return [
                f"Max open positions reached: {len(open_positions)} / "
                f"{self.max_open_positions}"
            ]
        return []

    def _check_risk_reward(self, signal: Signal) -> List[str]:
        rr = signal.risk_reward_ratio
        if rr is None:
            return ["Cannot compute R:R without entry_price on signal"]
        if rr < self.min_risk_reward_ratio:
            return [
                f"R:R {rr:.2f} < minimum {self.min_risk_reward_ratio:.2f}"
            ]
        return []

    def _check_duplicate(
        self, signal: Signal, open_positions: List[Position]
    ) -> List[str]:
        for pos in open_positions:
            if pos.symbol == signal.symbol and pos.direction == signal.direction:
                return [
                    f"Duplicate position: already {signal.direction.value} "
                    f"{signal.symbol}"
                ]
        return []

    def _compute_qty(self, signal: Signal, nav: float) -> float:
        """Size the position: cap at MAX_POSITION_SIZE_PCT of NAV, reserve fee buffer."""
        if signal.entry_price is None or signal.entry_price <= 0:
            return 0.0
        effective_pct = self.max_position_size_pct - self.fee_buffer_pct
        max_notional = nav * effective_pct
        return max_notional / signal.entry_price