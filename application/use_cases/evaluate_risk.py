"""EvaluateRisk use case — runs all risk checks before any order is placed.

This use case is called synchronously by RunStrategy before building an Order.
It delegates to the domain RiskEngine and optionally queries the CircuitBreaker.
It returns a RiskVerdict so the caller can decide what to do.
"""

from __future__ import annotations

import logging

from domain.entities.signal import Signal
from domain.portfolio.portfolio_state import PortfolioState
from domain.risk.circuit_breaker import CircuitBreaker
from domain.risk.risk_engine import RiskEngine, RiskVerdict

logger = logging.getLogger(__name__)


class EvaluateRisk:
    """Application service wrapping the domain RiskEngine.

    Separating this into its own use case keeps RunStrategy lean and makes
    risk evaluation independently testable.

    Args:
        risk_engine:      Domain risk engine (holds the hardcoded limits).
        circuit_breaker:  Domain circuit breaker (daily-loss kill-switch).
        daily_loss_tracker: Callable that returns today's cumulative loss.
                            Defaults to returning 0.0 if not supplied.
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._risk_engine = risk_engine
        self._circuit_breaker = circuit_breaker

    def run(self, signal: Signal, portfolio: PortfolioState) -> RiskVerdict:
        """Evaluate signal against all risk rules and return a RiskVerdict.

        Args:
            signal:    The candidate Signal to evaluate.
            portfolio: Current portfolio state (NAV, open positions).

        Returns:
            RiskVerdict.approved=True only when every rule passes.
        """
        # Fast-path: circuit breaker already tripped
        if self._circuit_breaker.is_tripped:
            logger.warning(
                "Risk evaluation short-circuited: circuit breaker tripped (%s)",
                self._circuit_breaker.trip_reason,
            )
            return RiskVerdict(
                approved=False,
                reasons=[f"Circuit breaker tripped: {self._circuit_breaker.trip_reason}"],
            )

        verdict = self._risk_engine.evaluate(
            signal=signal,
            nav=portfolio.nav,
            daily_loss=self._circuit_breaker.daily_loss,
            open_positions=portfolio.open_positions,
        )

        if verdict.approved:
            logger.debug(
                "Risk approved: %s %s qty=%.6f",
                signal.symbol,
                signal.direction.value,
                verdict.adjusted_qty,
            )
        else:
            logger.info(
                "Risk rejected %s %s — %s",
                signal.symbol,
                signal.direction.value,
                "; ".join(verdict.reasons),
            )

        return verdict
