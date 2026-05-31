"""PositionSizer — computes trade size using Kelly Criterion or fixed-fraction methods.

This is a stateless domain service.  It has no knowledge of exchanges, databases,
or any infrastructure.  The application layer selects a sizing method and passes
the result to the risk engine for final capping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SizingMethod(Enum):
    """Available position-sizing algorithms."""

    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    FIXED_FRACTION = "fixed_fraction"
    FIXED_NOTIONAL = "fixed_notional"


@dataclass(frozen=True)
class SizingResult:
    """Output of a position-sizing calculation.

    Attributes:
        quantity:        Recommended base-asset quantity.
        notional:        Equivalent notional value at the given price.
        fraction_of_nav: What fraction of NAV this represents.
        method:          Which algorithm produced this result.
    """

    quantity: float
    notional: float
    fraction_of_nav: float
    method: SizingMethod


class PositionSizer:
    """Calculates position sizes using either Kelly Criterion or fixed-fraction logic.

    Kelly Criterion formula:
        f* = (p * b - q) / b
        where p = win probability, q = 1-p, b = avg_win / avg_loss

    Half-Kelly halves f* to reduce variance in live trading.
    Fixed-fraction simply allocates a constant percentage of NAV.
    Fixed-notional allocates an absolute dollar amount regardless of NAV.
    """

    def __init__(
        self,
        max_fraction: float = 0.02,
        fixed_fraction: float = 0.01,
        fixed_notional: float = 100.0,
    ) -> None:
        self.max_fraction = max_fraction
        self.fixed_fraction = fixed_fraction
        self.fixed_notional = fixed_notional

    def kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        nav: float,
        price: float,
        half: bool = False,
    ) -> SizingResult:
        """Full or Half-Kelly sizing.

        Args:
            win_rate:  Historical fraction of winning trades (0 < p < 1).
            avg_win:   Average profit per winning trade (positive).
            avg_loss:  Average loss per losing trade (positive magnitude).
            nav:       Current net asset value.
            price:     Entry price of the instrument.
            half:      If True, apply half-Kelly dampening.
        """
        if avg_loss <= 0:
            raise ValueError("avg_loss must be positive")
        if not 0 < win_rate < 1:
            raise ValueError("win_rate must be in (0, 1)")

        b = avg_win / avg_loss
        q = 1 - win_rate
        f_star = (win_rate * b - q) / b
        f_star = max(0.0, f_star)  # never go short via Kelly

        if half:
            f_star /= 2.0

        fraction = min(f_star, self.max_fraction)
        notional = nav * fraction
        quantity = notional / price if price > 0 else 0.0
        method = SizingMethod.HALF_KELLY if half else SizingMethod.KELLY
        return SizingResult(
            quantity=quantity,
            notional=notional,
            fraction_of_nav=fraction,
            method=method,
        )

    def fixed_fraction_size(
        self, nav: float, price: float, fraction: float | None = None
    ) -> SizingResult:
        """Allocate a fixed fraction of NAV to a trade."""
        frac = min(fraction or self.fixed_fraction, self.max_fraction)
        notional = nav * frac
        quantity = notional / price if price > 0 else 0.0
        return SizingResult(
            quantity=quantity,
            notional=notional,
            fraction_of_nav=frac,
            method=SizingMethod.FIXED_FRACTION,
        )

    def fixed_notional_size(self, price: float, notional: float | None = None) -> SizingResult:
        """Trade a fixed dollar notional regardless of portfolio size."""
        amount = notional or self.fixed_notional
        quantity = amount / price if price > 0 else 0.0
        return SizingResult(
            quantity=quantity,
            notional=amount,
            fraction_of_nav=0.0,
            method=SizingMethod.FIXED_NOTIONAL,
        )

    def optimal_f(self, trades: list[float], nav: float, price: float) -> SizingResult:
        """Approximate Optimal-f (Vince) via grid search over historical trade PnLs.

        The HPR for each trade is: 1 + f * pnl / worst_loss
        where worst_loss is the largest single loss (positive magnitude).
        TWR = product of all HPRs; we maximise TWR over f in (0, max_fraction].

        Falls back to fixed_fraction when there are no historical losses (no
        worst_loss anchor means the formula is undefined).

        Args:
            trades: Net PnL per trade (positive = profit, negative = loss).
            nav:    Current net asset value.
            price:  Entry price for quantity calculation.
        """
        if not trades:
            return self.fixed_fraction_size(nav, price)

        losses = [p for p in trades if p < 0]
        if not losses:
            # No loss history → can't anchor the HPR formula; fall back safely.
            return self.fixed_fraction_size(nav, price)

        worst_loss = max(abs(p) for p in losses)

        # Baseline: f=0 means no trade, TWR=1.0.  Any f that beats 1.0 is worth taking.
        best_f, best_twr = 0.0, 1.0
        for f_candidate in (i / 100 for i in range(1, 51)):
            twr = 1.0
            for pnl in trades:
                factor = 1.0 + f_candidate * pnl / worst_loss
                if factor <= 0:
                    twr = 0.0
                    break
                twr *= factor
            if twr > best_twr:
                best_twr, best_f = twr, f_candidate

        best_f = min(best_f, self.max_fraction)
        notional = nav * best_f
        quantity = notional / price if price > 0 else 0.0
        return SizingResult(
            quantity=quantity,
            notional=notional,
            fraction_of_nav=best_f,
            method=SizingMethod.FIXED_FRACTION,
        )
