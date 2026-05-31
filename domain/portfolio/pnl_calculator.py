"""PnLCalculator — computes realized and unrealized profit & loss for the portfolio.

This is a stateless domain service that operates only on domain entities.
It aggregates PnL metrics across a collection of trades and positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from domain.entities.position import Position
from domain.entities.trade import Trade


@dataclass(frozen=True)
class PnLReport:
    """Comprehensive PnL summary across realised trades and open positions.

    Attributes:
        realized_pnl:        Total net PnL from closed trades.
        unrealized_pnl:      Mark-to-market PnL from open positions.
        total_pnl:           realized + unrealized.
        total_fees:          Sum of all fees paid (entry + exit).
        win_count:           Number of profitable closed trades.
        loss_count:          Number of unprofitable closed trades.
        win_rate:            win_count / (win_count + loss_count).
        avg_win:             Average net PnL of winning trades.
        avg_loss:            Average net PnL of losing trades (magnitude).
        profit_factor:       gross_wins / gross_losses (> 1 is positive edge).
        largest_win:         Best single trade net PnL.
        largest_loss:        Worst single trade net PnL (negative number).
        expectancy:          win_rate × avg_win − (1 − win_rate) × avg_loss.
    """

    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    win_count: int
    loss_count: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_trades(self) -> int:
        return self.win_count + self.loss_count

    @property
    def win_rate(self) -> float:
        return self.win_count / self.total_trades if self.total_trades else 0.0

    @property
    def profit_factor(self) -> Optional[float]:
        """Gross wins / gross losses; None when there are no losing trades."""
        gross_wins = self.win_count * self.avg_win
        gross_losses = self.loss_count * self.avg_loss
        return gross_wins / gross_losses if gross_losses > 0 else None

    @property
    def expectancy(self) -> float:
        """Expected value per trade in dollar terms."""
        return self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss


class PnLCalculator:
    """Stateless service for computing PnL metrics from domain collections."""

    @staticmethod
    def compute_realized(trades: List[Trade]) -> float:
        """Sum net PnL across all completed trades."""
        return sum(t.net_pnl for t in trades)

    @staticmethod
    def compute_unrealized(positions: List[Position]) -> float:
        """Sum mark-to-market PnL across all open positions."""
        return sum(p.unrealized_pnl for p in positions)

    @staticmethod
    def total_fees(trades: List[Trade]) -> float:
        """Aggregate all fees paid across completed trades."""
        return sum(t.total_fees for t in trades)

    @classmethod
    def full_report(
        cls,
        trades: List[Trade],
        positions: List[Position],
    ) -> PnLReport:
        """Build a PnLReport from a list of completed trades and open positions."""
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]

        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(t.net_pnl for t in losses) / len(losses)) if losses else 0.0
        largest_win = max((t.net_pnl for t in wins), default=0.0)
        largest_loss = min((t.net_pnl for t in losses), default=0.0)

        return PnLReport(
            realized_pnl=cls.compute_realized(trades),
            unrealized_pnl=cls.compute_unrealized(positions),
            total_fees=cls.total_fees(trades),
            win_count=len(wins),
            loss_count=len(losses),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )

    @staticmethod
    def max_drawdown(equity_curve: List[float]) -> float:
        """Compute maximum drawdown from a list of equity values."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def sharpe_ratio(
        returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> Optional[float]:
        """Annualised Sharpe ratio from a list of periodic returns."""
        if len(returns) < 2:
            return None
        import statistics
        excess = [r - risk_free_rate / periods_per_year for r in returns]
        mean_excess = statistics.mean(excess)
        std_excess = statistics.stdev(excess)
        if std_excess == 0:
            return None
        return (mean_excess / std_excess) * (periods_per_year ** 0.5)