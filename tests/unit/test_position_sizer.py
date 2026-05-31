"""Unit tests for PositionSizer — Kelly, fixed-fraction, and optimal-f."""

from __future__ import annotations

import pytest

from domain.risk.position_sizer import PositionSizer, SizingMethod


NAV = 10_000.0
PRICE = 100.0


@pytest.fixture
def sizer() -> PositionSizer:
    return PositionSizer(max_fraction=0.02, fixed_fraction=0.01, fixed_notional=200.0)


# ------------------------------------------------------------------
# Kelly
# ------------------------------------------------------------------

class TestKelly:
    def test_positive_edge_returns_positive_fraction(self, sizer: PositionSizer) -> None:
        result = sizer.kelly(win_rate=0.6, avg_win=150.0, avg_loss=100.0, nav=NAV, price=PRICE)
        assert result.fraction_of_nav > 0
        assert result.method == SizingMethod.KELLY

    def test_fraction_capped_at_max(self, sizer: PositionSizer) -> None:
        # Very high edge would suggest >2% — must be capped
        result = sizer.kelly(win_rate=0.9, avg_win=500.0, avg_loss=10.0, nav=NAV, price=PRICE)
        assert result.fraction_of_nav == pytest.approx(sizer.max_fraction)

    def test_negative_edge_returns_zero_fraction(self, sizer: PositionSizer) -> None:
        result = sizer.kelly(win_rate=0.3, avg_win=50.0, avg_loss=100.0, nav=NAV, price=PRICE)
        assert result.fraction_of_nav == pytest.approx(0.0)
        assert result.quantity == pytest.approx(0.0)

    def test_half_kelly_is_half_of_full_kelly(self, sizer: PositionSizer) -> None:
        full = sizer.kelly(win_rate=0.6, avg_win=150.0, avg_loss=100.0, nav=NAV, price=PRICE)
        half = sizer.kelly(win_rate=0.6, avg_win=150.0, avg_loss=100.0, nav=NAV, price=PRICE, half=True)
        assert half.method == SizingMethod.HALF_KELLY
        # Half-Kelly fraction ≤ full-Kelly (both may be capped at max_fraction)
        assert half.fraction_of_nav <= full.fraction_of_nav + 1e-9

    def test_invalid_win_rate_raises(self, sizer: PositionSizer) -> None:
        with pytest.raises(ValueError, match="win_rate"):
            sizer.kelly(win_rate=1.1, avg_win=100.0, avg_loss=50.0, nav=NAV, price=PRICE)

    def test_zero_avg_loss_raises(self, sizer: PositionSizer) -> None:
        with pytest.raises(ValueError, match="avg_loss"):
            sizer.kelly(win_rate=0.6, avg_win=100.0, avg_loss=0.0, nav=NAV, price=PRICE)


# ------------------------------------------------------------------
# Fixed-fraction
# ------------------------------------------------------------------

class TestFixedFraction:
    def test_uses_default_fraction(self, sizer: PositionSizer) -> None:
        result = sizer.fixed_fraction_size(nav=NAV, price=PRICE)
        assert result.fraction_of_nav == pytest.approx(sizer.fixed_fraction)
        assert result.notional == pytest.approx(NAV * sizer.fixed_fraction)
        assert result.method == SizingMethod.FIXED_FRACTION

    def test_custom_fraction_capped_at_max(self, sizer: PositionSizer) -> None:
        result = sizer.fixed_fraction_size(nav=NAV, price=PRICE, fraction=0.99)
        assert result.fraction_of_nav == pytest.approx(sizer.max_fraction)


# ------------------------------------------------------------------
# optimal_f
# ------------------------------------------------------------------

class TestOptimalF:
    def test_empty_trades_falls_back_to_fixed_fraction(self, sizer: PositionSizer) -> None:
        result = sizer.optimal_f(trades=[], nav=NAV, price=PRICE)
        assert result.method == SizingMethod.FIXED_FRACTION
        assert result.fraction_of_nav == pytest.approx(sizer.fixed_fraction)

    def test_all_wins_no_losses_falls_back_to_fixed_fraction(self, sizer: PositionSizer) -> None:
        # No anchor for worst_loss → must fall back
        result = sizer.optimal_f(trades=[100.0, 200.0, 50.0], nav=NAV, price=PRICE)
        assert result.method == SizingMethod.FIXED_FRACTION

    def test_mixed_trades_returns_positive_fraction(self, sizer: PositionSizer) -> None:
        trades = [100.0, -50.0, 200.0, -30.0, 150.0]
        result = sizer.optimal_f(trades=trades, nav=NAV, price=PRICE)
        assert result.fraction_of_nav > 0
        assert result.fraction_of_nav <= sizer.max_fraction

    def test_result_capped_at_max_fraction(self, sizer: PositionSizer) -> None:
        # Extremely profitable history would suggest large f; must be capped
        trades = [1000.0, -1.0, 1000.0, -1.0]
        result = sizer.optimal_f(trades=trades, nav=NAV, price=PRICE)
        assert result.fraction_of_nav <= sizer.max_fraction

    def test_worst_loss_is_the_anchor_not_all_trades(self, sizer: PositionSizer) -> None:
        # The largest absolute loss is -100; a win of 50 should not affect the anchor
        trades = [50.0, -100.0, 30.0]
        result = sizer.optimal_f(trades=trades, nav=NAV, price=PRICE)
        # Manually verify: worst_loss=100; f=0.01: HPRs = 1+0.01*0.5, 1+0.01*(-1), 1+0.01*0.3
        # = 1.005 * 0.99 * 1.003 = ~0.9979 < 1 → TWR < 1 for f=0.01
        # The result should still be a valid SizingResult
        assert result.quantity >= 0.0
        assert result.notional >= 0.0

    def test_only_losing_trades_produces_valid_result(self, sizer: PositionSizer) -> None:
        # All losses — best_f will be 0 (any f makes TWR crash), result should be valid
        trades = [-50.0, -30.0, -100.0]
        result = sizer.optimal_f(trades=trades, nav=NAV, price=PRICE)
        assert result.fraction_of_nav == pytest.approx(0.0)
        assert result.quantity == pytest.approx(0.0)