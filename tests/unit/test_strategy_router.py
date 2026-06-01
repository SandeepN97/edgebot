"""Unit tests for StrategyRouter — zero infrastructure dependencies."""
from __future__ import annotations

import pytest

from domain.strategy.strategy_router import RouterDecision, StrategyRouter


class TestStrategyRouter:
    def setup_method(self) -> None:
        self.router = StrategyRouter()

    def test_trending_routes_to_ema(self) -> None:
        d = self.router.route("TRENDING")
        assert d.strategy_id == "ema_crossover_4h"

    def test_ranging_sits_out(self) -> None:
        d = self.router.route("RANGING")
        assert d.strategy_id is None

    def test_neutral_sits_out(self) -> None:
        d = self.router.route("NEUTRAL")
        assert d.strategy_id is None

    def test_volatile_sits_out(self) -> None:
        d = self.router.route("VOLATILE")
        assert d.strategy_id is None

    def test_unknown_regime_sits_out(self) -> None:
        d = self.router.route("UNKNOWN")
        assert d.strategy_id is None

    def test_trending_should_trade_is_true(self) -> None:
        assert self.router.route("TRENDING").should_trade is True

    def test_ranging_should_trade_is_false(self) -> None:
        assert self.router.route("RANGING").should_trade is False

    def test_neutral_should_trade_is_false(self) -> None:
        assert self.router.route("NEUTRAL").should_trade is False

    def test_volatile_should_trade_is_false(self) -> None:
        assert self.router.route("VOLATILE").should_trade is False

    def test_regime_stored_on_decision(self) -> None:
        d = self.router.route("RANGING")
        assert d.regime == "RANGING"

    def test_decision_is_frozen(self) -> None:
        d = self.router.route("TRENDING")
        with pytest.raises((AttributeError, TypeError)):
            d.strategy_id = "other"  # type: ignore[misc]

    def test_all_known_regimes_return_router_decision(self) -> None:
        for regime in ("TRENDING", "RANGING", "NEUTRAL", "VOLATILE"):
            d = self.router.route(regime)
            assert isinstance(d, RouterDecision)