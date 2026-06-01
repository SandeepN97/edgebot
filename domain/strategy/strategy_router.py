"""StrategyRouter — pure domain class, zero infrastructure imports.

Maps the current bar's regime to a strategy_id so only the right strategy
fires each bar:
    TRENDING  (ADX ≥ 25) → ema_crossover_4h
    RANGING   (ADX < 20) → mean_reversion_4h
    NEUTRAL              → sit out
    VOLATILE             → sit out

Unknown regimes are treated as sit-out for safety.
"""
from __future__ import annotations

from dataclasses import dataclass

_REGIME_MAP: dict[str, str | None] = {
    "TRENDING": "ema_crossover_4h",
    "RANGING": None,   # sit out — MR not routed this round
    "NEUTRAL": None,
    "VOLATILE": None,
}


@dataclass(frozen=True)
class RouterDecision:
    regime: str
    strategy_id: str | None  # None → sit out this bar

    @property
    def should_trade(self) -> bool:
        return self.strategy_id is not None


class StrategyRouter:
    """Select the appropriate strategy for the current market regime."""

    def route(self, regime: str) -> RouterDecision:
        strategy_id = _REGIME_MAP.get(regime)  # unknown → None (sit out)
        return RouterDecision(regime=regime, strategy_id=strategy_id)