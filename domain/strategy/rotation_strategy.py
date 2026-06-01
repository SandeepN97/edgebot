"""Cross-sectional relative-strength rotation — pure domain logic.

Identical mathematical spec to the validated backtest:
  - 90-day risk-adjusted momentum: total_return / daily_return_std
  - Hold top-2 by score, equal weight
  - Cash filter: sit out when all scores ≤ 0
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOOKBACK: int = 90
TOP_N: int = 2
REBALANCE_EVERY: int = 5   # trading days between rebalances


@dataclass
class RotationDecision:
    target: dict[str, float]   # {symbol: portfolio fraction} — empty means cash
    scores: dict[str, float]   # all computed scores this cycle
    in_cash: bool


def compute_decision(price_history: dict[str, list[float]]) -> RotationDecision:
    """Return the target basket given rolling daily close histories.

    Args:
        price_history: {symbol: [close_0, ..., close_N]}, newest last.
                       Lists shorter than LOOKBACK+1 bars are skipped.
    """
    scores: dict[str, float] = {}
    for sym, prices in price_history.items():
        score = _momentum_score(prices)
        if score is not None:
            scores[sym] = score

    pos_scores = {s: v for s, v in scores.items() if v > 0}
    if not pos_scores:
        return RotationDecision(target={}, scores=scores, in_cash=True)

    top = sorted(pos_scores, key=pos_scores.__getitem__, reverse=True)[:TOP_N]
    frac = 1.0 / len(top)
    return RotationDecision(
        target={s: frac for s in top},
        scores=scores,
        in_cash=False,
    )


def _momentum_score(prices: list[float]) -> float | None:
    if len(prices) < LOOKBACK + 1:
        return None
    window = prices[-(LOOKBACK + 1):]
    daily_rets = [window[i + 1] / window[i] - 1.0 for i in range(len(window) - 1)]
    vol = float(np.std(daily_rets))
    if vol <= 0:
        return None
    total_ret = window[-1] / window[0] - 1.0
    return total_ret / vol
