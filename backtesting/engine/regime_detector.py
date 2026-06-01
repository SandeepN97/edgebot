"""Classify each bar into a market regime based on ADX and ATR.

Priority order (first match wins):
    VOLATILE  — ATR > rolling_mean(ATR, 50) + 2 × std  (outlier spike)
    TRENDING  — ADX > 25
    RANGING   — ADX < 20
    NEUTRAL   — 20 ≤ ADX ≤ 25  (transitional)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ATR_WINDOW = 50
_ADX_TRENDING = 25.0
_ADX_RANGING = 20.0


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a ``regime`` column added."""
    out = df.copy()

    atr_mean = out["atr14"].rolling(_ATR_WINDOW).mean()
    atr_std = out["atr14"].rolling(_ATR_WINDOW).std(ddof=1)
    volatile_mask = out["atr14"] > atr_mean + 2.0 * atr_std

    conditions = [
        volatile_mask,
        out["adx14"] > _ADX_TRENDING,
        out["adx14"] < _ADX_RANGING,
    ]
    choices = ["VOLATILE", "TRENDING", "RANGING"]
    out["regime"] = np.select(conditions, choices, default="NEUTRAL")
    return out