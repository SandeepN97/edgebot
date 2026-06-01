"""Cross-sectional relative-strength rotation backtest.

Pure pandas/numpy simulation — no Backtrader (multi-asset rotation is awkward
in a single-feed engine).

Universe : 10 coins — BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT
Signal   : risk-adjusted 90-day momentum = total_return / daily_return_std
Holding  : top-2 by score, equal weight, rebalanced every 5 trading days
Cash filter : hold cash when ALL candidate scores are ≤ 0
Commission  : 0.1 % per trade side (applied symmetrically on buys and sells)
Annualisation: 252 bars / year (user-specified daily convention)

Benchmarks (same tune window, same starting cash):
  1. Buy-and-hold BTC — single-asset baseline
  2. Equal-weight hold-all — buy all available symbols on day 1, never rebalance

Verdict gate: strategy beats BOTH benchmarks on total return
              AND Sharpe ≥ 0.5
              → "MOMENTUM SHOWS LIFE"
              otherwise → "NO EDGE OVER BUY-AND-HOLD"

Tune window : earliest available → 2021-12-31
Sealed      : 2022-2024 — never touched
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

ROTATION_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT",
    "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
]

TUNE_START      = "2017-01-01"
TUNE_END        = "2022-01-01"   # exclusive; 2022-2024 sealed
LOOKBACK        = 90             # days for momentum score
REBALANCE_EVERY = 5              # trading days between rebalances
TOP_N           = 2              # coins to hold
COMMISSION      = 0.001          # 0.1 % per side
STARTING_CASH   = 100.0
BARS_PER_YEAR   = 252
MIN_TRADE_USD   = 0.50           # skip legs smaller than this to avoid float dust


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RotationResult:
    period: str
    sharpe: float | None
    max_dd_pct: float | None
    total_return_pct: float
    final_value: float
    n_rebalances: int
    n_trades: int
    cash_periods: int               # rebalance periods when all scores ≤ 0
    weekly_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    name: str
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float | None
    final_value: float


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _sharpe(daily_returns: pd.Series) -> float | None:
    if len(daily_returns) < 20:
        return None
    std = float(daily_returns.std())
    if std <= 0:
        return None
    return float(daily_returns.mean() / std * np.sqrt(BARS_PER_YEAR))


def _max_dd(values: pd.Series) -> float | None:
    if len(values) < 2:
        return None
    running_max = values.cummax()
    dd = (values - running_max) / running_max.replace(0.0, np.nan)
    return float(abs(dd.min()) * 100)


def _ok(x: Any) -> bool:
    """Return True iff x is a finite, non-NaN number."""
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _momentum_score(close_series: pd.Series) -> float | None:
    """Risk-adjusted 90-day momentum: total_return / daily_return_std."""
    if len(close_series) < LOOKBACK + 1:
        return None
    window = close_series.iloc[-(LOOKBACK + 1):]
    if window.isna().any():
        return None
    daily_rets = window.pct_change().dropna()
    vol = float(daily_rets.std())
    if vol <= 0:
        return None
    total_ret = float(window.iloc[-1] / window.iloc[0]) - 1.0
    return total_ret / vol


# ---------------------------------------------------------------------------
# Price matrix
# ---------------------------------------------------------------------------

def load_price_matrix(symbols: list[str] | None = None) -> pd.DataFrame:
    """Load daily close prices for all rotation symbols into one DataFrame.

    Symbols with no cached parquet are silently excluded so the rotation
    runs on whatever subset is available.
    """
    if symbols is None:
        symbols = ROTATION_SYMBOLS
    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        path = RAW_DIR / f"{sym.replace('/', '_')}_1d.parquet"
        if not path.exists():
            logger.warning("No daily raw data for %s — excluded from rotation", sym)
            continue
        df = pd.read_parquet(path)
        frames[sym] = df["close"]
    if not frames:
        raise RuntimeError("No daily data found for any rotation symbol")
    matrix = pd.DataFrame(frames).sort_index()
    if matrix.index.tz is None:
        matrix.index = matrix.index.tz_localize("UTC")
    return matrix


# ---------------------------------------------------------------------------
# Rebalancing logic
# ---------------------------------------------------------------------------

def _rebalance(
    holdings: dict[str, float],
    cash: float,
    port_val: float,
    target: dict[str, float],        # {symbol: fraction of portfolio}
    prices: pd.Series,
) -> dict[str, Any]:
    """Execute partial rebalancing — only trade positions that change."""
    holdings = dict(holdings)        # local copy
    n_trades = 0

    # 1. Sell coins leaving the portfolio
    for sym in [s for s in list(holdings.keys()) if s not in target]:
        p = float(prices.get(sym, np.nan))
        shares = holdings.pop(sym, 0.0)
        if _ok(p) and shares > 0:
            cash += shares * p * (1 - COMMISSION)
            n_trades += 1

    # 2. Recompute available value after sells
    port_val_now = cash + sum(
        holdings.get(s, 0.0) * float(prices[s])
        for s in holdings
        if s in prices.index and _ok(prices[s])
    )

    # 3. Adjust each target coin toward its target fraction
    for sym, frac in target.items():
        p = float(prices.get(sym, np.nan))
        if not _ok(p):
            continue
        target_val = port_val_now * frac
        current_val = holdings.get(sym, 0.0) * p
        diff = target_val - current_val

        if diff > MIN_TRADE_USD:
            spend = min(diff, cash)
            if spend < MIN_TRADE_USD:
                continue
            shares_bought = spend * (1 - COMMISSION) / p
            holdings[sym] = holdings.get(sym, 0.0) + shares_bought
            cash -= spend
            n_trades += 1
        elif diff < -MIN_TRADE_USD:
            shares_to_sell = min(abs(diff) / p, holdings.get(sym, 0.0))
            if shares_to_sell * p < MIN_TRADE_USD:
                continue
            cash += shares_to_sell * p * (1 - COMMISSION)
            holdings[sym] = holdings.get(sym, 0.0) - shares_to_sell
            if holdings[sym] < 1e-12:
                del holdings[sym]
            n_trades += 1

    return {"holdings": holdings, "cash": cash, "n_trades": n_trades}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_rotation(price_matrix: pd.DataFrame) -> RotationResult:
    """Run cross-sectional rotation on the tune window. Sealed 2022-2024 untouched."""
    mask = (price_matrix.index >= TUNE_START) & (price_matrix.index < TUNE_END)
    tune_dates = price_matrix.index[mask]

    if len(tune_dates) < LOOKBACK + REBALANCE_EVERY:
        raise RuntimeError(
            f"Tune window only {len(tune_dates)} bars — need ≥ {LOOKBACK + REBALANCE_EVERY}"
        )

    cash: float = STARTING_CASH
    holdings: dict[str, float] = {}
    port_values: list[float] = []
    n_rebalances = 0
    n_trades = 0
    cash_periods = 0
    weekly_log: list[dict[str, Any]] = []

    for i, date in enumerate(tune_dates):
        day_prices = price_matrix.loc[date]

        # Portfolio value at end of day
        port_val = cash + sum(
            holdings.get(s, 0.0) * float(day_prices[s])
            for s in holdings
            if s in day_prices.index and _ok(day_prices[s])
        )
        port_values.append(port_val)

        # Rebalance every REBALANCE_EVERY days
        if i % REBALANCE_EVERY != 0:
            continue

        n_rebalances += 1

        # Compute momentum scores using full history up to this date
        scores: dict[str, float] = {}
        for sym in price_matrix.columns:
            hist = price_matrix[sym].loc[:date].dropna()
            score = _momentum_score(hist)
            if score is not None:
                scores[sym] = score

        pos_scores = {s: v for s, v in scores.items() if v > 0}

        if not pos_scores:
            # Cash filter: no positive-momentum coins — liquidate
            cash_periods += 1
            for sym, shares in list(holdings.items()):
                p = float(day_prices.get(sym, np.nan))
                if _ok(p) and shares > 0:
                    cash += shares * p * (1 - COMMISSION)
                    n_trades += 1
            holdings.clear()
            weekly_log.append({
                "date": date, "held": [], "scores": {},
                "port_val": port_val, "action": "CASH",
            })
            continue

        top = sorted(pos_scores, key=pos_scores.get, reverse=True)[:TOP_N]
        target = {s: 1.0 / len(top) for s in top}

        prev_held = set(holdings.keys())
        result = _rebalance(holdings, cash, port_val, target, day_prices)
        holdings = result["holdings"]
        cash = result["cash"]
        n_trades += result["n_trades"]

        action = "HOLD" if set(holdings.keys()) == prev_held and result["n_trades"] == 0 else "ROTATE"
        weekly_log.append({
            "date": date,
            "held": top,
            "scores": {s: round(scores[s], 3) for s in top},
            "port_val": port_val,
            "action": action,
        })

    values = pd.Series(port_values, index=tune_dates)
    daily_rets = values.pct_change().dropna()
    total_ret = float((values.iloc[-1] / values.iloc[0] - 1.0) * 100)

    return RotationResult(
        period=f"TUNE {tune_dates[0].date()} → {tune_dates[-1].date()}",
        sharpe=_sharpe(daily_rets),
        max_dd_pct=_max_dd(values),
        total_return_pct=total_ret,
        final_value=float(values.iloc[-1]),
        n_rebalances=n_rebalances,
        n_trades=n_trades,
        cash_periods=cash_periods,
        weekly_log=weekly_log,
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(price_matrix: pd.DataFrame) -> tuple[BenchmarkResult, BenchmarkResult]:
    """BTC buy-and-hold and equal-weight hold-all benchmarks over tune window."""
    mask = (price_matrix.index >= TUNE_START) & (price_matrix.index < TUNE_END)
    prices = price_matrix[mask]

    # --- BTC buy-and-hold ---
    btc = prices["BTC/USDT"].dropna()
    if len(btc) < 2:
        raise RuntimeError("Insufficient BTC/USDT data for benchmark")
    btc_vals = STARTING_CASH * btc / float(btc.iloc[0])
    btc_rets = btc_vals.pct_change().dropna()
    btc_bm = BenchmarkResult(
        name="BTC buy-and-hold",
        total_return_pct=float((btc_vals.iloc[-1] / btc_vals.iloc[0] - 1.0) * 100),
        sharpe=_sharpe(btc_rets),
        max_dd_pct=_max_dd(btc_vals),
        final_value=float(btc_vals.iloc[-1]),
    )

    # --- Equal-weight hold-all ---
    # Invest equally in every symbol available on the FIRST day of the tune window.
    # Coins listing after day 1 are excluded (no survivorship-bias adjustment needed
    # for a simple baseline — the bias, if any, favours the benchmark not the strategy).
    first_row = prices.dropna(how="all").iloc[0]
    available = [s for s in first_row.index if _ok(first_row[s])]
    if not available:
        raise RuntimeError("No symbols available on first tune day for EW benchmark")
    per_coin = STARTING_CASH / len(available)
    initial_shares = {sym: per_coin / float(first_row[sym]) for sym in available}

    ew_values: list[float] = []
    for _, row in prices.iterrows():
        val = sum(
            initial_shares[sym] * float(row[sym])
            for sym in available
            if sym in row.index and _ok(row[sym])
        )
        ew_values.append(val)

    ew_vals = pd.Series(ew_values, index=prices.index)
    ew_rets = ew_vals.pct_change().dropna()
    ew_bm = BenchmarkResult(
        name=f"Equal-weight hold-all ({len(available)} coins)",
        total_return_pct=float((ew_vals.iloc[-1] / ew_vals.iloc[0] - 1.0) * 100),
        sharpe=_sharpe(ew_rets),
        max_dd_pct=_max_dd(ew_vals),
        final_value=float(ew_vals.iloc[-1]),
    )

    return btc_bm, ew_bm
