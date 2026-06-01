"""ONE-SHOT FINAL VERDICT — sealed 2022-2024 holdout.

This script is run ONCE. The holdout is spent after this execution regardless
of outcome. No parameter changes, no tuning, no re-runs.

Strategy spec (locked — identical to tune-window run):
  Universe    : 10 coins — BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT
  Signal      : 90-day risk-adjusted momentum (return / volatility)
  Holding     : top-2 equal weight
  Rebalance   : every 5 trading days
  Cash filter : ON
  Commission  : 0.1% per side

Verdict:
  Beats BOTH benchmarks on holdout
      → "VALIDATED EDGE — rotation's downside protection justified itself
         on out-of-sample bear data."
  Otherwise
      → "NO EDGE — rotation does not add value over simply holding the
         basket, even with a bear leg."

No tweaks proposed regardless of outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import backtesting.engine.rotation_backtest as _rb

# -------------------------------------------------------------------
# Patch date window to holdout — strategy constants unchanged
# -------------------------------------------------------------------
HOLDOUT_START = "2022-01-01"
HOLDOUT_END   = "2025-01-01"   # exclusive; covers through 2024-12-31

_rb.TUNE_START = HOLDOUT_START
_rb.TUNE_END   = HOLDOUT_END

from backtesting.engine.rotation_backtest import (   # noqa: E402
    LOOKBACK, REBALANCE_EVERY, TOP_N,
    load_price_matrix, run_benchmarks, run_rotation,
)

_SEP = "=" * 68


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def _fmt(v: float | None, fmt: str = "+.3f") -> str:
    return f"{v:{fmt}}" if v is not None else "  N/A "


def main() -> None:
    print(_SEP)
    print("EdgeBot — FINAL HOLDOUT VERDICT (one-shot, sealed 2022-2024)")
    print("Strategy spec locked — no changes made before or after this run")
    print(f"Universe : BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT")
    print(f"Signal   : {LOOKBACK}d risk-adjusted momentum | top-{TOP_N} EW | "
          f"rebalance every {REBALANCE_EVERY}d | cash filter ON | 0.1% commission")
    print(f"Window   : {HOLDOUT_START} → {HOLDOUT_END[:-5]}-12-31 (out-of-sample)")
    print(_SEP)

    print("\nLoading price matrix from cache...")
    pm = load_price_matrix()
    n_sym = len(pm.columns)
    n_bars = len(pm)
    print(f"  {n_sym} symbols × {n_bars} total dates in cache")

    holdout_mask = (pm.index >= HOLDOUT_START) & (pm.index < HOLDOUT_END)
    n_holdout = holdout_mask.sum()
    print(f"  Holdout bars: {n_holdout} ({HOLDOUT_START} → 2024-12-31)")
    if n_holdout < LOOKBACK + REBALANCE_EVERY:
        print("ERROR: insufficient holdout data")
        sys.exit(1)

    print("\nRunning rotation on holdout window...")
    rotation = run_rotation(pm)
    print(f"  {rotation.period}")
    print(f"  Rebalances: {rotation.n_rebalances}  |  Trades: {rotation.n_trades}  "
          f"|  Cash periods: {rotation.cash_periods}")

    print("\nComputing benchmarks over holdout window...")
    btc_bm, ew_bm = run_benchmarks(pm)

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print("HOLDOUT RESULTS — 2022-2024 (out-of-sample, never seen during tuning)")
    print(_SEP)
    print(f"  {'Metric':<22} {'Rotation':>14} {'BTC B&H':>14} {'EW Hold-All':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")

    def row(label, r, b, e):
        print(f"  {label:<22} {r:>14} {b:>14} {e:>14}")

    row("Total return",
        _pct(rotation.total_return_pct),
        _pct(btc_bm.total_return_pct),
        _pct(ew_bm.total_return_pct))
    row("Final value ($)",
        f"${rotation.final_value:.2f}",
        f"${btc_bm.final_value:.2f}",
        f"${ew_bm.final_value:.2f}")
    row("Sharpe",
        _fmt(rotation.sharpe),
        _fmt(btc_bm.sharpe),
        _fmt(ew_bm.sharpe))
    row("Max drawdown",
        f"{rotation.max_dd_pct:.1f}%" if rotation.max_dd_pct else "N/A",
        f"{btc_bm.max_dd_pct:.1f}%" if btc_bm.max_dd_pct else "N/A",
        f"{ew_bm.max_dd_pct:.1f}%" if ew_bm.max_dd_pct else "N/A")

    # ------------------------------------------------------------------
    # Last 10 rebalancing decisions
    # ------------------------------------------------------------------
    if rotation.weekly_log:
        print(f"\n{_SEP}")
        print("LAST 10 REBALANCING DECISIONS (most recent first)")
        print(_SEP)
        print(f"  {'Date':<12} {'Action':<8} {'Held':<28} {'Scores'}")
        print(f"  {'-'*12} {'-'*8} {'-'*28} {'-'*30}")
        for entry in reversed(rotation.weekly_log[-10:]):
            held = ", ".join(s.split("/")[0] for s in entry["held"]) or "CASH"
            scores = "  ".join(
                f"{s.split('/')[0]}={v:+.2f}" for s, v in entry["scores"].items()
            ) or "—"
            print(f"  {str(entry['date'])[:10]:<12} {entry['action']:<8} {held:<28} {scores}")

    # ------------------------------------------------------------------
    # Verdict gate
    # ------------------------------------------------------------------
    beats_btc = rotation.total_return_pct > btc_bm.total_return_pct
    beats_ew  = rotation.total_return_pct > ew_bm.total_return_pct
    sharpe_ok = rotation.sharpe is not None and rotation.sharpe >= 0.5

    print(f"\n{_SEP}")
    print("FINAL VERDICT GATE")
    print(_SEP)

    def gate(label, passed, detail):
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {label:<42} {detail}")

    gate("Beats BTC buy-and-hold (total return)",
         beats_btc,
         f"{_pct(rotation.total_return_pct)} vs {_pct(btc_bm.total_return_pct)}")
    gate("Beats EW hold-all (total return)",
         beats_ew,
         f"{_pct(rotation.total_return_pct)} vs {_pct(ew_bm.total_return_pct)}")
    gate("Sharpe ≥ 0.5",
         sharpe_ok,
         f"Sharpe = {_fmt(rotation.sharpe)}")

    passed = beats_btc and beats_ew and sharpe_ok

    print(f"\n{_SEP}")
    if passed:
        print("VALIDATED EDGE — rotation's downside protection justified itself")
        print("on out-of-sample bear data.")
    else:
        print("NO EDGE — rotation does not add value over simply holding the")
        print("basket, even with a bear leg.")
    print(f"{_SEP}")
    print("\nThe holdout is spent. This result is final.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
