"""Orchestrate the cross-sectional relative-strength rotation pipeline.

Steps:
    1. Download 1d OHLCV for 10-coin universe (cached per symbol)
    2. Load price matrix from raw parquets
    3. Run rotation on tune window (earliest available → 2021-12-31)
    4. Compute BTC B&H and equal-weight-hold-all benchmarks
    5. Print comparison table
    6. Verdict: beats BOTH benchmarks AND Sharpe ≥ 0.5
               → "MOMENTUM SHOWS LIFE"
               otherwise → "NO EDGE OVER BUY-AND-HOLD"

Sealed 2022-2024 holdout is never touched.

Usage:
    python backtesting/run_rotation.py
    python backtesting/run_rotation.py --force   # re-download raw data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
_SEP = "=" * 68


def _fmt(v: float | None, fmt: str = "+.3f") -> str:
    return f"{v:{fmt}}" if v is not None else "  N/A "


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def _print_comparison(
    rotation,
    btc_bm,
    ew_bm,
) -> None:
    from backtesting.engine.rotation_backtest import LOOKBACK, REBALANCE_EVERY, TOP_N

    print(f"\n{_SEP}")
    print("ROTATION vs BENCHMARKS — tune window (2022-2024 sealed)")
    print(_SEP)
    print(f"  {'Metric':<22} {'Rotation':>14} {'BTC B&H':>14} {'EW Hold-All':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")

    def row(label: str, r_val: str, b_val: str, e_val: str) -> None:
        print(f"  {label:<22} {r_val:>14} {b_val:>14} {e_val:>14}")

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

    print(f"\n  Strategy details ({rotation.period})")
    print(f"    Lookback: {LOOKBACK}d | Hold top-{TOP_N} | Rebalance every {REBALANCE_EVERY}d | Cash filter ON")
    print(f"    Rebalances: {rotation.n_rebalances}  |  Trades: {rotation.n_trades}  |  Cash periods: {rotation.cash_periods}")


def _print_recent_holdings(weekly_log: list) -> None:
    if not weekly_log:
        return
    print(f"\n{_SEP}")
    print("LAST 10 REBALANCING DECISIONS (most recent first)")
    print(_SEP)
    print(f"  {'Date':<12} {'Action':<8} {'Held':<28} {'Scores'}")
    print(f"  {'-'*12} {'-'*8} {'-'*28} {'-'*30}")
    for entry in reversed(weekly_log[-10:]):
        held = ", ".join(s.split("/")[0] for s in entry["held"]) or "CASH"
        scores = "  ".join(
            f"{s.split('/')[0]}={v:+.2f}" for s, v in entry["scores"].items()
        ) or "—"
        print(f"  {str(entry['date'])[:10]:<12} {entry['action']:<8} {held:<28} {scores}")


def _verdict(rotation, btc_bm, ew_bm) -> bool:
    sharpe_floor = 0.5

    beats_btc = rotation.total_return_pct > btc_bm.total_return_pct
    beats_ew  = rotation.total_return_pct > ew_bm.total_return_pct
    sharpe_ok = rotation.sharpe is not None and rotation.sharpe >= sharpe_floor

    print(f"\n{_SEP}")
    print("VERDICT GATE")
    print(_SEP)

    def gate(label: str, passed: bool, detail: str) -> None:
        mark = "✓" if passed else "✗"
        print(f"  {mark}  {label:<38} {detail}")

    gate("Beats BTC buy-and-hold (total return)",
         beats_btc,
         f"{_pct(rotation.total_return_pct)} vs {_pct(btc_bm.total_return_pct)}")
    gate("Beats EW hold-all (total return)",
         beats_ew,
         f"{_pct(rotation.total_return_pct)} vs {_pct(ew_bm.total_return_pct)}")
    gate(f"Sharpe ≥ {sharpe_floor}",
         sharpe_ok,
         f"Sharpe = {_fmt(rotation.sharpe)}")

    passed = beats_btc and beats_ew and sharpe_ok
    print(f"\n{_SEP}")
    if passed:
        print("✓  MOMENTUM SHOWS LIFE")
        print("   Rotation clears all three gates. Unseal 2022-2024 when ready for")
        print("   the final holdout verdict.")
    else:
        print("✗  NO EDGE OVER BUY-AND-HOLD")
        print("   Cross-sectional momentum with these parameters does not beat passive")
        print("   holding. No further parameter iteration — the verdict stands.")
    print(_SEP)
    return passed


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run cross-sectional momentum rotation pipeline"
    )
    parser.add_argument("--force", action="store_true", help="Re-download raw data")
    args = parser.parse_args()

    print(_SEP)
    print("EdgeBot Cross-Sectional Momentum Rotation")
    print("Universe : BTC ETH BNB SOL XRP ADA DOGE AVAX LINK DOT")
    print("Signal   : 90-day risk-adjusted momentum (return / volatility)")
    print("Holding  : top-2 equal weight | rebalance every 5 days | cash filter ON")
    print("Tune     : earliest available → 2021-12-31  |  Sealed: 2022-2024")
    print(_SEP)

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    print("\n[1/3] Downloading 1d OHLCV for 10-coin universe (cached if available)...")
    from backtesting.data.downloader import download_rotation_all

    downloaded = download_rotation_all(force=args.force)
    print(f"  {len(downloaded)}/10 symbols downloaded/cached successfully")

    # ------------------------------------------------------------------
    # Step 2: Simulate
    # ------------------------------------------------------------------
    print("\n[2/3] Loading price matrix and running rotation simulation...")
    from backtesting.engine.rotation_backtest import load_price_matrix, run_rotation

    price_matrix = load_price_matrix()
    n_sym = len(price_matrix.columns)
    n_bars = len(price_matrix)
    print(f"  Price matrix: {n_sym} symbols × {n_bars} dates")
    print(f"  Date range  : {price_matrix.index[0].date()} → {price_matrix.index[-1].date()}")

    rotation = run_rotation(price_matrix)
    print(f"  Tune window : {rotation.period}")
    print(f"  Rebalances  : {rotation.n_rebalances}  |  Trades: {rotation.n_trades}")

    # ------------------------------------------------------------------
    # Step 3: Benchmarks
    # ------------------------------------------------------------------
    print("\n[3/3] Computing benchmarks...")
    from backtesting.engine.rotation_backtest import run_benchmarks

    btc_bm, ew_bm = run_benchmarks(price_matrix)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    _print_comparison(rotation, btc_bm, ew_bm)
    _print_recent_holdings(rotation.weekly_log)

    passed = _verdict(rotation, btc_bm, ew_bm)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
