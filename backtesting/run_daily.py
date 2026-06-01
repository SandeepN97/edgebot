"""Run the EMA-trending-only strategy on daily (1d) candles.

Tune window: earliest available history through 2021-12-31.
Sealed holdout: 2022-2024 — NOT run, NOT printed, NOT referenced.

Verdict rule (hard stop, no further iteration):
  ≥ 2/3 symbols clear Sharpe ≥ 0.5 with ≥ 20 trades
      → "READY FOR SEALED VERDICT"
  otherwise
      → "TREND-FOLLOWING LACKS EDGE ACROSS TIMEFRAMES"

Usage:
    python backtesting/run_daily.py
    python backtesting/run_daily.py --force   # re-download raw data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

_SEP = "=" * 64


def _fmt_sharpe(v: float | None) -> str:
    return f"{v:+.3f}" if v is not None else "  N/A "


def _fmt_dd(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "N/A"


def _print_results(results: dict) -> None:
    print(f"\n{_SEP}")
    print("DAILY EMA TUNE RESULTS — TUNE WINDOW ~2017-2021  (2022-2024 sealed)")
    print(_SEP)

    header = (
        f"{'Symbol':<12} {'Period':<34} {'Sharpe':>8} {'MaxDD':>8} "
        f"{'WinRate':>8} {'Trades':>7} {'Final $':>9} {'AvgDur':>8}"
    )
    print(header)
    print("-" * len(header))

    for symbol, r in results.items():
        avg_d = f"{r.avg_bars:.0f}d" if r.avg_bars else "N/A"
        print(
            f"{symbol:<12} {r.period:<34} "
            f"{_fmt_sharpe(r.sharpe):>8} "
            f"{_fmt_dd(r.max_dd_pct):>8} "
            f"{r.win_rate:>7.1%} "
            f"{r.total_trades:>7} "
            f"${r.final_value:>8.2f} "
            f"{avg_d:>8}"
        )


def _check_and_verdict(results: dict) -> bool:
    print(f"\n{_SEP}")
    print("DAILY TUNE GATE — Sharpe ≥ 0.5, ≥ 20 trades  (2022-2024 still sealed)")
    print(_SEP)

    min_trades = 20
    sharpe_floor = 0.5
    passing = 0

    for symbol, r in results.items():
        issues: list[str] = []

        if r.total_trades < min_trades:
            issues.append(f"Only {r.total_trades} trades — need ≥{min_trades}")

        if r.sharpe is None:
            issues.append("Sharpe unavailable — too few trades")
        elif r.sharpe < sharpe_floor:
            issues.append(f"Sharpe {r.sharpe:.3f} < {sharpe_floor:.1f} floor")

        if r.max_dd_pct is not None and r.max_dd_pct > 30.0:
            issues.append(f"Max DD {r.max_dd_pct:.1f}% > 30% ceiling")

        cleared = not issues
        if cleared:
            passing += 1
        status = "PASS" if cleared else "FAIL"

        print(f"\n{symbol}  [{status}]")
        for issue in issues:
            print(f"  ⚠  {issue}")
        if cleared:
            print("  ✓  Daily tune gate cleared")

    needed = 2
    print(f"\n{_SEP}")
    print(f"Symbols clearing gate: {passing}/{len(results)}  (need ≥{needed}/{len(results)})")

    if passing >= needed:
        print("\n✓  READY FOR SEALED VERDICT")
        print("   ≥2/3 symbols show Sharpe ≥0.5 with ≥20 trades on daily bars.")
        print("   Unseal 2022-2024 when ready for the final holdout run.")
        print(_SEP)
        return True
    else:
        print("\n✗  TREND-FOLLOWING LACKS EDGE ACROSS TIMEFRAMES")
        print("   EMA crossover failed to clear Sharpe ≥0.5 on both 4h and daily bars.")
        print("   This is the hard stop — no further parameter iteration recommended.")
        print(_SEP)
        return False


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run EMA daily backtesting pipeline")
    parser.add_argument("--force", action="store_true", help="Re-download raw data even if cached")
    args = parser.parse_args()

    print(_SEP)
    print("EdgeBot Daily EMA Backtesting Pipeline")
    print("Symbols: BTC/USDT  ETH/USDT  BNB/USDT")
    print("Tune: ~2017-01-01 → 2021-12-31  |  Sealed holdout: 2022-2024")
    print("Sharpe factor: 252 bars/year  |  Params: EMA 9/21, RSI 40-70, ADX≥25, SL 2×ATR, TP 3×ATR")
    print(_SEP)

    # ------------------------------------------------------------------
    # Step 1: Download daily OHLCV
    # ------------------------------------------------------------------
    print("\n[1/4] Downloading 1d OHLCV (cached if available)...")
    from backtesting.data.downloader import SYMBOLS, download_daily_all

    download_daily_all(force=args.force)

    # ------------------------------------------------------------------
    # Step 2: Compute features on daily bars
    # ------------------------------------------------------------------
    print("\n[2/4] Computing technical indicators on daily bars...")
    from backtesting.features.feature_pipeline import compute_features_all

    compute_features_all(timeframe="1d")

    # ------------------------------------------------------------------
    # Step 3: Classify regimes
    # ------------------------------------------------------------------
    print("\n[3/4] Classifying market regimes...")
    import pandas as pd

    from backtesting.engine.regime_detector import add_regime
    from backtesting.features.feature_pipeline import FEAT_DIR

    for symbol in SYMBOLS:
        filename = symbol.replace("/", "_")
        feat_path = FEAT_DIR / f"{filename}_1d.parquet"
        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            df = add_regime(df)
            df.to_parquet(feat_path)
            counts = df["regime"].value_counts().to_dict()
            print(f"  {symbol}: {counts}")

    # ------------------------------------------------------------------
    # Step 4: Run daily EMA tune (sealed window intact)
    # ------------------------------------------------------------------
    print("\n[4/4] Running daily EMA-trending-only backtest on tune window...")
    from backtesting.engine.backtest_runner import run_daily_tune_all

    results = run_daily_tune_all()

    if not results:
        print("No results — check that raw daily data downloaded successfully.")
        sys.exit(1)

    _print_results(results)

    passed = _check_and_verdict(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
