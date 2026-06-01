"""Orchestrate the full 4h backtesting pipeline.

Steps:
    1. Download 4h OHLCV from Binance (cached — skips if parquet exists)
    2. Compute EMA / RSI / MACD / ATR / ADX / BB / volume_ratio features
    3. Classify each bar into a market regime (TRENDING / RANGING / VOLATILE / NEUTRAL)
    4. Run Backtrader on train (2021-2023) and test (2024) periods per symbol
    5. Print train-vs-test Sharpe comparison with overfitting flag
    6. Generate backtesting/reports/backtest_report.md
    7. Apply decision gate: warn and suggest adjustments if thresholds not met

Usage:
    python backtesting/run_all.py
    python backtesting/run_all.py --force   # re-download raw data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable regardless of invocation directory
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

_SEP = "=" * 64


def _fmt_sharpe(v: float | None) -> str:
    return f"{v:+.3f}" if v is not None else "  N/A "


def _fmt_dd(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "N/A"


def _print_results(results: dict) -> None:
    """Print either a tune dict (symbol→PeriodResult) or full dict (symbol→tuple)."""
    print(f"\n{_SEP}")
    # Detect whether we have a tune dict or full train/test dict
    first = next(iter(results.values()))
    is_tune = not isinstance(first, tuple)

    if is_tune:
        print("TUNE RESULTS — 2021-2022 ONLY  (2023-2024 sealed)")
    else:
        print("BACKTEST RESULTS — Train (2021-2023) vs Test (2024)")
    print(_SEP)

    header = (
        f"{'Symbol':<12} {'Period':<18} {'Sharpe':>8} {'MaxDD':>8} "
        f"{'WinRate':>8} {'Trades':>7} {'Final $':>9} {'AvgDur':>8}"
    )
    print(header)
    print("-" * len(header))

    if is_tune:
        for symbol, r in results.items():
            avg_h = f"{r.avg_bars * 4:.0f}h" if r.avg_bars else "N/A"
            print(
                f"{symbol:<12} {r.period:<18} "
                f"{_fmt_sharpe(r.sharpe):>8} "
                f"{_fmt_dd(r.max_dd_pct):>8} "
                f"{r.win_rate:>7.1%} "
                f"{r.total_trades:>7} "
                f"${r.final_value:>8.2f} "
                f"{avg_h:>8}"
            )
    else:
        for symbol, (train, test) in results.items():
            for r in (train, test):
                avg_h = f"{r.avg_bars * 4:.0f}h" if r.avg_bars else "N/A"
                print(
                    f"{symbol:<12} {r.period:<18} "
                    f"{_fmt_sharpe(r.sharpe):>8} "
                    f"{_fmt_dd(r.max_dd_pct):>8} "
                    f"{r.win_rate:>7.1%} "
                    f"{r.total_trades:>7} "
                    f"${r.final_value:>8.2f} "
                    f"{avg_h:>8}"
                )
            print()


def _check_tune_gate(results: dict) -> bool:
    """Gate check for the tune window (2021-2022). Sharpe floor 0.5, DD ceiling 30%."""
    print(_SEP)
    print("TUNE GATE — 2021-2022  (holdout 2023-2024 still sealed)")
    print(_SEP)

    first = next(iter(results.values()))
    is_tune = not isinstance(first, tuple)

    all_pass = True
    items = results.items() if is_tune else [
        (sym, r) for sym, (r, _) in results.items()
    ]

    for symbol, r in items:
        issues: list[str] = []

        if r.sharpe is not None and r.sharpe < 0.5:
            issues.append(f"Sharpe {r.sharpe:.3f} < 0.50 floor")
            all_pass = False
        elif r.sharpe is None:
            issues.append("Sharpe unavailable — too few trades")

        if r.max_dd_pct is not None and r.max_dd_pct > 30.0:
            issues.append(f"Max DD {r.max_dd_pct:.1f}% > 30% ceiling")
            all_pass = False

        status = "FAIL" if issues else "PASS"
        print(f"\n{symbol}  [{status}]")
        for issue in issues:
            print(f"  ⚠  {issue}")
        if not issues:
            print("  ✓  Tune gate cleared")

    if not all_pass:
        _suggest_adjustments(results)

    print(f"\n{_SEP}")
    if all_pass:
        print("✓  Tune gate PASSED — ready to unseal 2023-2024 for final verdict")
    else:
        print("✗  Tune gate FAILED — iterate before unsealing holdout")
    print(_SEP)
    return all_pass


def _check_decision_gate(results: dict) -> bool:
    """Full train/test gate — only called for the final holdout run."""
    print(_SEP)
    print("DECISION GATE")
    print(_SEP)

    all_pass = True

    for symbol, (train, test) in results.items():
        issues: list[str] = []

        if test.sharpe is not None and test.sharpe < 0.5:
            issues.append(f"Test Sharpe {test.sharpe:.3f} < 0.50 (floor)")
            all_pass = False
        elif test.sharpe is None:
            issues.append("Test Sharpe unavailable (too few trades?)")

        if test.max_dd_pct is not None and test.max_dd_pct > 30.0:
            issues.append(f"Test max DD {test.max_dd_pct:.1f}% > 30% ceiling")
            all_pass = False

        if train.sharpe and train.sharpe > 0 and test.sharpe is not None:
            ratio = test.sharpe / train.sharpe
            if ratio < 0.70:
                issues.append(
                    f"OVERFIT — test Sharpe ({test.sharpe:.3f}) is only "
                    f"{ratio:.0%} of train Sharpe ({train.sharpe:.3f}), need ≥70%"
                )
                all_pass = False

        status = "FAIL" if issues else "PASS"
        print(f"\n{symbol}  [{status}]")
        for issue in issues:
            print(f"  ⚠  {issue}")
        if not issues:
            print("  ✓  All gates cleared")

    if not all_pass:
        _suggest_adjustments(results)

    print(f"\n{_SEP}")
    if all_pass:
        print("✓  Decision gate PASSED — parameters cleared for Phase 2 live deployment")
    else:
        print("✗  Decision gate FAILED — iterate on parameters before proceeding")
    print(_SEP)
    return all_pass


def _suggest_adjustments(results: dict) -> None:
    print("\nSuggested parameter adjustments to try:\n")

    # Collect regime performance data to give targeted advice
    all_trades: list[dict] = []
    first = next(iter(results.values()))
    is_tune = not isinstance(first, tuple)
    for symbol, val in results.items():
        period_results = [val] if is_tune else list(val)
        for r in period_results:
            for t in r.trade_log:
                all_trades.append({**t, "symbol": symbol})

    if all_trades:
        import pandas as pd

        df = pd.DataFrame(all_trades)
        regime_wr = df.groupby("regime")["pnl"].apply(lambda x: (x > 0).mean())
        worst_regime = regime_wr.idxmin() if not regime_wr.empty else None
        if worst_regime:
            print(f"  • Worst-performing regime: {worst_regime} (win rate {regime_wr[worst_regime]:.0%})")
            if worst_regime == "RANGING":
                print("    → Add ADX > 20 as entry filter to skip ranging markets")
                print("    → Tighten RSI band to [45, 65] in ranging conditions")
            elif worst_regime == "VOLATILE":
                print("    → Widen SL multiplier to 2.0× ATR to survive volatile spikes")
                print("    → Require volume_ratio > 1.5× in volatile regimes")

    print("\n  General parameter suggestions:")
    print("  1. EMA periods: try 12/26 (less whipsaw) or 5/13 (more responsive)")
    print("  2. RSI band: widen to [35, 75] to catch more trend-continuation entries")
    print("  3. ATR SL multiplier: 2.0× → more room, fewer premature stops")
    print("  4. ATR TP multiplier: 2.5× → quicker profit-taking reduces retracement risk")
    print("  5. Add ADX > 20 entry filter → only trade when trend is confirmed")
    print("  6. Volume threshold: try 1.5× → higher bar reduces false breakouts")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run full EdgeBot 4h backtesting pipeline")
    parser.add_argument("--force", action="store_true", help="Re-download raw data even if cached")
    args = parser.parse_args()

    print(_SEP)
    print("EdgeBot 4h Backtesting Pipeline")
    print("Symbols: BTC/USDT  ETH/USDT  BNB/USDT")
    print("Train: 2021-01-01 → 2023-12-31  |  Test: 2024-01-01 → 2024-12-31")
    print(_SEP)

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    print("\n[1/4] Downloading 4h OHLCV from Binance (cached if available)...")
    from backtesting.data.downloader import SYMBOLS, download_all

    download_all(force=args.force)

    # ------------------------------------------------------------------
    # Step 2: Features
    # ------------------------------------------------------------------
    print("\n[2/4] Computing technical indicators...")
    from backtesting.features.feature_pipeline import compute_features_all

    compute_features_all()

    # ------------------------------------------------------------------
    # Step 3: Regime classification (updates feature parquet in-place)
    # ------------------------------------------------------------------
    print("\n[3/4] Classifying market regimes...")
    import pandas as pd

    from backtesting.engine.regime_detector import add_regime
    from backtesting.features.feature_pipeline import FEAT_DIR

    for symbol in SYMBOLS:
        filename = symbol.replace("/", "_")
        feat_path = FEAT_DIR / f"{filename}_4h.parquet"
        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            df = add_regime(df)
            df.to_parquet(feat_path)
            counts = df["regime"].value_counts().to_dict()
            print(f"  {symbol}: {counts}")

    # ------------------------------------------------------------------
    # Step 4: Backtest — tune window only (2021-2022)
    # ------------------------------------------------------------------
    print("\n[4/4] Running backtest on TUNE window 2021-2022 (holdout sealed)...")
    from backtesting.engine.backtest_runner import run_tune_all

    results = run_tune_all()

    if not results:
        print("No results — check that raw data downloaded successfully.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    _print_results(results)

    passed = _check_tune_gate(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()