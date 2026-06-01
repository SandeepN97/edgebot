"""Generate backtesting/reports/backtest_report.md from PeriodResult objects.

Sections:
    1. Summary table  — symbol × period: Sharpe, drawdown, win rate, trades
    2. Regime breakdown — performance by TRENDING / RANGING / VOLATILE / NEUTRAL
    3. Monthly PnL table — exit-month aggregation across all trades
    4. Top 5 wins and top 5 losses with entry reason
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtesting.engine.backtest_runner import PeriodResult

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).parent / "backtest_report.md"


def _fmt(v: float | None, fmt: str = ".2f", fallback: str = "N/A") -> str:
    return format(v, fmt) if v is not None else fallback


def _pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "N/A"


def _hours(avg_bars: float) -> str:
    return f"{avg_bars * 4:.0f}h" if avg_bars else "N/A"


def _all_trades(results: dict[str, tuple[PeriodResult, PeriodResult]]) -> list[dict]:
    rows = []
    for symbol, (train, test) in results.items():
        for period_result in (train, test):
            for t in period_result.trade_log:
                rows.append({**t, "symbol": symbol, "split": period_result.period})
    return rows


def generate_report(
    results: dict[str, tuple[PeriodResult, PeriodResult]],
    path: Path = REPORT_PATH,
) -> None:
    import pandas as pd

    lines: list[str] = []
    lines.append("# EdgeBot 4h Backtesting Report\n")
    lines.append(
        "Train: 2021-01-01 → 2023-12-31 | Test: 2024-01-01 → 2024-12-31  \n"
        "Capital: $100 | Commission: 0.1% | SL 1.5×ATR | TP 3.0×ATR\n"
    )

    # ------------------------------------------------------------------
    # 1. Summary table
    # ------------------------------------------------------------------
    lines.append("\n## 1. Summary — Train vs Test\n")
    lines.append(
        "| Symbol | Period | Sharpe | Max DD | Win Rate | Trades "
        "| Final Value | Avg Duration |\n"
        "|--------|--------|--------|--------|----------|--------|"
        "-------------|---------------|\n"
    )
    for symbol, (train, test) in results.items():
        for r in (train, test):
            lines.append(
                f"| {symbol} | {r.period} "
                f"| {_fmt(r.sharpe)} "
                f"| {_pct(r.max_dd_pct)} "
                f"| {r.win_rate:.1%} "
                f"| {r.total_trades} "
                f"| ${r.final_value:.2f} "
                f"| {_hours(r.avg_bars)} |\n"
            )

    # Overfitting flag
    lines.append("\n### Overfitting Check (Test Sharpe < 70% of Train Sharpe)\n\n")
    for symbol, (train, test) in results.items():
        if train.sharpe and test.sharpe is not None and train.sharpe > 0:
            ratio = test.sharpe / train.sharpe
            flag = "⚠ OVERFITTING" if ratio < 0.70 else "✓ OK"
            lines.append(
                f"- **{symbol}**: train={_fmt(train.sharpe)} test={_fmt(test.sharpe)}"
                f" ratio={ratio:.0%} — {flag}\n"
            )
        else:
            lines.append(f"- **{symbol}**: insufficient data for comparison\n")

    # ------------------------------------------------------------------
    # 2. Regime breakdown
    # ------------------------------------------------------------------
    lines.append("\n## 2. Regime Breakdown\n")
    all_rows = _all_trades(results)
    if all_rows:
        df = pd.DataFrame(all_rows)
        by_regime = (
            df.groupby("regime")
            .agg(
                trades=("pnl", "count"),
                win_rate=("pnl", lambda x: (x > 0).mean()),
                avg_pnl=("pnl", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        lines.append(
            "| Regime | Trades | Win Rate | Avg PnL | Total PnL |\n"
            "|--------|--------|----------|---------|----------|\n"
        )
        for _, row in by_regime.iterrows():
            lines.append(
                f"| {row['regime']} | {int(row['trades'])} "
                f"| {row['win_rate']:.1%} "
                f"| ${row['avg_pnl']:.4f} "
                f"| ${row['total_pnl']:.4f} |\n"
            )
    else:
        lines.append("_No trades recorded._\n")

    # ------------------------------------------------------------------
    # 3. Monthly PnL table
    # ------------------------------------------------------------------
    lines.append("\n## 3. Monthly PnL\n")
    if all_rows:
        df = pd.DataFrame(all_rows)
        df["exit_dt"] = pd.to_datetime(df["exit_dt"])
        df["month"] = df["exit_dt"].dt.to_period("M").astype(str)
        monthly = (
            df.groupby(["month", "symbol"])["pnl"]
            .sum()
            .unstack(fill_value=0.0)
            .sort_index()
        )
        cols = list(monthly.columns)
        lines.append("| Month | " + " | ".join(cols) + " | Total |\n")
        lines.append("|-------|" + "|".join(["-------"] * len(cols)) + "|-------|\n")
        for month, row in monthly.iterrows():
            total = row.sum()
            vals = " | ".join(f"${v:.4f}" for v in row)
            lines.append(f"| {month} | {vals} | ${total:.4f} |\n")
    else:
        lines.append("_No trades recorded._\n")

    # ------------------------------------------------------------------
    # 4. Top 5 wins and top 5 losses
    # ------------------------------------------------------------------
    lines.append("\n## 4. Top 5 Wins\n")
    if all_rows:
        df = pd.DataFrame(all_rows)
        df["exit_dt"] = pd.to_datetime(df["exit_dt"])
        df["entry_dt"] = pd.to_datetime(df["entry_dt"])
        top_wins = df.nlargest(5, "pnl")
        lines.append(
            "| # | Symbol | Direction | Entry | Exit | PnL | Regime | Reason |\n"
            "|---|--------|-----------|-------|------|-----|--------|--------|\n"
        )
        for i, (_, t) in enumerate(top_wins.iterrows(), 1):
            entry = t["entry_dt"].strftime("%Y-%m-%d %H:%M") if pd.notna(t["entry_dt"]) else "?"
            exit_ = t["exit_dt"].strftime("%Y-%m-%d %H:%M") if pd.notna(t["exit_dt"]) else "?"
            reason = str(t["reason"])[:60]
            lines.append(
                f"| {i} | {t['symbol']} | {t['direction']} | {entry} | {exit_} "
                f"| ${t['pnl']:.4f} | {t['regime']} | {reason} |\n"
            )

        lines.append("\n## 5. Top 5 Losses\n")
        top_losses = df.nsmallest(5, "pnl")
        lines.append(
            "| # | Symbol | Direction | Entry | Exit | PnL | Regime | Reason |\n"
            "|---|--------|-----------|-------|------|-----|--------|--------|\n"
        )
        for i, (_, t) in enumerate(top_losses.iterrows(), 1):
            entry = t["entry_dt"].strftime("%Y-%m-%d %H:%M") if pd.notna(t["entry_dt"]) else "?"
            exit_ = t["exit_dt"].strftime("%Y-%m-%d %H:%M") if pd.notna(t["exit_dt"]) else "?"
            reason = str(t["reason"])[:60]
            lines.append(
                f"| {i} | {t['symbol']} | {t['direction']} | {entry} | {exit_} "
                f"| ${t['pnl']:.4f} | {t['regime']} | {reason} |\n"
            )
    else:
        lines.append("_No trades recorded._\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)
    print(f"\nReport saved to {path}")