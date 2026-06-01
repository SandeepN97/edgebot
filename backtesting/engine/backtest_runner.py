"""Backtrader engine for 4h EMA crossover strategy with train/test split.

Train period: 2021-01-01 → 2023-12-31
Test  period: 2024-01-01 → 2024-12-31

Sharpe is annualised assuming 24/7 crypto: 365 days × 6 bars/day = 2190 bars/year.
Commission: 0.1% per trade (Binance taker).
Starting capital: $100.
Position sizing: 2% of equity at risk per trade (MAX_POSITION_SIZE_PCT).
SL/TP: 1.5× ATR / 3.0× ATR — identical to live EmaCrossoverStrategy constants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
FEAT_DIR = Path(__file__).parent.parent / "data" / "features"

TRAIN_START = "2021-01-01"
TRAIN_END = "2024-01-01"   # exclusive upper bound
TEST_START = "2024-01-01"
TEST_END = "2025-01-01"    # exclusive upper bound

# Tune-only window (2021-2022): used for iterative parameter search.
# 2023 and 2024 are the sealed holdout — never run until final verdict.
TUNE_START = "2021-01-01"
TUNE_END = "2023-01-01"    # exclusive: 2021 + 2022 only

STARTING_CASH = 100.0
COMMISSION = 0.001         # 0.1 % Binance taker
BARS_PER_YEAR = 2190       # 365 × 6  (24/7 crypto, 4h bars)


@dataclass
class PeriodResult:
    symbol: str
    period: str             # "TRAIN" | "TEST"
    sharpe: float | None
    max_dd_pct: float | None
    win_rate: float
    total_trades: int
    final_value: float
    avg_bars: float         # avg trade duration in 4h bars
    trade_log: list[dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------
# Backtrader strategy definition
# ------------------------------------------------------------------

def _make_strategy_class() -> type:
    """Return the Backtrader Strategy class (imported lazily to avoid BT at import time)."""
    import backtrader as bt  # type: ignore[import]

    class EmaCrossoverBT(bt.Strategy):
        params = dict(
            ema_fast=9,
            ema_slow=21,
            rsi_period=14,
            rsi_low=40.0,
            rsi_high=70.0,
            vol_period=20,
            vol_mult=1.2,
            atr_period=14,
            atr_sl_mult=2.0,
            atr_tp_mult=3.0,
            adx_period=14,
            adx_min=25.0,       # confirmed trend only (matches live _ADX_MIN)
            risk_pct=0.02,
            features_df=None,   # passed at cerebro.addstrategy() time
        )

        def __init__(self) -> None:
            self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
            self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
            self.rsi = bt.ind.RSI(period=self.p.rsi_period, safediv=True)
            self.atr = bt.ind.ATR(period=self.p.atr_period)
            self.adx = bt.ind.AverageDirectionalMovementIndex(period=self.p.adx_period)
            self.vol_ma = bt.ind.SMA(self.data.volume, period=self.p.vol_period)
            self.crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)
            self.order: bt.Order | None = None
            self.sl_price: float = 0.0
            self.tp_price: float = 0.0
            self.pending_entry: dict[str, Any] = {}
            self.trade_log: list[dict[str, Any]] = []

        def _regime_at(self, bar_idx: int) -> str:
            df = self.p.features_df
            if df is None or not (0 <= bar_idx < len(df)):
                return "UNKNOWN"
            return str(df.iloc[bar_idx].get("regime", "UNKNOWN"))

        def notify_order(self, order: bt.Order) -> None:
            if order.status in (order.Canceled, order.Margin, order.Rejected):
                logger.debug("Order %s status %s", order.ref, order.getstatusname())
            self.order = None

        def notify_trade(self, trade: bt.Trade) -> None:
            if not trade.isclosed:
                return
            import backtrader as bt2  # noqa: F811  (local re-import for bt.num2date)
            entry = self.pending_entry
            self.trade_log.append(
                {
                    "entry_dt": bt2.num2date(trade.dtopen),
                    "exit_dt": bt2.num2date(trade.dtclose),
                    "direction": entry.get("direction", "?"),
                    "pnl": trade.pnlcomm,
                    "reason": entry.get("reason", ""),
                    "regime": entry.get("regime", "UNKNOWN"),
                    "bars": trade.barlen,
                }
            )
            self.pending_entry = {}

        def next(self) -> None:
            if self.order:
                return

            # ---- Manage open position (SL / TP / reverse-cross exit) ----
            if self.position.size > 0:               # long
                if self.data.low[0] <= self.sl_price or self.data.high[0] >= self.tp_price:
                    self.order = self.close()
                    return
                if self.crossover[0] < 0:
                    self.order = self.close()
                return

            if self.position.size < 0:               # short
                if self.data.high[0] >= self.sl_price or self.data.low[0] <= self.tp_price:
                    self.order = self.close()
                    return
                if self.crossover[0] > 0:
                    self.order = self.close()
                return

            # ---- Entry logic (mirrors live EmaCrossoverStrategy.next()) ----
            rsi_ok = self.p.rsi_low <= self.rsi[0] <= self.p.rsi_high
            vol_ok = self.vol_ma[0] > 0 and (
                self.data.volume[0] > self.vol_ma[0] * self.p.vol_mult
            )
            adx_ok = self.adx[0] >= self.p.adx_min
            atr = self.atr[0]
            if atr <= 0 or not (rsi_ok and vol_ok and adx_ok):
                return

            bar_idx = len(self.data) - 1
            vol_ratio = (
                self.data.volume[0] / self.vol_ma[0] if self.vol_ma[0] > 0 else 0.0
            )

            if self.crossover[0] > 0:
                entry_price = self.data.close[0]
                sl = entry_price - atr * self.p.atr_sl_mult
                tp = entry_price + atr * self.p.atr_tp_mult
                risk_per_unit = entry_price - sl
                if risk_per_unit <= 0:
                    return
                size = min(
                    (self.broker.cash * self.p.risk_pct) / risk_per_unit,
                    self.broker.cash * 0.95 / entry_price,
                )
                if size <= 0:
                    return
                self.sl_price, self.tp_price = sl, tp
                self.pending_entry = {
                    "direction": "LONG",
                    "reason": (
                        f"EMA{self.p.ema_fast} crossed EMA{self.p.ema_slow}; "
                        f"RSI={self.rsi[0]:.1f}; ADX={self.adx[0]:.1f}; vol_ratio={vol_ratio:.2f}"
                    ),
                    "regime": self._regime_at(bar_idx),
                }
                self.order = self.buy(size=size)

            elif self.crossover[0] < 0:
                entry_price = self.data.close[0]
                sl = entry_price + atr * self.p.atr_sl_mult
                tp = entry_price - atr * self.p.atr_tp_mult
                risk_per_unit = sl - entry_price
                if risk_per_unit <= 0:
                    return
                size = min(
                    (self.broker.cash * self.p.risk_pct) / risk_per_unit,
                    self.broker.cash * 0.95 / entry_price,
                )
                if size <= 0:
                    return
                self.sl_price, self.tp_price = sl, tp
                self.pending_entry = {
                    "direction": "SHORT",
                    "reason": (
                        f"EMA{self.p.ema_fast} crossed below EMA{self.p.ema_slow}; "
                        f"RSI={self.rsi[0]:.1f}; vol_ratio={vol_ratio:.2f}"
                    ),
                    "regime": self._regime_at(bar_idx),
                }
                self.order = self.sell(size=size)

    return EmaCrossoverBT


# ------------------------------------------------------------------
# Single-period runner
# ------------------------------------------------------------------

def _run_period(
    features: pd.DataFrame,
    symbol: str,
    period_label: str,
    start: str,
    end: str,
    starting_cash: float = STARTING_CASH,
) -> PeriodResult:
    import backtrader as bt  # type: ignore[import]

    mask = (features.index >= start) & (features.index < end)
    period_df = features[mask].copy()

    if len(period_df) < 100:
        logger.warning(
            "%s %s: only %d bars — skipping (need ≥100)", symbol, period_label, len(period_df)
        )
        return PeriodResult(symbol, period_label, None, None, 0.0, 0, starting_cash, 0.0)

    # Backtrader needs a tz-naive DatetimeIndex
    feed_df = period_df[["open", "high", "low", "close", "volume"]].copy()
    if feed_df.index.tz is not None:
        feed_df.index = feed_df.index.tz_localize(None)

    EmaCrossoverBT = _make_strategy_class()

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=feed_df,
            timeframe=bt.TimeFrame.Minutes,
            compression=240,
        )
    )
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addstrategy(EmaCrossoverBT, features_df=period_df)

    # Annualise Sharpe for 24/7 crypto 4h bars: sqrt(2190) factor
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Minutes,
        compression=240,
        factor=BARS_PER_YEAR,
        annualize=True,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run(maxcpus=1)
    strat = results[0]

    sharpe_raw = strat.analyzers.sharpe.get_analysis()
    dd_raw = strat.analyzers.drawdown.get_analysis()
    trades_raw = strat.analyzers.trades.get_analysis()

    sharpe = sharpe_raw.get("sharperatio")
    max_dd = dd_raw.get("max", {}).get("drawdown")
    total = trades_raw.get("total", {}).get("closed", 0)
    won = trades_raw.get("won", {}).get("total", 0)
    win_rate = won / total if total else 0.0
    final_value = cerebro.broker.getvalue()
    avg_bars = (
        sum(t["bars"] for t in strat.trade_log) / len(strat.trade_log)
        if strat.trade_log
        else 0.0
    )

    return PeriodResult(
        symbol=symbol,
        period=period_label,
        sharpe=sharpe,
        max_dd_pct=max_dd,
        win_rate=win_rate,
        total_trades=total,
        final_value=final_value,
        avg_bars=avg_bars,
        trade_log=strat.trade_log,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run_backtest_symbol(symbol: str) -> tuple[PeriodResult, PeriodResult]:
    filename = symbol.replace("/", "_")
    feat_path = FEAT_DIR / f"{filename}_4h.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path} — run feature pipeline first")

    features = pd.read_parquet(feat_path)
    logger.info("Backtesting %s — %d bars total", symbol, len(features))

    train = _run_period(features, symbol, "TRAIN", TRAIN_START, TRAIN_END)
    test = _run_period(features, symbol, "TEST", TEST_START, TEST_END)
    return train, test


def run_backtest_all() -> dict[str, tuple[PeriodResult, PeriodResult]]:
    results: dict[str, tuple[PeriodResult, PeriodResult]] = {}
    for symbol in SYMBOLS:
        try:
            results[symbol] = run_backtest_symbol(symbol)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
    return results


def run_tune_symbol(symbol: str) -> PeriodResult:
    """Run only the 2021-2022 tune window. 2023-2024 remain sealed."""
    filename = symbol.replace("/", "_")
    feat_path = FEAT_DIR / f"{filename}_4h.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path} — run feature pipeline first")

    features = pd.read_parquet(feat_path)
    return _run_period(features, symbol, "TUNE 2021-2022", TUNE_START, TUNE_END)


def run_tune_all() -> dict[str, PeriodResult]:
    """Run all symbols on the tune window only. Does not touch the holdout."""
    results: dict[str, PeriodResult] = {}
    for symbol in SYMBOLS:
        try:
            results[symbol] = run_tune_symbol(symbol)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
    return results


# ------------------------------------------------------------------
# Combined router mode: EMA (TRENDING) + Mean-Reversion (RANGING)
# ------------------------------------------------------------------

def _make_combined_strategy_class() -> type:
    """Return a Backtrader Strategy that routes EMA/MR per regime each bar."""
    import backtrader as bt  # type: ignore[import]

    class CombinedRouterBT(bt.Strategy):
        params = dict(
            # EMA crossover
            ema_fast=9,
            ema_slow=21,
            vol_period=20,
            vol_mult=1.2,
            atr_tp_mult=3.0,
            # ADX regime thresholds (mirrors StrategyRouter + regime_detector)
            adx_trending=25.0,
            adx_ranging=20.0,
            # Shared
            rsi_period=14,
            atr_period=14,
            atr_sl_mult=2.0,
            risk_pct=0.02,
            # Mean-reversion
            bb_period=20,
            bb_devfactor=2.0,
            rsi_mr_long=35.0,
            rsi_mr_short=65.0,
            # EMA RSI gate
            rsi_ema_low=40.0,
            rsi_ema_high=70.0,
            features_df=None,
        )

        def __init__(self) -> None:
            # EMA crossover indicators
            self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
            self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
            self.crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)
            self.vol_ma = bt.ind.SMA(self.data.volume, period=self.p.vol_period)
            # Shared
            self.rsi = bt.ind.RSI(period=self.p.rsi_period, safediv=True)
            self.atr = bt.ind.ATR(period=self.p.atr_period)
            self.adx = bt.ind.AverageDirectionalMovementIndex(period=self.p.atr_period)
            # Mean-reversion
            self.bb = bt.ind.BollingerBands(
                period=self.p.bb_period, devfactor=self.p.bb_devfactor
            )
            self.order: bt.Order | None = None
            self.sl_price: float = 0.0
            self.tp_price: float = 0.0        # fixed TP for EMA trades
            self.active_strategy: str = ""    # "ema" | "mr"
            self.pending_entry: dict[str, Any] = {}
            self.trade_log: list[dict[str, Any]] = []

        def _regime(self) -> str:
            adx = self.adx[0]
            if adx >= self.p.adx_trending:
                return "TRENDING"
            if adx < self.p.adx_ranging:
                return "RANGING"
            return "NEUTRAL"

        def _regime_at(self, bar_idx: int) -> str:
            df = self.p.features_df
            if df is None or not (0 <= bar_idx < len(df)):
                return "UNKNOWN"
            return str(df.iloc[bar_idx].get("regime", "UNKNOWN"))

        def notify_order(self, order: bt.Order) -> None:
            if order.status in (order.Canceled, order.Margin, order.Rejected):
                logger.debug("Order %s %s", order.ref, order.getstatusname())
            self.order = None

        def notify_trade(self, trade: bt.Trade) -> None:
            if not trade.isclosed:
                return
            import backtrader as bt2  # noqa: F811
            entry = self.pending_entry
            self.trade_log.append({
                "entry_dt": bt2.num2date(trade.dtopen),
                "exit_dt": bt2.num2date(trade.dtclose),
                "direction": entry.get("direction", "?"),
                "pnl": trade.pnlcomm,
                "reason": entry.get("reason", ""),
                "regime": entry.get("regime", "UNKNOWN"),
                "strategy": entry.get("strategy", "?"),
                "bars": trade.barlen,
            })
            self.pending_entry = {}

        def next(self) -> None:
            if self.order:
                return

            atr = self.atr[0]

            # ---- Manage open position ----
            if self.position.size > 0:   # long
                if self.data.low[0] <= self.sl_price:
                    self.order = self.close()
                    return
                if self.active_strategy == "ema":
                    if self.data.high[0] >= self.tp_price or self.crossover[0] < 0:
                        self.order = self.close()
                elif self.active_strategy == "mr":
                    if self.data.close[0] >= self.bb.mid[0]:
                        self.order = self.close()
                return

            if self.position.size < 0:   # short
                if self.data.high[0] >= self.sl_price:
                    self.order = self.close()
                    return
                if self.active_strategy == "ema":
                    if self.data.low[0] <= self.tp_price or self.crossover[0] > 0:
                        self.order = self.close()
                elif self.active_strategy == "mr":
                    if self.data.close[0] <= self.bb.mid[0]:
                        self.order = self.close()
                return

            # ---- Entry ----
            if atr <= 0:
                return

            regime = self._regime()
            bar_idx = len(self.data) - 1

            if regime == "TRENDING":
                self._try_ema_entry(atr, bar_idx)
            # RANGING / NEUTRAL / VOLATILE → sit out (mirrors router)

        def _try_ema_entry(self, atr: float, bar_idx: int) -> None:
            rsi_ok = self.p.rsi_ema_low <= self.rsi[0] <= self.p.rsi_ema_high
            vol_ok = self.vol_ma[0] > 0 and (
                self.data.volume[0] > self.vol_ma[0] * self.p.vol_mult
            )
            if not (rsi_ok and vol_ok):
                return

            vol_ratio = self.data.volume[0] / self.vol_ma[0] if self.vol_ma[0] > 0 else 0.0
            regime_label = self._regime_at(bar_idx)

            if self.crossover[0] > 0:
                entry = self.data.close[0]
                sl = entry - atr * self.p.atr_sl_mult
                tp = entry + atr * self.p.atr_tp_mult
                risk = entry - sl
                if risk <= 0:
                    return
                size = min(
                    (self.broker.cash * self.p.risk_pct) / risk,
                    self.broker.cash * 0.95 / entry,
                )
                if size <= 0:
                    return
                self.sl_price, self.tp_price = sl, tp
                self.active_strategy = "ema"
                self.pending_entry = {
                    "direction": "LONG",
                    "strategy": "ema",
                    "regime": regime_label,
                    "reason": (
                        f"EMA{self.p.ema_fast} crossed EMA{self.p.ema_slow}; "
                        f"RSI={self.rsi[0]:.1f}; ADX={self.adx[0]:.1f}; "
                        f"vol_ratio={vol_ratio:.2f}"
                    ),
                }
                self.order = self.buy(size=size)

            elif self.crossover[0] < 0:
                entry = self.data.close[0]
                sl = entry + atr * self.p.atr_sl_mult
                tp = entry - atr * self.p.atr_tp_mult
                risk = sl - entry
                if risk <= 0:
                    return
                size = min(
                    (self.broker.cash * self.p.risk_pct) / risk,
                    self.broker.cash * 0.95 / entry,
                )
                if size <= 0:
                    return
                self.sl_price, self.tp_price = sl, tp
                self.active_strategy = "ema"
                self.pending_entry = {
                    "direction": "SHORT",
                    "strategy": "ema",
                    "regime": regime_label,
                    "reason": (
                        f"EMA{self.p.ema_fast} crossed below EMA{self.p.ema_slow}; "
                        f"RSI={self.rsi[0]:.1f}; ADX={self.adx[0]:.1f}; "
                        f"vol_ratio={vol_ratio:.2f}"
                    ),
                }
                self.order = self.sell(size=size)

        def _try_mr_entry(self, atr: float, bar_idx: int) -> None:
            close = self.data.close[0]
            regime_label = self._regime_at(bar_idx)

            if close < self.bb.bot[0] and self.rsi[0] < self.p.rsi_mr_long:
                entry = close
                sl = entry - atr * self.p.atr_sl_mult
                tp = self.bb.mid[0]
                risk = entry - sl
                if risk <= 0 or tp <= entry:
                    return
                size = min(
                    (self.broker.cash * self.p.risk_pct) / risk,
                    self.broker.cash * 0.95 / entry,
                )
                if size <= 0:
                    return
                self.sl_price = sl
                self.active_strategy = "mr"
                self.pending_entry = {
                    "direction": "LONG",
                    "strategy": "mr",
                    "regime": regime_label,
                    "reason": (
                        f"Close {close:.2f} < BB_lower {self.bb.bot[0]:.2f}; "
                        f"RSI={self.rsi[0]:.1f} < {self.p.rsi_mr_long}"
                    ),
                }
                self.order = self.buy(size=size)

            elif close > self.bb.top[0] and self.rsi[0] > self.p.rsi_mr_short:
                entry = close
                sl = entry + atr * self.p.atr_sl_mult
                tp = self.bb.mid[0]
                risk = sl - entry
                if risk <= 0 or tp >= entry:
                    return
                size = min(
                    (self.broker.cash * self.p.risk_pct) / risk,
                    self.broker.cash * 0.95 / entry,
                )
                if size <= 0:
                    return
                self.sl_price = sl
                self.active_strategy = "mr"
                self.pending_entry = {
                    "direction": "SHORT",
                    "strategy": "mr",
                    "regime": regime_label,
                    "reason": (
                        f"Close {close:.2f} > BB_upper {self.bb.top[0]:.2f}; "
                        f"RSI={self.rsi[0]:.1f} > {self.p.rsi_mr_short}"
                    ),
                }
                self.order = self.sell(size=size)

    return CombinedRouterBT


def _run_combined_period(
    features: pd.DataFrame,
    symbol: str,
    period_label: str,
    start: str,
    end: str,
    starting_cash: float = STARTING_CASH,
) -> PeriodResult:
    import backtrader as bt  # type: ignore[import]

    mask = (features.index >= start) & (features.index < end)
    period_df = features[mask].copy()

    if len(period_df) < 100:
        logger.warning(
            "%s %s: only %d bars — skipping", symbol, period_label, len(period_df)
        )
        return PeriodResult(symbol, period_label, None, None, 0.0, 0, starting_cash, 0.0)

    feed_df = period_df[["open", "high", "low", "close", "volume"]].copy()
    if feed_df.index.tz is not None:
        feed_df.index = feed_df.index.tz_localize(None)

    CombinedRouterBT = _make_combined_strategy_class()

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=feed_df,
            timeframe=bt.TimeFrame.Minutes,
            compression=240,
        )
    )
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addstrategy(CombinedRouterBT, features_df=period_df)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Minutes,
        compression=240,
        factor=BARS_PER_YEAR,
        annualize=True,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run(maxcpus=1)
    strat = results[0]

    sharpe_raw = strat.analyzers.sharpe.get_analysis()
    dd_raw = strat.analyzers.drawdown.get_analysis()
    trades_raw = strat.analyzers.trades.get_analysis()

    sharpe = sharpe_raw.get("sharperatio")
    max_dd = dd_raw.get("max", {}).get("drawdown")
    total = trades_raw.get("total", {}).get("closed", 0)
    won = trades_raw.get("won", {}).get("total", 0)
    win_rate = won / total if total else 0.0
    final_value = cerebro.broker.getvalue()
    avg_bars = (
        sum(t["bars"] for t in strat.trade_log) / len(strat.trade_log)
        if strat.trade_log else 0.0
    )

    return PeriodResult(
        symbol=symbol,
        period=period_label,
        sharpe=sharpe,
        max_dd_pct=max_dd,
        win_rate=win_rate,
        total_trades=total,
        final_value=final_value,
        avg_bars=avg_bars,
        trade_log=strat.trade_log,
    )


def run_combined_tune_symbol(symbol: str) -> PeriodResult:
    """Run the combined EMA+MR router on the 2021-2022 tune window."""
    filename = symbol.replace("/", "_")
    feat_path = FEAT_DIR / f"{filename}_4h.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")

    features = pd.read_parquet(feat_path)
    return _run_combined_period(features, symbol, "COMBINED TUNE 2021-2022", TUNE_START, TUNE_END)


def run_combined_tune_all() -> dict[str, PeriodResult]:
    """Run combined router for all symbols on the tune window."""
    results: dict[str, PeriodResult] = {}
    for symbol in SYMBOLS:
        try:
            results[symbol] = run_combined_tune_symbol(symbol)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
    return results