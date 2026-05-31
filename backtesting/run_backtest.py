"""Backtrader backtest runner for EdgeBot strategies.

Downloads 2022-2024 OHLCV data via yfinance, runs the EMA crossover strategy,
and prints Sharpe ratio, maximum drawdown, and win rate to stdout.

Usage::

    python backtesting/run_backtest.py --symbol BTC-USD --cash 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EdgeBot Backtrader backtest runner")
    parser.add_argument("--symbol", default="BTC-USD", help="yfinance symbol (default BTC-USD)")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Starting capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission per trade")
    parser.add_argument("--interval", default="1d", help="yfinance data interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        import backtrader as bt  # type: ignore[import]
        import yfinance as yf  # type: ignore[import]
        import pandas as pd  # type: ignore[import]
    except ImportError as exc:
        logger.error("Missing dependency: %s — run: pip install backtrader yfinance pandas", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Download data
    # ------------------------------------------------------------------
    logger.info("Downloading %s from %s to %s", args.symbol, args.start, args.end)
    raw = yf.download(
        args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        logger.error("No data returned for %s", args.symbol)
        sys.exit(1)

    # yfinance >= 0.2 returns a MultiIndex (Price, Ticker) — flatten to simple names
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    # Backtrader PandasData expects these exact column names
    raw.rename(columns={"adj close": "close"}, inplace=True, errors="ignore")
    raw.index = pd.to_datetime(raw.index)
    # Strip timezone so Backtrader doesn't choke on tz-aware index
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    logger.info("Downloaded %d bars (columns: %s)", len(raw), list(raw.columns))

    # ------------------------------------------------------------------
    # Backtrader strategy
    # ------------------------------------------------------------------

    class EmaCrossoverBT(bt.Strategy):
        """EMA9/EMA21 crossover strategy for Backtrader."""

        params = dict(
            ema_fast=9,
            ema_slow=21,
            rsi_period=14,
            rsi_low=40.0,
            rsi_high=70.0,
            vol_period=20,
            vol_mult=1.2,
            atr_period=14,
            atr_sl_mult=1.5,
            atr_tp_mult=3.0,
        )

        def __init__(self) -> None:
            self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
            self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
            self.rsi = bt.ind.RSI(period=self.p.rsi_period)
            self.atr = bt.ind.ATR(period=self.p.atr_period)
            self.vol_ma = bt.ind.SMA(self.data.volume, period=self.p.vol_period)
            self.crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)
            self.order = None
            self.trade_log: list = []

        def notify_order(self, order) -> None:
            if order.status in (order.Completed,):
                direction = "BUY" if order.isbuy() else "SELL"
                logger.debug(
                    "%s EXEC @%.2f size=%.4f",
                    direction,
                    order.executed.price,
                    order.executed.size,
                )
            self.order = None

        def notify_trade(self, trade) -> None:
            if trade.isclosed:
                self.trade_log.append(
                    {
                        "pnl": trade.pnlcomm,
                        "won": trade.pnlcomm > 0,
                    }
                )

        def next(self) -> None:
            if self.order:
                return

            rsi_ok = self.p.rsi_low <= self.rsi[0] <= self.p.rsi_high
            vol_ok = self.data.volume[0] > self.vol_ma[0] * self.p.vol_mult

            if not self.position:
                if self.crossover[0] > 0 and rsi_ok and vol_ok:
                    sl = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
                    tp = self.data.close[0] + self.atr[0] * self.p.atr_tp_mult
                    size = (self.broker.cash * 0.02) / self.data.close[0]
                    self.order = self.buy(size=size)
                elif self.crossover[0] < 0 and rsi_ok and vol_ok:
                    size = (self.broker.cash * 0.02) / self.data.close[0]
                    self.order = self.sell(size=size)
            else:
                if self.position.size > 0 and self.crossover[0] < 0:
                    self.order = self.close()
                elif self.position.size < 0 and self.crossover[0] > 0:
                    self.order = self.close()

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EmaCrossoverBT)

    data_feed = bt.feeds.PandasData(dataname=raw)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    logger.info("Running backtest …")
    results = cerebro.run()
    strat = results[0]

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    sharpe_raw = strat.analyzers.sharpe.get_analysis()
    drawdown_raw = strat.analyzers.drawdown.get_analysis()
    trades_raw = strat.analyzers.trades.get_analysis()

    sharpe = sharpe_raw.get("sharperatio", None)
    max_dd = drawdown_raw.get("max", {}).get("drawdown", None)

    total_trades = trades_raw.get("total", {}).get("closed", 0)
    won = trades_raw.get("won", {}).get("total", 0)
    win_rate = won / total_trades if total_trades else 0.0

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - args.cash) / args.cash

    print("\n" + "=" * 50)
    print(f"EdgeBot Backtest Results — {args.symbol}")
    print(f"Period  : {args.start} → {args.end}")
    print(f"Capital : ${args.cash:,.0f} → ${final_value:,.2f}")
    print(f"Return  : {total_return:+.2%}")
    print(f"Sharpe  : {sharpe:.4f}" if sharpe else "Sharpe  : N/A")
    print(f"Max DD  : {max_dd:.2f}%" if max_dd is not None else "Max DD  : N/A")
    print(f"Trades  : {total_trades} total, {won} wins ({win_rate:.1%} win rate)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()