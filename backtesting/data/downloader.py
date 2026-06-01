"""Download 4h OHLCV candles from Binance via CCXT REST with pagination.

Binance caps fetch_ohlcv at 1000 candles per call; this module paginates
automatically and caches results as parquet so repeated runs are instant.

Usage:
    python backtesting/data/downloader.py           # skip if cached
    python backtesting/data/downloader.py --force   # always re-download
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import ccxt  # type: ignore[import]
import pandas as pd

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
TIMEFRAME = "4h"
DAILY_START_DATE = "2017-01-01"   # fetch longest available history for daily
DAILY_TIMEFRAME = "1d"

# 10-coin universe for cross-sectional rotation
ROTATION_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT",
    "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
]
CANDLES_PER_REQUEST = 1000
RAW_DIR = Path(__file__).parent / "raw"


def _safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_")


def _to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def download_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    timeframe: str = TIMEFRAME,
    force: bool = False,
) -> pd.DataFrame:
    out_path = RAW_DIR / f"{_safe_filename(symbol)}_{timeframe}.parquet"
    if out_path.exists() and not force:
        logger.info("[cache] %s — %s already exists, skipping", symbol, out_path.name)
        df = pd.read_parquet(out_path)
        print(f"  {symbol}: {len(df)} candles loaded from cache")
        return df

    start_ms = _to_ms(start)
    end_ms = _to_ms(end) + 24 * 60 * 60 * 1000  # include full last day

    all_candles: list[list] = []
    since = start_ms
    logger.info("Downloading %s %s candles %s → %s", symbol, timeframe, start, end)

    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=CANDLES_PER_REQUEST)
        if not batch:
            break
        batch = [c for c in batch if c[0] < end_ms]
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1  # one ms past the last candle
        print(f"\r  {symbol}: {len(all_candles)} candles...", end="", flush=True)
        time.sleep(max(exchange.rateLimit / 1000, 0.1))
        if len(batch) < CANDLES_PER_REQUEST:
            break  # reached end of available data

    print()
    if not all_candles:
        raise RuntimeError(f"No candles returned for {symbol}")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("Saved %d candles → %s", len(df), out_path)
    return df


def _make_exchange() -> ccxt.Exchange:
    """Return the first reachable exchange that has all required symbols and 4h timeframe."""
    candidates = ["binanceus", "okx", "binance"]
    for name in candidates:
        try:
            ex: ccxt.Exchange = getattr(ccxt, name)({"enableRateLimit": True})
            markets = ex.load_markets()
            if all(s in markets for s in SYMBOLS) and "4h" in ex.timeframes:
                logger.info("Using exchange: %s", name)
                print(f"  Exchange: {name}")
                return ex
        except Exception as exc:
            logger.debug("%s unavailable: %s", name, exc)
    raise RuntimeError(
        f"No reachable exchange with BTC/USDT, ETH/USDT, BNB/USDT and 4h timeframe. "
        f"Tried: {candidates}"
    )


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    exchange = _make_exchange()
    results: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        try:
            results[symbol] = download_symbol(exchange, symbol, force=force)
        except Exception as exc:
            logger.error("Failed to download %s: %s", symbol, exc)
    return results


def download_daily_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for all symbols, fetching the longest history available."""
    exchange = _make_exchange()
    results: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        try:
            results[symbol] = download_symbol(
                exchange,
                symbol,
                start=DAILY_START_DATE,
                end=END_DATE,
                timeframe=DAILY_TIMEFRAME,
                force=force,
            )
        except Exception as exc:
            logger.error("Failed to download daily %s: %s", symbol, exc)
    return results


def download_rotation_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download daily 1d OHLCV for the 10-coin rotation universe.

    Skips symbols not available on the exchange rather than raising, so the
    rotation engine works with whatever subset downloaded successfully.
    """
    exchange = _make_exchange()
    results: dict[str, pd.DataFrame] = {}
    for symbol in ROTATION_SYMBOLS:
        try:
            results[symbol] = download_symbol(
                exchange,
                symbol,
                start=DAILY_START_DATE,
                end=END_DATE,
                timeframe=DAILY_TIMEFRAME,
                force=force,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download 4h OHLCV from Binance via CCXT")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    download_all(force=args.force)


if __name__ == "__main__":
    main()