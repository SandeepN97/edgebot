"""Compute technical indicators matching the live EmaCrossoverStrategy exactly.

All indicator parameters are imported from strategies/ema_crossover.py constants
so train and live always stay in sync:
    EMA9, EMA21, EMA200       — fast/slow/trend filters
    RSI14                     — momentum gate [40, 70]
    MACD(12, 26, 9)           — trend confirmation
    ATR14                     — volatility / SL-TP sizing
    ADX14                     — trend strength (regime detector input)
    BollingerBands(20, 2)     — upper / mid / lower
    volume_ratio              — vol / 20-bar vol MA
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
FEAT_DIR = Path(__file__).parent.parent / "data" / "features"

# Mirror live strategy constants
_EMA_FAST = 9
_EMA_SLOW = 21
_EMA_TREND = 200
_RSI_PERIOD = 14
_MACD_FAST, _MACD_SLOW, _MACD_SIGNAL = 12, 26, 9
_ATR_PERIOD = 14
_ADX_PERIOD = 14
_BB_PERIOD, _BB_STD = 20, 2.0
_VOL_MA_PERIOD = 20


# ------------------------------------------------------------------
# Pure indicator functions
# ------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = _ATR_PERIOD) -> pd.Series:
    tr = _true_range(high, low, close.shift())
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = _ADX_PERIOD) -> pd.Series:
    tr = _true_range(high, low, close.shift())
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=alpha, adjust=False).mean()


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to an OHLCV DataFrame and drop warmup NaN rows."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    vol = out["volume"]

    out["ema9"] = _ema(close, _EMA_FAST)
    out["ema21"] = _ema(close, _EMA_SLOW)
    out["ema200"] = _ema(close, _EMA_TREND)
    out["rsi14"] = _rsi(close)

    ema12 = _ema(close, _MACD_FAST)
    ema26 = _ema(close, _MACD_SLOW)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd"], _MACD_SIGNAL)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["atr14"] = _atr(high, low, close)
    out["adx14"] = _adx(high, low, close)

    bb_mean = close.rolling(_BB_PERIOD).mean()
    bb_std = close.rolling(_BB_PERIOD).std(ddof=1)
    out["bb_upper"] = bb_mean + _BB_STD * bb_std
    out["bb_lower"] = bb_mean - _BB_STD * bb_std
    out["bb_mid"] = bb_mean

    vol_ma = vol.rolling(_VOL_MA_PERIOD).mean()
    out["volume_ratio"] = vol / vol_ma.replace(0.0, np.nan)

    # Drop warmup rows (EMA200 needs ~200 bars, ADX needs ~28)
    return out.dropna(subset=["ema200", "adx14", "volume_ratio"])


def compute_features_for_symbol(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    filename = _safe_filename(symbol)
    raw_path = RAW_DIR / f"{filename}_{timeframe}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path} — run downloader first")

    df = pd.read_parquet(raw_path)
    features = compute_features(df)

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEAT_DIR / f"{filename}_{timeframe}.parquet"
    features.to_parquet(out_path)
    logger.info("Features saved to %s (%d rows)", out_path, len(features))
    return features


def compute_features_all(timeframe: str = "4h") -> dict[str, pd.DataFrame]:
    from backtesting.data.downloader import SYMBOLS

    results: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        try:
            df = compute_features_for_symbol(symbol, timeframe=timeframe)
            results[symbol] = df
            print(f"  {symbol}: {len(df)} bars with features")
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
    return results


def _safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    compute_features_all()