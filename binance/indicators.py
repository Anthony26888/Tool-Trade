"""Deterministic Python technical-indicator engine (Phase 2).

Computes trading indicators from validated, *closed* Binance candles only.
Every calculation uses data up to and including the current candle — there is
no look-ahead and no dependence on any external indicator library. Values are
computed explicitly by Python so an LLM never has to calculate them.

Indicators:
    EMA20 / EMA50 / EMA200
    RSI14 (Wilder), MACD (12/26/9), ATR14 (Wilder), ADX14 (Wilder)
    Bollinger Bands (20, 2 standard deviations)
    Volume SMA (20) and Volume Ratio (volume / Volume SMA)

Rows before an indicator's warm-up period contain ``NaN``. A snapshot request
that needs the full set requires at least :data:`MIN_REQUIRED_CANDLES` (200)
closed candles so the EMA200 has warmed up.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import pairwise

import numpy as np
import pandas as pd

from .market_data import Candle, candles_to_dataframe

# Number of candles each indicator needs before its first valid value.
# Earlier rows are NaN in the output matrix.
WARM_UP_CANDLES: dict[str, int] = {
    "ema20": 20,
    "ema50": 50,
    "ema200": 200,
    "rsi14": 15,
    "macd": 34,
    "atr14": 15,
    "adx14": 28,
    "bollinger": 20,
    "volume_ratio": 20,
}

# Fewer candles than this cannot produce a fully warmed-up indicator set.
MIN_REQUIRED_CANDLES = max(WARM_UP_CANDLES.values())

_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_RSI_PERIOD = 14
_ATR_PERIOD = 14
_ADX_PERIOD = 14
_BB_PERIOD = 20
_BB_STD_DEV = 2.0
_VOLUME_PERIOD = 20


class IndicatorError(Exception):
    """Base class for indicator-engine errors."""


class EmptyIndicatorDataError(IndicatorError):
    """No closed candles were provided to compute indicators from."""


class FormingCandleError(IndicatorError):
    """An unfinished (still-forming) candle was passed to the engine."""


class InsufficientDataError(IndicatorError):
    """Fewer than :data:`MIN_REQUIRED_CANDLES` closed candles were provided."""


def _ema_array(values: np.ndarray, period: int, alpha: float | None = None) -> np.ndarray:
    """EMA of ``values`` seeded with the SMA of the first ``period`` valid values.

    Earlier positions, and positions before ``period`` valid inputs exist, are
    NaN. ``alpha`` defaults to the standard EMA smoothing ``2 / (period + 1)``;
    pass ``1 / period`` for Wilder-style smoothing.
    """
    n = len(values)
    out = np.full(n, np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < period:
        return out
    start = int(valid[period - 1])
    out[start] = float(np.mean(values[valid[:period]]))
    if alpha is None:
        alpha = 2.0 / (period + 1.0)
    for i in range(start + 1, n):
        if np.isnan(values[i]):
            break
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder_array(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder-style smoothing (``alpha = 1 / period``) seeded by an SMA."""
    return _ema_array(values, period, alpha=1.0 / period)


def _rsi_series(closes: np.ndarray, period: int = _RSI_PERIOD) -> np.ndarray:
    """Wilder RSI, first valid after ``period + 1`` closes.

    Flat price produces 50; no losses produces 100; no gains produces 0.
    """
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def _value(avg_gain: float, avg_loss: float) -> float:
        if avg_gain == 0.0 and avg_loss == 0.0:
            return 50.0
        if avg_loss == 0.0:
            return 100.0
        if avg_gain == 0.0:
            return 0.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _value(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _value(avg_gain, avg_loss)
    return out


def _macd_series(closes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, and histogram (12/26/9 convention)."""
    fast = _ema_array(closes, _MACD_FAST)
    slow = _ema_array(closes, _MACD_SLOW)
    macd_line = fast - slow
    signal = _ema_array(macd_line, _MACD_SIGNAL)
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _true_range_series(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(1, n):
        out[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return out


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder ATR, first valid from the 15th candle."""
    return _wilder_array(_true_range_series(high, low, close), _ATR_PERIOD)


def _adx_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = _ADX_PERIOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder ADX plus its +DI and -DI components (first valid at candle 28)."""
    n = len(high)
    tr = _true_range_series(high, low, close)
    up_raw = np.full(n, np.nan)
    dn_raw = np.full(n, np.nan)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        up_raw[i] = up if (up > dn and up > 0) else 0.0
        dn_raw[i] = dn if (dn > up and dn > 0) else 0.0

    tr_smooth = _wilder_array(tr, period)
    up_smooth = _wilder_array(up_raw, period)
    dn_smooth = _wilder_array(dn_raw, period)

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(tr_smooth[i]):
            continue
        if tr_smooth[i] == 0.0:
            plus_di[i], minus_di[i] = 0.0, 0.0
        else:
            plus_di[i] = 100.0 * up_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * dn_smooth[i] / tr_smooth[i]
        total = plus_di[i] + minus_di[i]
        dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / total if total else 0.0

    adx = _wilder_array(dx, period)
    return adx, plus_di, minus_di


def _rolling_sma(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values).rolling(period).mean().to_numpy()


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values).rolling(period).std(ddof=0).to_numpy()


def _bollinger_series(
    closes: np.ndarray, period: int = _BB_PERIOD, num_std: float = _BB_STD_DEV
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = _rolling_sma(closes, period)
    std = _rolling_std(closes, period)
    return mid, mid + num_std * std, mid - num_std * std


def _validate_closed_candles(candles: list[Candle]) -> None:
    if not isinstance(candles, list) or not candles:
        raise EmptyIndicatorDataError("at least one closed candle is required")
    for index, candle in enumerate(candles):
        if not candle.is_closed:
            raise FormingCandleError(
                f"candle at index {index} (timestamp {candle.timestamp}) is still forming"
            )
    for prev, curr in pairwise(candles):
        if curr.timestamp <= prev.timestamp:
            raise IndicatorError(
                "candles must be strictly ascending by timestamp, "
                f"got {prev.timestamp} then {curr.timestamp}"
            )


def compute_indicator_matrix(candles: list[Candle]) -> pd.DataFrame:
    """Compute the indicator matrix, one row per closed candle.

    Rows before an indicator's warm-up period contain NaN. Raises
    ``EmptyIndicatorDataError`` for no candles, ``FormingCandleError`` for any
    unfinished candle, and ``IndicatorError`` for unsorted timestamps.
    """
    _validate_closed_candles(candles)
    df = candles_to_dataframe(candles)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    df["ema20"] = _ema_array(close, 20)
    df["ema50"] = _ema_array(close, 50)
    df["ema200"] = _ema_array(close, 200)
    df["rsi14"] = _rsi_series(close)
    macd_line, macd_signal, macd_histogram = _macd_series(close)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_histogram
    df["atr14"] = _atr_series(high, low, close)
    adx, plus_di, minus_di = _adx_series(high, low, close)
    df["adx14"] = adx
    df["adx_plus_di"] = plus_di
    df["adx_minus_di"] = minus_di
    bb_mid, bb_upper, bb_lower = _bollinger_series(close)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["volume_sma20"] = _rolling_sma(volume, _VOLUME_PERIOD)
    # A zero-volume window yields inf, never a signal: normalize to NaN so
    # downstream snapshot validation treats it like any warm-up gap.
    df["volume_ratio"] = (df["volume"] / df["volume_sma20"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return df


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Indicator values for the most recent closed 1H candle."""

    timestamp: int
    last_close: float
    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    macd: float
    macd_signal: float
    macd_histogram: float
    atr14: float
    adx14: float
    adx_plus_di: float
    adx_minus_di: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    volume_sma20: float
    volume_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def latest_indicators(candles: list[Candle]) -> IndicatorSnapshot:
    """Snapshot of indicator values for the latest closed candle.

    Raises ``InsufficientDataError`` when fewer than
    :data:`MIN_REQUIRED_CANDLES` (200) closed candles are provided, because the
    EMA200 has not warmed up yet.
    """
    df = compute_indicator_matrix(candles)
    if len(df) < MIN_REQUIRED_CANDLES:
        raise InsufficientDataError(
            f"at least {MIN_REQUIRED_CANDLES} closed candles are required for "
            f"the full indicator set, got {len(df)}"
        )
    last = df.iloc[-1]
    if pd.isna(last["ema200"]):
        raise InsufficientDataError(
            "ema200 has not warmed up; provide more closed candles"
        )
    return IndicatorSnapshot(
        timestamp=int(last["timestamp"]),
        last_close=float(last["close"]),
        ema20=float(last["ema20"]),
        ema50=float(last["ema50"]),
        ema200=float(last["ema200"]),
        rsi14=float(last["rsi14"]),
        macd=float(last["macd"]),
        macd_signal=float(last["macd_signal"]),
        macd_histogram=float(last["macd_histogram"]),
        atr14=float(last["atr14"]),
        adx14=float(last["adx14"]),
        adx_plus_di=float(last["adx_plus_di"]),
        adx_minus_di=float(last["adx_minus_di"]),
        bb_mid=float(last["bb_mid"]),
        bb_upper=float(last["bb_upper"]),
        bb_lower=float(last["bb_lower"]),
        volume_sma20=float(last["volume_sma20"]),
        volume_ratio=float(last["volume_ratio"]),
    )
