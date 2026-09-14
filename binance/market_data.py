"""Binance USDT-M Futures public OHLCV (klines) market data.

Phase 1 scope: market data only. Produces validated :class:`Candle` objects
with an explicit ``is_closed`` flag so later phases never consume a
currently-forming candle as if it were complete.

The default symbol is ``BTCUSDT``; supported intervals are ``1m``, ``5m``,
``15m``, ``1h``, and ``4h``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .client import BinanceError, BinanceFuturesClient

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "BTCUSDT"
SUPPORTED_INTERVALS = ("1m", "5m", "15m", "1h", "4h")

KLINE_PATH = "/fapi/v1/klines"

# Binance returns 12 columns per kline: open_time, open, high, low, close,
# volume, close_time, quote_volume, count, taker_buy_volume,
# taker_buy_quote_volume, ignore.
_MIN_KLINE_FIELDS = 12
_SYMBOL_RE = re.compile(r"^[A-Z0-9]+$")


class InvalidSymbolError(ValueError):
    """The symbol is missing, malformed, or unsupported."""


class InvalidIntervalError(ValueError):
    """The interval is not in ``SUPPORTED_INTERVALS``."""


class MalformedKlineError(BinanceError):
    """A returned kline row failed structural or consistency validation."""


class EmptyKlineError(BinanceError):
    """The klines endpoint returned no usable candle data."""


@dataclass(frozen=True)
class Candle:
    """A single OHLCV candle from the Binance klines endpoint.

    ``timestamp`` is the kline open time in epoch milliseconds (UTC) and
    ``close_time`` is the kline close time. ``is_closed`` is ``False`` for the
    currently-forming candle, which downstream phases must never treat as a
    completed candle.
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    is_closed: bool


def validate_symbol(symbol: str) -> str:
    """Normalize and validate a Binance futures symbol (e.g. ``BTCUSDT``)."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise InvalidSymbolError(f"symbol must be a non-empty string, got {symbol!r}")
    normalized = symbol.strip().upper()
    if not _SYMBOL_RE.fullmatch(normalized):
        raise InvalidSymbolError(
            f"invalid symbol {symbol!r}; expected an alphanumeric pair such as BTCUSDT"
        )
    return normalized


def validate_interval(interval: str) -> str:
    """Validate an interval against ``SUPPORTED_INTERVALS``, normalizing case."""
    if not isinstance(interval, str) or not interval.strip():
        raise InvalidIntervalError(
            f"interval must be a non-empty string, got {interval!r}"
        )
    normalized = interval.strip().lower()
    if normalized not in SUPPORTED_INTERVALS:
        raise InvalidIntervalError(
            f"unsupported interval {interval!r}; choose one of {SUPPORTED_INTERVALS}"
        )
    return normalized


def is_candle_closed(close_time_ms: int, now_ms: int) -> bool:
    """Whether a candle with close time ``close_time_ms`` has closed by ``now_ms``.

    The comparison is strict: a candle is complete only once ``now_ms`` has
    moved past its close time, so an in-progress candle at the boundary is
    never treated as a completed candle.
    """
    return now_ms > close_time_ms


def _parse_kline(row: Any, now_ms: int) -> Candle:
    if not isinstance(row, list) or len(row) < _MIN_KLINE_FIELDS:
        raise MalformedKlineError(
            f"kline must be a list of at least {_MIN_KLINE_FIELDS} fields, got {row!r}"
        )
    try:
        timestamp = int(row[0])
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])
        close_time = int(row[6])
    except (TypeError, ValueError) as exc:
        raise MalformedKlineError(f"non-numeric field in kline row: {row!r}") from exc

    if timestamp < 0 or close_time < 0 or close_time < timestamp:
        raise MalformedKlineError(f"invalid timestamps in kline row: {row!r}")
    prices = (open_price, high, low, close)
    if not all(math.isfinite(p) and p >= 0 for p in prices):
        raise MalformedKlineError(f"non-finite or negative price in kline row: {row!r}")
    if not (math.isfinite(volume) and volume >= 0):
        raise MalformedKlineError(f"non-finite or negative volume in kline row: {row!r}")
    if high < max(open_price, close):
        raise MalformedKlineError(f"high below open/close in kline row: {row!r}")
    if low > min(open_price, close):
        raise MalformedKlineError(f"low above open/close in kline row: {row!r}")

    return Candle(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time=close_time,
        is_closed=is_candle_closed(close_time, now_ms),
    )


def parse_klines(raw: Any, now_ms: int) -> list[Candle]:
    """Parse and validate a raw klines payload into :class:`Candle` objects.

    Rejects malformed rows and out-of-order timestamps. An empty payload
    returns an empty list; the no-data policy is decided by the caller.
    """
    if not isinstance(raw, list):
        raise MalformedKlineError(
            f"expected a list of klines, got {type(raw).__name__}"
        )
    candles = [_parse_kline(row, now_ms) for row in raw]
    for prev, curr in zip(candles, candles[1:], strict=False):
        if curr.timestamp <= prev.timestamp:
            raise MalformedKlineError(
                "kline timestamps must be strictly ascending, "
                f"got {prev.timestamp} then {curr.timestamp}"
            )
    return candles


def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    """Render candles as a pandas DataFrame (one row per candle)."""
    return pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
            "close_time": [c.close_time for c in candles],
            "is_closed": [c.is_closed for c in candles],
        }
    )


def _current_time_ms() -> int:
    return int(time.time() * 1000)


class BinanceMarketData:
    """Public OHLCV accessor for Binance USDT-M Futures (klines endpoint)."""

    def __init__(self, client: BinanceFuturesClient | None = None) -> None:
        self.client = client or BinanceFuturesClient()

    def fetch_klines(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = "1h",
        limit: int = 500,
        end_time_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[Candle]:
        """Fetch OHLCV klines for ``symbol``/``interval``.

        ``limit`` caps the number of candles and ``end_time_ms`` bounds the
        fetch window. ``now_ms`` overrides the reference time used to flag the
        currently-forming candle (injectable for tests/determinism). Raises
        ``EmptyKlineError`` when the API returns no candles.
        """
        normalized_symbol = validate_symbol(symbol)
        normalized_interval = validate_interval(interval)
        params: dict[str, Any] = {
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "limit": limit,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        raw = self.client.get(KLINE_PATH, params=params)
        now = now_ms if now_ms is not None else _current_time_ms()
        candles = parse_klines(raw, now)
        if not candles:
            raise EmptyKlineError(
                f"no klines returned for {normalized_symbol} {normalized_interval}"
            )
        return candles

    def fetch_closed_klines(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = "1h",
        limit: int = 500,
        end_time_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[Candle]:
        """Fetch only CLOSED candles, excluding any currently-forming one."""
        candles = self.fetch_klines(symbol, interval, limit, end_time_ms, now_ms)
        closed = [c for c in candles if c.is_closed]
        if not closed:
            raise EmptyKlineError(
                f"no closed klines available for {validate_symbol(symbol)} {validate_interval(interval)}"
            )
        return closed
