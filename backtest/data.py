"""Historical market data for the backtest engine (Phase 12).

Basis data is Binance klines validated exactly like Phase 1 candles but frozen
in time: every candle must be closed and strictly ascending. The engine never
touches the network; a backtest consumes one ``HistoricalData`` object that
holds the closed decision-timeframe candles (1H by default) plus the closed
execution-timeframe candles (1m) used for entry / TP / SL confirmation and
same-candle ordering.

``HistoricalDataProvider`` is a small abstraction so the engine is agnostic
about where data comes from (memory, a JSON file, or a later phase's downloader).
A local JSON file is the built-in deterministic source for a reproducible run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from binance.market_data import Candle, validate_symbol

from .config import (
    DEFAULT_DECISION_TIMEFRAME,
    DEFAULT_EXECUTION_INTERVAL,
    BacktestDataError,
    interval_to_ms,
)


def _as_float(value: Any, field: str) -> float:
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):  # NaN handling
        raise BacktestDataError(f"{field} must be finite")
    return result


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise BacktestDataError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BacktestDataError(f"{field} must be an integer") from exc


def candle_to_dict(candle: Candle) -> dict[str, Any]:
    """Serialize a candle to a JSON-friendly record with ``str`` numbers."""
    return {
        "timestamp": candle.timestamp,
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": str(candle.volume),
        "close_time": candle.close_time,
        "is_closed": candle.is_closed,
    }


def candle_from_dict(record: Any) -> Candle:
    """Parse a JSON record into a ``Candle``, rejecting malformed shapes."""
    if not isinstance(record, dict):
        raise BacktestDataError(f"candle record must be an object, got {type(record).__name__}")
    required = ("timestamp", "open", "high", "low", "close", "volume", "close_time")
    missing = [name for name in required if name not in record]
    if missing:
        raise BacktestDataError(f"candle record missing fields: {', '.join(missing)}")
    timestamp = _as_int(record["timestamp"], "timestamp")
    close_time = _as_int(record["close_time"], "close_time")
    if close_time < timestamp:
        raise BacktestDataError("candle close_time must be >= timestamp")
    try:
        high = _as_float(record["high"], "high")
        low = _as_float(record["low"], "low")
        open_price = _as_float(record["open"], "open")
        close = _as_float(record["close"], "close")
        volume = _as_float(record["volume"], "volume")
    except (TypeError, ValueError) as exc:
        raise BacktestDataError(f"non-numeric price/volume in candle: {record!r}") from exc
    if high < max(open_price, close) or low > min(open_price, close):
        raise BacktestDataError(f"candle OHLC inconsistent: {record!r}")
    is_closed = bool(record.get("is_closed", True))
    return Candle(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time=close_time,
        is_closed=is_closed,
    )


def candles_to_stored(candles: list[Candle]) -> list[dict[str, Any]]:
    """Serialize a candle list to JSON-friendly records."""
    return [candle_to_dict(candle) for candle in candles]


def stored_to_candles(records: Any, field: str) -> list[Candle]:
    """Parse and validate a JSON candle list, enforcing the closed/ordered rules."""
    if not isinstance(records, list) or not records:
        raise BacktestDataError(f"{field} must be a non-empty list of candles")
    candles = [candle_from_dict(record) for record in records]
    for candle in candles:
        if not candle.is_closed:
            raise BacktestDataError(
                f"{field} candle at timestamp {candle.timestamp} must be closed; "
                "forming candles are never used in a backtest"
            )
    for prev, curr in zip(candles, candles[1:], strict=False):
        if curr.timestamp <= prev.timestamp:
            raise BacktestDataError(
                f"{field} timestamps must be strictly ascending, got "
                f"{prev.timestamp} then {curr.timestamp}"
            )
    if len({c.timestamp for c in candles}) != len(candles):
        raise BacktestDataError(f"{field} contains duplicate timestamps")
    return candles


@dataclass(frozen=True)
class HistoricalData:
    """Validated, immutable market data for a single symbolic run.

    ``hour_candles`` are the closed decision-timeframe candles; ``minute_candles``
    are the closed execution-timeframe candles. ``finer_candles`` is an optional
    finer-resolution series (e.g. seconds) that a provider MAY supply to resolve
    same-candle TP/SL ordering; the Phase 12 engine keeps the conservative
    production behavior and does not use it.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = DEFAULT_DECISION_TIMEFRAME
    execution_interval: str = DEFAULT_EXECUTION_INTERVAL
    hour_candles: tuple[Candle, ...] = field(default_factory=tuple)
    minute_candles: tuple[Candle, ...] = field(default_factory=tuple)
    finer_candles: tuple[Candle, ...] | None = None

    def __post_init__(self) -> None:
        validate_historical_data(self)


def validate_historical_data(data: HistoricalData) -> None:
    """Validate a :class:`HistoricalData` object or raise :class:`BacktestDataError`."""
    if not isinstance(data, HistoricalData):
        raise BacktestDataError("data must be a HistoricalData")
    validate_symbol(data.symbol)
    interval_to_ms(data.timeframe)
    interval_to_ms(data.execution_interval)
    if not data.hour_candles:
        raise BacktestDataError("historical data requires the decision-timeframe candles")
    if not data.minute_candles:
        raise BacktestDataError("historical data requires the execution-timeframe candles")
    for candle in data.hour_candles:
        if not isinstance(candle, Candle) or not candle.is_closed:
            raise BacktestDataError("hour candles must be closed Candle instances")
    for candle in data.minute_candles:
        if not isinstance(candle, Candle) or not candle.is_closed:
            raise BacktestDataError("minute candles must be closed Candle instances")
    _validate_ascending(data.hour_candles, "hour_candles")
    _validate_ascending(data.minute_candles, "minute_candles")
    if data.finer_candles is not None:
        for candle in data.finer_candles:
            if not isinstance(candle, Candle) or not candle.is_closed:
                raise BacktestDataError("finer_candles must be closed Candle instances")
        _validate_ascending(data.finer_candles, "finer_candles")
    if not data.hour_candles[0].is_closed or not data.minute_candles[0].is_closed:
        raise BacktestDataError("the first candle of every series must be closed")


def _validate_ascending(candles: tuple[Candle, ...], field: str) -> None:
    for prev, curr in zip(candles, candles[1:], strict=False):
        if curr.timestamp <= prev.timestamp:
            raise BacktestDataError(
                f"{field} timestamps must be strictly ascending, got "
                f"{prev.timestamp} then {curr.timestamp}"
            )
    if len({c.timestamp for c in candles}) != len(candles):
        raise BacktestDataError(f"{field} contains duplicate timestamps")


class HistoricalDataProvider(Protocol):
    """Something that produces a validated :class:`HistoricalData`."""

    def get_historical_data(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = DEFAULT_DECISION_TIMEFRAME,
        execution_interval: str = DEFAULT_EXECUTION_INTERVAL,
    ) -> HistoricalData: ...


@dataclass(frozen=True)
class MemoryHistoricalDataProvider:
    """A provider that hands back a pre-built :class:`HistoricalData`."""

    data: HistoricalData

    def get_historical_data(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = DEFAULT_DECISION_TIMEFRAME,
        execution_interval: str = DEFAULT_EXECUTION_INTERVAL,
    ) -> HistoricalData:
        return self.data


def load_historical_data_json(path: str | os.PathLike[str]) -> HistoricalData:
    """Load and validate a backtest data file (see ``save_historical_data_json``)."""
    filename = str(path)
    try:
        with open(filename, encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BacktestDataError(f"invalid JSON in backtest data file: {exc}") from exc
    except OSError as exc:
        raise BacktestDataError(f"could not read backtest data file: {exc}") from exc
    if not isinstance(payload, dict):
        raise BacktestDataError("backtest data file must contain a JSON object")
    symbol = str(payload.get("symbol", "BTCUSDT"))
    timeframe = str(payload.get("timeframe", DEFAULT_DECISION_TIMEFRAME))
    execution_interval = str(payload.get("execution_interval", DEFAULT_EXECUTION_INTERVAL))
    validate_symbol(symbol)
    interval_to_ms(timeframe)
    interval_to_ms(execution_interval)
    if "hour_candles" not in payload or "minute_candles" not in payload:
        raise BacktestDataError(
            "backtest data file must contain 'hour_candles' and 'minute_candles'"
        )
    hour_candles = stored_to_candles(payload["hour_candles"], "hour_candles")
    minute_candles = stored_to_candles(payload["minute_candles"], "minute_candles")
    finer = (
        stored_to_candles(payload["finer_candles"], "finer_candles")
        if "finer_candles" in payload
        else None
    )
    return HistoricalData(
        symbol=symbol,
        timeframe=timeframe,
        execution_interval=execution_interval,
        hour_candles=tuple(hour_candles),
        minute_candles=tuple(minute_candles),
        finer_candles=tuple(finer) if finer is not None else None,
    )


def save_historical_data_json(
    path: str | os.PathLike[str], data: HistoricalData, *, indent: int = 2
) -> None:
    """Write a validated :class:`HistoricalData` to a deterministic JSON file."""
    validate_historical_data(data)
    payload = {
        "symbol": data.symbol,
        "timeframe": data.timeframe,
        "execution_interval": data.execution_interval,
        "hour_candles": candles_to_stored(list(data.hour_candles)),
        "minute_candles": candles_to_stored(list(data.minute_candles)),
    }
    if data.finer_candles is not None:
        payload["finer_candles"] = candles_to_stored(list(data.finer_candles))
    try:
        with open(str(path), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
            handle.write("\n")
    except OSError as exc:
        raise BacktestDataError(f"could not write backtest data file: {exc}") from exc


class JsonFileHistoricalDataProvider:
    """A provider that loads a local JSON data file once (deterministic replay)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)

    def get_historical_data(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = DEFAULT_DECISION_TIMEFRAME,
        execution_interval: str = DEFAULT_EXECUTION_INTERVAL,
    ) -> HistoricalData:
        data = load_historical_data_json(self.path)
        if data.symbol != validate_symbol(symbol):
            raise BacktestDataError(
                f"data file symbol {data.symbol} does not match requested {symbol}"
            )
        return data


__all__ = [
    "HistoricalData",
    "HistoricalDataProvider",
    "JsonFileHistoricalDataProvider",
    "MemoryHistoricalDataProvider",
    "candle_from_dict",
    "candle_to_dict",
    "candles_to_stored",
    "load_historical_data_json",
    "save_historical_data_json",
    "stored_to_candles",
    "validate_historical_data",
]
