"""Phase 12: historical data handling, validation, JSON round-trips, providers."""

from __future__ import annotations

import json

import pytest

from backtest.config import BacktestDataError
from backtest.data import (
    HistoricalData,
    JsonFileHistoricalDataProvider,
    MemoryHistoricalDataProvider,
    candle_from_dict,
    candle_to_dict,
    load_historical_data_json,
    save_historical_data_json,
    stored_to_candles,
)
from binance.market_data import Candle
from tests.backtest_helpers import (
    MINUTE_MS,
    covering_minutes,
    dataset,
    make_hours,
    minute,
)

HOUR_MS = 3_600_000


def test_hour_candles_must_be_closed():
    hours = make_hours(5)
    forming = Candle(
        hours[-1].timestamp + HOUR_MS,
        hours[-1].close,
        hours[-1].close + 5,
        hours[-1].close - 5,
        hours[-1].close + 2,
        1.0,
        hours[-1].timestamp + 2 * HOUR_MS - 1,
        False,
    )
    with pytest.raises(BacktestDataError):
        HistoricalData(hour_candles=tuple(hours) + (forming,), minute_candles=tuple(covering_minutes(hours)))


def test_requires_nonempty_series():
    with pytest.raises(BacktestDataError):
        HistoricalData(hour_candles=(), minute_candles=())
    with pytest.raises(BacktestDataError):
        HistoricalData(hour_candles=tuple(make_hours(3)), minute_candles=())


def test_rejects_forming_minute_candles():
    hours = make_hours(3)
    minutes = covering_minutes(hours)
    from binance.market_data import Candle

    ts = minutes[-1].timestamp + MINUTE_MS
    bad = Candle(ts, 60000.0, 60020.0, 59980.0, 60000.0, 1.0, ts + MINUTE_MS - 1, is_closed=False)
    with pytest.raises(BacktestDataError):
        HistoricalData(
            hour_candles=tuple(hours),
            minute_candles=tuple(minutes) + (bad,),
        )


def test_rejects_unsorted_and_duplicate_timestamps():
    hours = make_hours(3)
    minutes = covering_minutes(hours)
    unsorted = [minutes[1], minutes[0]] + minutes[2:]
    with pytest.raises(BacktestDataError):
        HistoricalData(hour_candles=tuple(hours), minute_candles=tuple(unsorted))
    duplicated = [minutes[0], minute(minutes[0].timestamp, 1.0, 1.0, 1.0, 1.0)] + minutes[1:]
    with pytest.raises(BacktestDataError):
        HistoricalData(hour_candles=tuple(hours), minute_candles=tuple(duplicated))


def test_candle_dict_round_trip():
    source = Candle(1728000000000, 60000.0, 60060.0, 59950.0, 60055.0, 100.5, 1728000036000 - 1, True)
    restored = candle_from_dict(candle_to_dict(source))
    assert restored == source


def test_candle_from_dict_accepts_string_numbers():
    payload = {
        "timestamp": 1728000000000,
        "open": "60000.0",
        "high": "60060.5",
        "low": "59950.0",
        "close": "60055.0",
        "volume": "100.5",
        "close_time": 1728000035999,
        "is_closed": True,
    }
    candle = candle_from_dict(payload)
    assert candle.high == 60060.5


@pytest.mark.parametrize(
    "payload",
    [
        {"timestamp": 1, "open": "1", "high": "1", "low": "1"},  # missing fields
        {"timestamp": 1, "open": "1", "high": "1", "low": "1", "close": "x", "volume": "1", "close_time": 2},
        {"timestamp": 2, "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1", "close_time": 1},
        {"timestamp": 1, "open": "2", "high": "1", "low": "1", "close": "2", "volume": "1", "close_time": 2},
        {"timestamp": 1, "open": "1", "high": "1", "low": "2", "close": "1", "volume": "1", "close_time": 2},
        [1, 2, 3],
    ],
)
def test_candle_from_dict_rejects_bad_records(payload):
    with pytest.raises(BacktestDataError):
        candle_from_dict(payload)


def test_json_round_trip(tmp_path):
    data = dataset(make_hours(10))
    path = tmp_path / "data.json"
    save_historical_data_json(path, data)
    restored = load_historical_data_json(path)
    assert restored.symbol == data.symbol
    assert restored.timeframe == data.timeframe
    assert restored.execution_interval == data.execution_interval
    assert restored.hour_candles == data.hour_candles
    assert restored.minute_candles == data.minute_candles
    assert (path.read_text(encoding="utf-8").strip().endswith("}"))


def test_json_rejects_malformed_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BacktestDataError):
        load_historical_data_json(path)


def test_json_rejects_missing_series(tmp_path):
    path = tmp_path / "missing.json"
    path.write_text('{"symbol": "BTCUSDT"}', encoding="utf-8")
    with pytest.raises(BacktestDataError):
        load_historical_data_json(path)


def test_json_stored_candles_validation(tmp_path):
    data = dataset(make_hours(5))
    payload = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "execution_interval": "1m",
        "hour_candles": [
            {
                "timestamp": c.timestamp,
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": str(c.volume),
                "close_time": c.close_time,
                "is_closed": False,  # forming candles are rejected
            }
            for c in data.hour_candles
        ],
        "minute_candles": [
            {
                "timestamp": c.timestamp,
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": str(c.volume),
                "close_time": c.close_time,
                "is_closed": True,
            }
            for c in data.minute_candles
        ],
    }
    path = tmp_path / "forming.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BacktestDataError):
        load_historical_data_json(path)


def test_stored_to_candles_validation():
    records = {
        "timestamp": 1,
        "open": "1",
        "high": "1",
        "low": "1",
        "close": "1",
        "volume": "1",
        "close_time": 2,
        "is_closed": True,
    }
    assert stored_to_candles([records], "field")
    with pytest.raises(BacktestDataError):
        stored_to_candles([], "field")
    with pytest.raises(BacktestDataError):
        stored_to_candles("nope", "field")


def test_memory_provider_returns_same():
    data = dataset(make_hours(3))
    provider = MemoryHistoricalDataProvider(data)
    assert provider.get_historical_data() == data


def test_json_file_provider(tmp_path):
    data = dataset(make_hours(3))
    path = tmp_path / "data.json"
    save_historical_data_json(path, data)
    provider = JsonFileHistoricalDataProvider(path)
    assert provider.get_historical_data().hour_candles == data.hour_candles


def test_json_file_provider_symbol_mismatch(tmp_path):
    data = dataset(make_hours(3))
    path = tmp_path / "data.json"
    save_historical_data_json(path, data)
    provider = JsonFileHistoricalDataProvider(path)
    with pytest.raises(BacktestDataError):
        provider.get_historical_data(symbol="ETHUSDT")


def test_invalid_close_time():
    with pytest.raises(BacktestDataError):
        candle_from_dict(
            {
                "timestamp": 10,
                "open": "1",
                "high": "1",
                "low": "1",
                "close": "1",
                "volume": "1",
                "close_time": 5,
            }
        )
