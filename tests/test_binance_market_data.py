"""Phase 1 unit tests: Binance USDT-M Futures public market data.

All HTTP calls are mocked — no real Binance requests are made.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from binance.client import (
    BinanceError,
    BinanceFuturesClient,
    BinanceHTTPError,
    BinanceRateLimitError,
    BinanceServerError,
)
from binance.market_data import (
    SUPPORTED_INTERVALS,
    BinanceMarketData,
    EmptyKlineError,
    InvalidIntervalError,
    InvalidSymbolError,
    MalformedKlineError,
    candles_to_dataframe,
    is_candle_closed,
    parse_klines,
    validate_interval,
    validate_symbol,
)

_HOUR_MS = 3_600_000


def _kline(
    timestamp,
    close_time=None,
    open_px="1000.0",
    high="1010.0",
    low="990.0",
    close="1005.0",
    volume="12.5",
):
    """A minimal valid 12-field Binance kline row."""
    return [
        timestamp,
        open_px,
        high,
        low,
        close,
        volume,
        close_time if close_time is not None else timestamp + _HOUR_MS,
        0,
        0,
        0,
        0,
        0,
    ]


def _sample_rows():
    return [
        _kline(1_600_000_000_000),
        _kline(1_600_003_600_000),
        _kline(1_600_007_200_000),
    ]


# c3 closes at 1_600_010_800_000; past that all candles are closed.
_SAMPLE_NOW_PAST_ALL = 1_600_010_800_000 + 60_000
# Between c2 close (1_600_007_200_000) and c3 close: only the third is forming.
_SAMPLE_NOW_THIRD_FORMING = 1_600_009_900_000


def _http(status_code=200, payload=None):
    text = "ok" if status_code == 200 else f"http {status_code}"
    return SimpleNamespace(status_code=status_code, text=text, json=lambda: payload)


def _patch_get(side_effect):
    effects = side_effect if isinstance(side_effect, list) else [side_effect]
    return mock.patch("binance.client.requests.get", side_effect=effects)


@pytest.mark.unit
class TestIntervalValidation(unittest.TestCase):
    def test_all_supported_intervals_accepted(self):
        for interval in SUPPORTED_INTERVALS:
            with self.subTest(interval=interval):
                self.assertEqual(validate_interval(interval), interval)

    def test_interval_normalized_to_lowercase(self):
        self.assertEqual(validate_interval("1H"), "1h")
        self.assertEqual(validate_interval("4H"), "4h")

    def test_unsupported_interval_rejected(self):
        for interval in ("3m", "1w", "2h", "1d", "30m", ""):
            with self.subTest(interval=interval), self.assertRaises(InvalidIntervalError):
                validate_interval(interval)

    def test_non_string_interval_rejected(self):
        with self.assertRaises(InvalidIntervalError):
            validate_interval(None)


@pytest.mark.unit
class TestSymbolValidation(unittest.TestCase):
    def test_btcusdt_accepted(self):
        self.assertEqual(validate_symbol("BTCUSDT"), "BTCUSDT")
        self.assertEqual(validate_symbol("btcusdt"), "BTCUSDT")
        self.assertEqual(validate_symbol(" BTCUSDT "), "BTCUSDT")

    def test_invalid_symbols_rejected(self):
        for symbol in ("", "  ", "BTC/USDT", "btc-usdt", "BTC_S", None, 1234):
            with self.subTest(symbol=symbol), self.assertRaises(InvalidSymbolError):
                validate_symbol(symbol)

    def test_other_pairs_accepted(self):
        self.assertEqual(validate_symbol("ETHUSDT"), "ETHUSDT")


@pytest.mark.unit
class TestClosedCandleDetermination(unittest.TestCase):
    def test_closed_when_now_past_close_time(self):
        self.assertTrue(is_candle_closed(1_000, 1_001))
        self.assertTrue(is_candle_closed(1_000, 2_000))

    def test_forming_when_now_at_or_before_close_time(self):
        self.assertFalse(is_candle_closed(1_000, 1_000))
        self.assertFalse(is_candle_closed(1_000, 999))


@pytest.mark.unit
class TestKlineParsing(unittest.TestCase):
    def test_parses_valid_ohlcv(self):
        candles = parse_klines(_sample_rows(), now_ms=_SAMPLE_NOW_PAST_ALL)
        self.assertEqual(len(candles), 3)
        first = candles[0]
        self.assertIsInstance(first.timestamp, int)
        self.assertEqual(first.timestamp, 1_600_000_000_000)
        self.assertEqual(first.open, 1000.0)
        self.assertEqual(first.high, 1010.0)
        self.assertEqual(first.low, 990.0)
        self.assertEqual(first.close, 1005.0)
        self.assertEqual(first.volume, 12.5)
        self.assertEqual(first.close_time, 1_600_003_600_000)
        self.assertTrue(all(c.is_closed for c in candles))

    def test_forming_candle_flagged_not_closed(self):
        candles = parse_klines(_sample_rows(), now_ms=_SAMPLE_NOW_THIRD_FORMING)
        self.assertEqual([c.is_closed for c in candles], [True, True, False])

    def test_empty_payload_returns_empty_list(self):
        self.assertEqual(parse_klines([], now_ms=0), [])

    def test_rejects_non_list_payload(self):
        with self.assertRaises(MalformedKlineError):
            parse_klines({"data": []}, now_ms=0)

    def test_rejects_row_with_too_few_fields(self):
        with self.assertRaises(MalformedKlineError):
            parse_klines([[1_600_000_000_000, "1000", "1010", "990", "1005", "12"]], now_ms=0)

    def test_rejects_non_numeric_price(self):
        row = _kline(1_600_000_000_000, open_px="abc")
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_negative_price(self):
        row = _kline(1_600_000_000_000, open_px="-1.0")
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_non_finite_price(self):
        row = _kline(1_600_000_000_000, open_px="nan")
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)
        inf_row = _kline(1_600_003_600_000, close="inf")
        with self.assertRaises(MalformedKlineError):
            parse_klines([inf_row], now_ms=0)

    def test_rejects_negative_volume(self):
        row = _kline(1_600_000_000_000, volume="-3.0")
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_inconsistent_high(self):
        row = _kline(1_600_000_000_000, high="900.0")
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_inconsistent_low(self):
        row = _kline(1_600_000_000_000, low="1100.0")
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_negative_timestamp(self):
        row = _kline(-1)
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_close_before_open(self):
        row = _kline(1_600_003_600_000, close_time=1_600_000_000_000)
        with self.assertRaises(MalformedKlineError):
            parse_klines([row], now_ms=0)

    def test_rejects_unsorted_timestamps(self):
        rows = [_kline(1_600_003_600_000), _kline(1_600_000_000_000)]
        with self.assertRaises(MalformedKlineError):
            parse_klines(rows, now_ms=_SAMPLE_NOW_PAST_ALL)

    def test_rejects_duplicate_timestamps(self):
        rows = [_kline(1_600_000_000_000), _kline(1_600_000_000_000)]
        with self.assertRaises(MalformedKlineError):
            parse_klines(rows, now_ms=_SAMPLE_NOW_PAST_ALL)


@pytest.mark.unit
class TestFetchKlines(unittest.TestCase):
    def test_fetch_klines_success(self):
        raw = _sample_rows()
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=raw)) as mocked:
            candles = market.fetch_klines(now_ms=_SAMPLE_NOW_PAST_ALL)
        self.assertEqual(len(candles), 3)
        self.assertTrue(all(c.is_closed for c in candles))
        self.assertEqual(mocked.call_count, 1)

    def test_btcusdt_symbol_passed_to_request(self):
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=_sample_rows())) as mocked:
            market.fetch_klines(symbol="btcusdt", now_ms=_SAMPLE_NOW_PAST_ALL)
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs["params"]["symbol"], "BTCUSDT")

    def test_interval_limit_endtime_passed_to_request(self):
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=_sample_rows())) as mocked:
            market.fetch_klines(
                interval="4H", limit=100, end_time_ms=1_600_010_800_000,
                now_ms=_SAMPLE_NOW_PAST_ALL,
            )
        params = mocked.call_args.kwargs["params"]
        self.assertEqual(params["interval"], "4h")
        self.assertEqual(params["limit"], 100)
        self.assertEqual(params["endTime"], 1_600_010_800_000)

    def test_fetch_klines_empty_payload_raises(self):
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=[])), self.assertRaises(EmptyKlineError):
            market.fetch_klines(now_ms=_SAMPLE_NOW_PAST_ALL)


@pytest.mark.unit
class TestFetchClosedKlines(unittest.TestCase):
    def test_excludes_forming_candle(self):
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=_sample_rows())):
            candles = market.fetch_closed_klines(now_ms=_SAMPLE_NOW_THIRD_FORMING)
        self.assertEqual(len(candles), 2)
        self.assertTrue(all(c.is_closed for c in candles))

    def test_all_closed_when_now_past_window(self):
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=_sample_rows())):
            candles = market.fetch_closed_klines(now_ms=_SAMPLE_NOW_PAST_ALL)
        self.assertEqual(len(candles), 3)

    def test_raises_when_no_candle_is_closed(self):
        rows = [_kline(1_600_000_000_000)]
        market = BinanceMarketData(client=BinanceFuturesClient(max_retries=0))
        with _patch_get(_http(payload=rows)), self.assertRaises(EmptyKlineError):
            market.fetch_closed_klines(now_ms=1_600_000_000_000)


@pytest.mark.unit
class TestClientRetry(unittest.TestCase):
    def test_retries_on_5xx_then_succeeds(self):
        client = BinanceFuturesClient(max_retries=5, backoff=0)
        with _patch_get([_http(500, payload=None), _http(200, payload=[1, 2])]) as mocked:
            payload = client.get("/fapi/v1/klines", params={"symbol": "BTCUSDT"})
        self.assertEqual(payload, [1, 2])
        self.assertEqual(mocked.call_count, 2)

    def test_retries_exhausted_raises(self):
        client = BinanceFuturesClient(max_retries=2, backoff=0)
        with _patch_get([_http(500) for _ in range(3)]) as mocked, self.assertRaises(
            BinanceServerError
        ):
            client.get("/fapi/v1/klines")
        self.assertEqual(mocked.call_count, 3)

    def test_429_exhausted_raises_rate_limit_error(self):
        client = BinanceFuturesClient(max_retries=1, backoff=0)
        with _patch_get([_http(429) for _ in range(2)]), self.assertRaises(
            BinanceRateLimitError
        ):
            client.get("/fapi/v1/klines")

    def test_http_4xx_raises_without_retry(self):
        client = BinanceFuturesClient(max_retries=5, backoff=0)
        with _patch_get(_http(400)) as mocked, self.assertRaises(BinanceHTTPError):
            client.get("/fapi/v1/klines")
        self.assertEqual(mocked.call_count, 1)

    def test_non_json_body_raises(self):
        def bad_json():
            raise ValueError("not json")

        resp = SimpleNamespace(status_code=200, text="<html>", json=bad_json)
        client = BinanceFuturesClient(max_retries=0)
        with _patch_get(resp), self.assertRaises(BinanceError):
            client.get("/fapi/v1/klines")


@pytest.mark.unit
class TestDataFrame(unittest.TestCase):
    def test_candles_to_dataframe_columns(self):
        candles = parse_klines(_sample_rows(), now_ms=_SAMPLE_NOW_PAST_ALL)
        df = candles_to_dataframe(candles)
        self.assertIsInstance(df, pd.DataFrame)
        for column in (
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "is_closed",
        ):
            self.assertIn(column, df.columns)
        self.assertEqual(len(df), 3)


@pytest.mark.unit
class TestConnectionErrorRetry(unittest.TestCase):
    def test_retries_on_connection_error_then_succeeds(self):
        client = BinanceFuturesClient(max_retries=5, backoff=0)
        import requests as _requests_module

        side_effects = [
            _requests_module.ConnectionError("boom"),
            _http(200, payload=[7]),
        ]
        with _patch_get(side_effects) as mocked:
            payload = client.get("/fapi/v1/klines")
        self.assertEqual(payload, [7])
        self.assertEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
