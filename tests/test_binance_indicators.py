"""Phase 2 unit tests: deterministic Binance indicator engine.

All candles are synthetic and cost nothing to compute — no network access.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from binance.indicators import (
    MIN_REQUIRED_CANDLES,
    WARM_UP_CANDLES,
    EmptyIndicatorDataError,
    FormingCandleError,
    IndicatorError,
    IndicatorSnapshot,
    InsufficientDataError,
    compute_indicator_matrix,
    latest_indicators,
)
from binance.market_data import Candle

_HOUR_MS = 3_600_000
_START = 1_600_000_000_000

_INDICATOR_COLUMNS = (
    "ema20",
    "ema50",
    "ema200",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr14",
    "adx14",
    "adx_plus_di",
    "adx_minus_di",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "volume_sma20",
    "volume_ratio",
)


def _make_candles(closes, volumes=None):
    """Point candles (open == high == low == close) that are all closed."""
    n = len(closes)
    volumes = volumes if volumes is not None else [1000.0] * n
    return [
        Candle(
            timestamp=_START + i * _HOUR_MS,
            open=float(closes[i]),
            high=float(closes[i]),
            low=float(closes[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            close_time=_START + (i + 1) * _HOUR_MS,
            is_closed=True,
        )
        for i in range(n)
    ]


def _constant_candles(n, price=100.0, volume=1000.0):
    return _make_candles([price] * n, [volume] * n)


def _first_valid(series: pd.Series) -> int:
    return int(series.notna().idxmax()) if series.notna().any() else -1


def _ref_ema(values, period):
    """Reference EMA (SMA seed) recomputed independently from the definition."""
    out = [float("nan")] * len(values)
    if len(values) < period:
        return out
    out[period - 1] = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _ref_macd(closes):
    line = [f - s for f, s in zip(_ref_ema(closes, 12), _ref_ema(closes, 26), strict=True)]
    start = next(i for i, value in enumerate(line) if value == value)
    signal = [float("nan")] * start + _ref_ema(line[start:], 9)
    hist = [
        value - sig if sig == sig else float("nan")
        for value, sig in zip(line, signal, strict=True)
    ]
    return line, signal, hist


@pytest.mark.unit
class TestConstants(unittest.TestCase):
    def test_min_required_candles_is_200(self):
        self.assertEqual(MIN_REQUIRED_CANDLES, 200)
        self.assertEqual(max(WARM_UP_CANDLES.values()), MIN_REQUIRED_CANDLES)

    def test_warm_up_values_positive(self):
        for name, warm in WARM_UP_CANDLES.items():
            with self.subTest(name=name):
                self.assertGreater(warm, 0)
                self.assertLessEqual(warm, MIN_REQUIRED_CANDLES)


@pytest.mark.unit
class TestComputeMatrix(unittest.TestCase):
    def test_returns_dataframe_with_indicator_columns(self):
        df = compute_indicator_matrix(_constant_candles(200))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 200)
        for column in _INDICATOR_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(column, df.columns)

    def test_preserves_timestamps_and_prices(self):
        candles = _constant_candles(5)
        df = compute_indicator_matrix(candles)
        self.assertEqual(list(df["timestamp"]), [c.timestamp for c in candles])
        self.assertEqual(list(df["close"]), [c.close for c in candles])

    def test_builds_from_tiny_input_without_error(self):
        df = compute_indicator_matrix(_constant_candles(3))
        self.assertEqual(len(df), 3)
        self.assertTrue(all(df[col].isna().all() for col in _INDICATOR_COLUMNS))


@pytest.mark.unit
class TestEma(unittest.TestCase):
    def test_ema20_seed_and_first_step_hand_computed(self):
        closes = [100.0] * 20 + [200.0]
        df = compute_indicator_matrix(_make_candles(closes))
        self.assertEqual(df["ema20"].iloc[19], 100.0)
        expected = 2300.0 / 21.0  # (2/21)*200 + (19/21)*100
        self.assertAlmostEqual(df["ema20"].iloc[20], expected, places=9)

    def test_ema_matches_simple_recursion(self):
        closes = [float(100 + 7 * (i % 9)) for i in range(120)]
        df = compute_indicator_matrix(_make_candles(closes))
        for col, period in (("ema20", 20), ("ema50", 50)):
            with self.subTest(col=col):
                expected = _ref_ema(closes, period)
                np.testing.assert_allclose(df[col].to_numpy(), expected, equal_nan=True)

    def test_ema_constant_tracks_price(self):
        df = compute_indicator_matrix(_constant_candles(200, price=123.5))
        for col, start in (("ema20", 19), ("ema50", 49), ("ema200", 199)):
            with self.subTest(col=col):
                np.testing.assert_allclose(df[col].iloc[start:], 123.5)

    def test_ema_warmup_rows_are_nan(self):
        df = compute_indicator_matrix(_constant_candles(200))
        self.assertEqual(_first_valid(df["ema20"]), 19)
        self.assertEqual(_first_valid(df["ema50"]), 49)
        self.assertEqual(_first_valid(df["ema200"]), 199)


@pytest.mark.unit
class TestRsi(unittest.TestCase):
    def test_rsi_hand_computed_wilder(self):
        closes = [100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106]
        df = compute_indicator_matrix(_make_candles(closes))
        # 10 gains of +1 and 4 losses of -1 -> RS = 2.5 -> RSI = 500/7.
        self.assertAlmostEqual(df["rsi14"].iloc[14], 500.0 / 7.0, places=9)

    def test_rsi_is_100_in_monotonic_uptrend(self):
        df = compute_indicator_matrix(_make_candles([float(i) for i in range(60)]))
        self.assertEqual(df["rsi14"].iloc[14], 100.0)
        self.assertEqual(df["rsi14"].iloc[59], 100.0)

    def test_rsi_is_0_in_monotonic_downtrend(self):
        closes = [float(200 - i) for i in range(60)]
        df = compute_indicator_matrix(_make_candles(closes))
        self.assertEqual(df["rsi14"].iloc[14], 0.0)
        self.assertEqual(df["rsi14"].iloc[59], 0.0)

    def test_rsi_is_50_when_price_is_flat(self):
        df = compute_indicator_matrix(_constant_candles(30))
        self.assertEqual(df["rsi14"].iloc[14], 50.0)

    def test_rsi_warmup_rows_are_nan(self):
        df = compute_indicator_matrix(_constant_candles(30))
        self.assertEqual(_first_valid(df["rsi14"]), 14)
        self.assertTrue(df["rsi14"].iloc[:14].isna().all())


@pytest.mark.unit
class TestMacd(unittest.TestCase):
    def test_macd_zero_when_price_is_flat(self):
        df = compute_indicator_matrix(_constant_candles(100))
        self.assertEqual(df["macd"].iloc[-1], 0.0)
        self.assertEqual(df["macd_signal"].iloc[-1], 0.0)
        self.assertEqual(df["macd_histogram"].iloc[-1], 0.0)

    def test_macd_matches_reference(self):
        closes = [float(100 + 3 * math.sin(i / 5) + i * 0.1) for i in range(80)]
        df = compute_indicator_matrix(_make_candles(closes))
        line, signal, hist = _ref_macd(closes)
        np.testing.assert_allclose(df["macd"].to_numpy(), line, equal_nan=True, atol=1e-9)
        np.testing.assert_allclose(
            df["macd_signal"].to_numpy(), signal, equal_nan=True, atol=1e-9
        )
        np.testing.assert_allclose(
            df["macd_histogram"].to_numpy(), hist, equal_nan=True, atol=1e-9
        )

    def test_macd_positive_in_uptrend_negative_in_downtrend(self):
        up = compute_indicator_matrix(_make_candles([float(i) for i in range(80)]))
        down = compute_indicator_matrix(_make_candles([float(200 - i) for i in range(80)]))
        self.assertGreater(up["macd"].iloc[-1], 0.0)
        self.assertLess(down["macd"].iloc[-1], 0.0)

    def test_macd_warmup(self):
        df = compute_indicator_matrix(_constant_candles(100))
        self.assertEqual(_first_valid(df["macd"]), 25)
        self.assertEqual(_first_valid(df["macd_signal"]), 33)
        self.assertTrue(df["macd_signal"].iloc[:33].isna().all())


@pytest.mark.unit
class TestAtr(unittest.TestCase):
    def test_atr_equals_constant_true_range(self):
        closes = [float(100 + 2 * i) for i in range(50)]
        df = compute_indicator_matrix(_make_candles(closes))
        self.assertEqual(df["atr14"].iloc[14], 2.0)
        self.assertEqual(df["atr14"].iloc[49], 2.0)

    def test_atr_warmup_rows_are_nan(self):
        df = compute_indicator_matrix(_constant_candles(30))
        self.assertEqual(_first_valid(df["atr14"]), 14)
        self.assertTrue(df["atr14"].iloc[:14].isna().all())


@pytest.mark.unit
class TestAdx(unittest.TestCase):
    def test_adx_peaks_in_strong_trend(self):
        closes = [float(100 + i) for i in range(40)]
        df = compute_indicator_matrix(_make_candles(closes))
        self.assertAlmostEqual(df["adx14"].iloc[27], 100.0, places=9)
        self.assertAlmostEqual(df["adx_plus_di"].iloc[27], 100.0, places=9)
        self.assertAlmostEqual(df["adx_minus_di"].iloc[27], 0.0, places=9)

    def test_adx_is_zero_when_price_is_flat(self):
        df = compute_indicator_matrix(_constant_candles(40))
        self.assertEqual(df["adx14"].iloc[27], 0.0)

    def test_adx_warmup_rows_are_nan(self):
        df = compute_indicator_matrix(_constant_candles(40))
        self.assertEqual(_first_valid(df["adx14"]), 27)
        self.assertTrue(df["adx14"].iloc[:27].isna().all())


@pytest.mark.unit
class TestBollinger(unittest.TestCase):
    def test_bands_collapse_when_price_is_flat(self):
        df = compute_indicator_matrix(_constant_candles(30))
        self.assertEqual(df["bb_mid"].iloc[-1], 100.0)
        self.assertEqual(df["bb_upper"].iloc[-1], 100.0)
        self.assertEqual(df["bb_lower"].iloc[-1], 100.0)

    def test_band_math_hand_computed(self):
        closes = [100.0, 110.0] * 10
        df = compute_indicator_matrix(_make_candles(closes))
        self.assertAlmostEqual(df["bb_mid"].iloc[19], 105.0, places=9)
        self.assertAlmostEqual(df["bb_upper"].iloc[19], 115.0, places=9)
        self.assertAlmostEqual(df["bb_lower"].iloc[19], 95.0, places=9)


@pytest.mark.unit
class TestVolume(unittest.TestCase):
    def test_volume_sma20_hand_computed(self):
        volumes = [100.0] * 20 + [400.0]
        df = compute_indicator_matrix(_make_candles([100.0] * 21, volumes))
        self.assertAlmostEqual(df["volume_sma20"].iloc[19], 100.0, places=9)
        self.assertAlmostEqual(df["volume_sma20"].iloc[20], 115.0, places=9)

    def test_volume_ratio_hand_computed(self):
        volumes = [100.0] * 20 + [400.0]
        df = compute_indicator_matrix(_make_candles([100.0] * 21, volumes))
        self.assertAlmostEqual(df["volume_ratio"].iloc[19], 1.0, places=9)
        self.assertAlmostEqual(df["volume_ratio"].iloc[20], 400.0 / 115.0, places=9)

    def test_volume_warmup_rows_are_nan(self):
        df = compute_indicator_matrix(_constant_candles(30))
        self.assertEqual(_first_valid(df["volume_sma20"]), 19)
        self.assertEqual(_first_valid(df["volume_ratio"]), 19)

    def test_zero_volume_window_yields_nan_not_inf(self):
        # 20 zero-volume bars then a print: sma is 0, so the raw ratio is
        # inf — normalized to NaN so snapshot validation treats it as a gap.
        volumes = [0.0] * 20 + [400.0]
        df = compute_indicator_matrix(_make_candles([100.0] * 21, volumes))
        self.assertTrue(math.isnan(df["volume_ratio"].iloc[19]))
        self.assertFalse(math.isinf(df["volume_ratio"].iloc[19]))


@pytest.mark.unit
class TestNoLookAhead(unittest.TestCase):
    def test_prefix_of_longer_series_is_identical(self):
        closes = [float(100 + 2 * math.sin(i / 3) + i * 0.25) for i in range(300)]
        full = compute_indicator_matrix(_make_candles(closes))
        first_200 = compute_indicator_matrix(_make_candles(closes[:200]))
        for col in ("timestamp", *_INDICATOR_COLUMNS):
            with self.subTest(col=col):
                got = full[col].iloc[:200].to_numpy()
                want = first_200[col].to_numpy()
                if col != "timestamp":
                    np.testing.assert_allclose(got, want, equal_nan=True)
                else:
                    np.testing.assert_array_equal(got, want)

    def test_repeated_computation_is_deterministic(self):
        closes = [float(100 + 5 * math.sin(i / 4)) for i in range(250)]
        first = compute_indicator_matrix(_make_candles(closes))
        second = compute_indicator_matrix(_make_candles(closes))
        pd.testing.assert_frame_equal(first, second)


@pytest.mark.unit
class TestInputValidation(unittest.TestCase):
    def test_empty_candles_raise(self):
        with self.assertRaises(EmptyIndicatorDataError):
            compute_indicator_matrix([])
        with self.assertRaises(EmptyIndicatorDataError):
            latest_indicators([])

    def test_forming_candle_raises(self):
        candles = _constant_candles(200)
        forming = replace(candles[-1], is_closed=False)
        with self.assertRaises(FormingCandleError):
            compute_indicator_matrix([*candles[:-1], forming])
        with self.assertRaises(FormingCandleError):
            latest_indicators([*candles[:-1], forming])

    def test_unsorted_timestamps_raise(self):
        candles = _constant_candles(5)
        with self.assertRaises(IndicatorError):
            compute_indicator_matrix(list(reversed(candles)))


@pytest.mark.unit
class TestLatestIndicators(unittest.TestCase):
    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            latest_indicators(_constant_candles(MIN_REQUIRED_CANDLES - 1))

    def test_sufficient_data_returns_snapshot(self):
        candles = _constant_candles(200, price=100.0, volume=1000.0)
        snap = latest_indicators(candles)
        self.assertIsInstance(snap, IndicatorSnapshot)
        self.assertEqual(snap.timestamp, candles[-1].timestamp)
        self.assertEqual(snap.last_close, 100.0)
        self.assertAlmostEqual(snap.ema200, 100.0, places=9)
        self.assertEqual(snap.rsi14, 50.0)
        self.assertEqual(snap.macd, 0.0)
        self.assertEqual(snap.atr14, 0.0)
        self.assertEqual(snap.adx14, 0.0)
        self.assertEqual(snap.bb_mid, 100.0)
        self.assertEqual(snap.volume_ratio, 1.0)

    def test_snapshot_to_dict(self):
        snap = latest_indicators(_constant_candles(200))
        data = snap.to_dict()
        self.assertEqual(len(data), len(snap.__dict__))


if __name__ == "__main__":
    unittest.main()
