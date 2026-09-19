"""Phase 5 unit tests: deterministic closed-candle analysis context.

``signal_engine/context.py`` builds the single bounded context handed to the
LLM. These tests prove it only ever reasons from closed candles, never includes
future or forming candles, requires aligned and complete indicators, and renders
deterministically.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pytest

from signal_engine.context import (
    SNAPSHOT_COLUMNS,
    AnalysisContextError,
    build_analysis_context,
)
from tests.signal_engine_test_helpers import (
    INTERVAL_MS,
    indicators_for,
    make_candles,
)


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@pytest.mark.unit
class TestBuildAnalysisContext(unittest.TestCase):
    def test_valid_context_from_closed_candles(self):
        candles = make_candles(240)
        context = build_analysis_context(candles, indicators_for(candles))
        self.assertEqual(context.symbol, "BTCUSDT")
        self.assertEqual(context.timeframe, "1h")
        self.assertEqual(context.total_candle_count, 240)
        self.assertEqual(context.last_close, candles[-1].close)
        self.assertEqual(context.snapshot.timestamp, candles[-1].timestamp)
        self.assertEqual(context.snapshot.last_close, candles[-1].close)

    def test_analysis_candle_is_latest_closed_candle(self):
        candles = make_candles(240)
        context = build_analysis_context(candles, indicators_for(candles))
        self.assertEqual(context.market_timestamp, _ms_to_iso(candles[-1].timestamp))
        self.assertEqual(context.closed_at, _ms_to_iso(candles[-1].close_time))

    def test_no_future_candle_is_included(self):
        candles = make_candles(240)
        context = build_analysis_context(candles, indicators_for(candles))
        max_timestamp = max(c.timestamp for c in candles)
        self.assertEqual(max_timestamp, candles[-1].timestamp)
        context_timestamps = set(context.candles["timestamp"])
        self.assertEqual(max(context_timestamps), candles[-1].timestamp)
        self.assertTrue(all(t <= candles[-1].close_time for t in context_timestamps))

    def test_forming_candle_is_never_an_analysis_candle(self):
        candles = make_candles(240, forming=True)
        with self.assertRaises(AnalysisContextError) as ctx:
            build_analysis_context(candles, indicators_for(make_candles(240)))
        self.assertIn("forming", str(ctx.exception).lower())
        self.assertIn("never be used as an analysis candle", str(ctx.exception).lower())

    def test_insufficient_candles_rejected(self):
        candles = make_candles(199)
        with self.assertRaises(AnalysisContextError) as ctx:
            build_analysis_context(candles, indicators_for(candles))
        self.assertIn("200", str(ctx.exception))

    def test_empty_candles_rejected(self):
        with self.assertRaises(AnalysisContextError):
            build_analysis_context([], indicators_for(make_candles(240)))

    def test_out_of_order_candles_rejected(self):
        candles = make_candles(240)
        candles.sort(key=lambda c: c.timestamp, reverse=True)
        with self.assertRaises(AnalysisContextError):
            build_analysis_context(candles, indicators_for(make_candles(240)))

    def test_misaligned_indicators_rejected(self):
        candles = make_candles(240)
        indicators = indicators_for(candles)
        tampered = indicators.copy()
        tampered.loc[tampered.index[-1], "timestamp"] = (
            candles[-1].timestamp + INTERVAL_MS
        )
        with self.assertRaises(AnalysisContextError) as ctx:
            build_analysis_context(candles, tampered)
        self.assertIn("aligned", str(ctx.exception).lower())

    def test_wrong_row_count_rejected(self):
        candles = make_candles(240)
        with self.assertRaises(AnalysisContextError):
            build_analysis_context(candles, indicators_for(candles).iloc[:200])

    def test_missing_indicator_column_rejected(self):
        candles = make_candles(240)
        indicators = indicators_for(candles).drop(columns=["rsi14"])
        with self.assertRaises(AnalysisContextError) as ctx:
            build_analysis_context(candles, indicators)
        self.assertIn("rsi14", str(ctx.exception))

    def test_nan_in_snapshot_rejected(self):
        candles = make_candles(240)
        indicators = indicators_for(candles)
        indicators.loc[indicators.index[-1], "ema200"] = float("nan")
        with self.assertRaises(AnalysisContextError) as ctx:
            build_analysis_context(candles, indicators)
        self.assertIn("ema200", str(ctx.exception))

    def test_warmup_nan_in_history_renders_as_n_a(self):
        candles = make_candles(210)
        context = build_analysis_context(
            candles, indicators_for(candles), max_candles=40
        )
        self.assertIn("n/a", context.rendered)

    def test_bounded_lookback(self):
        candles = make_candles(240)
        context = build_analysis_context(
            candles, indicators_for(candles), max_candles=40
        )
        self.assertEqual(context.max_candles, 40)
        self.assertEqual(len(context.candles), 40)
        self.assertEqual(len(context.indicators), 40)

    def test_default_max_candles_is_trimmed_for_token_budget(self):
        # Phase A: the default look-back is 20 candles (not 40) so each LLM
        # call costs roughly half the prompt tokens.
        candles = make_candles(240)
        context = build_analysis_context(candles, indicators_for(candles))
        self.assertEqual(context.max_candles, 20)
        self.assertEqual(len(context.candles), 20)
        self.assertEqual(len(context.indicators), 20)

    def test_history_table_omits_macd_signal(self):
        # The history table drops macd_signal (snapshot keeps MACD/signal/
        # histogram); the snapshot must still carry the signal line.
        candles = make_candles(240)
        context = build_analysis_context(candles, indicators_for(candles))
        snapshot, history = context.rendered.split("## Indicator history")
        self.assertIn("signal=", snapshot)
        self.assertNotIn("macd_signal", history)

    def test_max_candles_clamped_to_total(self):
        candles = make_candles(240)
        context = build_analysis_context(
            candles, indicators_for(candles), max_candles=500
        )
        self.assertEqual(context.max_candles, 240)
        self.assertEqual(len(context.candles), 240)

    def test_symbol_and_timeframe_normalized(self):
        candles = make_candles(240)
        context = build_analysis_context(
            candles, indicators_for(candles), symbol=" btcusdt ", timeframe="1H"
        )
        self.assertEqual(context.symbol, "BTCUSDT")
        self.assertEqual(context.timeframe, "1h")

    def test_rendered_is_deterministic(self):
        candles_a = make_candles(240)
        first = build_analysis_context(candles_a, indicators_for(candles_a))
        second = build_analysis_context(candles_a, indicators_for(candles_a))
        candles_b = make_candles(240)
        third = build_analysis_context(candles_b, indicators_for(candles_b))
        self.assertEqual(first.rendered, second.rendered)
        self.assertEqual(first.rendered, third.rendered)

    def test_rendered_is_bounded_in_size(self):
        candles = make_candles(240)
        context = build_analysis_context(
            candles, indicators_for(candles), max_candles=40
        )
        self.assertLess(len(context.rendered), 20_000)

    def test_rendered_states_closed_candle_source(self):
        candles = make_candles(240)
        context = build_analysis_context(candles, indicators_for(candles))
        self.assertIn("CLOSED", context.rendered.upper())
        self.assertIn("Binance USDT-M Futures", context.rendered)

    def test_rendered_most_recent_candle_first(self):
        candles = make_candles(240)
        context = build_analysis_context(
            candles, indicators_for(candles), max_candles=40
        )
        section = context.rendered.split("## Recent closed candles")[1]
        first_row = next(ln for ln in section.splitlines() if ln.startswith("| 1 |"))
        self.assertIn(_ms_to_iso(candles[-1].timestamp), first_row)

    def test_snapshot_matches_last_indicator_row(self):
        candles = make_candles(240)
        indicators = indicators_for(candles)
        context = build_analysis_context(candles, indicators)
        last = indicators.iloc[-1]
        self.assertEqual(context.snapshot.ema20, float(last["ema20"]))
        self.assertEqual(context.snapshot.rsi14, float(last["rsi14"]))
        self.assertEqual(context.snapshot.atr14, float(last["atr14"]))
        self.assertEqual(context.snapshot.volume_ratio, float(last["volume_ratio"]))

    def test_snapshot_columns_present_in_indicator_matrix(self):
        candles = make_candles(240)
        indicators = indicators_for(candles)
        for column in SNAPSHOT_COLUMNS:
            self.assertIn(column, indicators.columns)
