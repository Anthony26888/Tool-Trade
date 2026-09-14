"""Phase 6 unit tests: the Signal Engine (signal_engine/engine.py).

Covers the full conversion pipeline from Phase 5 analysis to a persisted OPEN
signal: LONG/SHORT/WAIT flows, the one-OPEN invariant (including concurrency),
the AI lock and its ordering, immutability, numeric safety through the whole
path, fail-safe behavior, metadata persistence, and proof that no Binance order
or Demo execution is ever touched.
"""

from __future__ import annotations

import sqlite3
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from signal_engine import SignalOutcome
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis


def _long(**overrides):
    return make_analysis("LONG", **overrides)


def _short(**overrides):
    defaults = {"entry_price": 59000.0, "stop_loss": 60000.0, "take_profit": 58000.0}
    defaults.update(overrides)
    return make_analysis("SHORT", **defaults)


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()

    def tearDown(self):
        self.harness.close()

    @property
    def engine(self):
        return self.harness.engine

    @property
    def repo(self):
        return self.harness.repository


@pytest.mark.unit
class TestLongFlow(EngineTestCase):
    def test_valid_long_creates_open_signal(self):
        result = self.engine.process(_long())
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        self.assertIsNotNone(result.signal)
        self.assertEqual(result.signal.status, "PENDING_ENTRY")
        self.assertEqual(result.signal.direction, "LONG")
        self.assertEqual(result.signal.entry, Decimal("61000.0"))
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_invalid_long_sl_ge_entry_rejected(self):
        for sl in (61000.0, 62000.0):
            with self.subTest(sl=sl):
                result = self.engine.process(_long(stop_loss=sl))
                self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_invalid_long_entry_ge_tp_rejected(self):
        for tp in (61000.0, 60000.0):
            with self.subTest(tp=tp):
                result = self.engine.process(_long(take_profit=tp))
                self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_missing_sl_rejected(self):
        result = self.engine.process(_long(stop_loss=None))
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_missing_tp_rejected(self):
        result = self.engine.process(_long(take_profit=None))
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_missing_entry_rejected(self):
        result = self.engine.process(_long(entry_price=None))
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])


@pytest.mark.unit
class TestShortFlow(EngineTestCase):
    def test_valid_short_creates_open_signal(self):
        result = self.engine.process(_short())
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        self.assertEqual(result.signal.direction, "SHORT")
        self.assertEqual(result.signal.entry, Decimal("59000.0"))
        self.assertEqual(result.signal.status, "PENDING_ENTRY")
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_invalid_short_tp_ge_entry_rejected(self):
        for tp in (59000.0, 61000.0):
            with self.subTest(tp=tp):
                result = self.engine.process(_short(take_profit=tp))
                self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_invalid_short_entry_ge_sl_rejected(self):
        for entry in (60000.0, 61000.0):
            with self.subTest(entry=entry):
                result = self.engine.process(_short(entry_price=entry))
                self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_missing_sl_and_tp_rejected(self):
        self.assertEqual(
            self.engine.process(_short(stop_loss=None)).outcome, SignalOutcome.REJECTED
        )
        self.assertEqual(
            self.engine.process(_short(take_profit=None)).outcome, SignalOutcome.REJECTED
        )
        self.assertEqual(self.repo.list_signals(), [])


@pytest.mark.unit
class TestWaitFlow(EngineTestCase):
    def test_wait_creates_no_signal(self):
        result = self.engine.process(make_analysis("WAIT"))
        self.assertEqual(result.outcome, SignalOutcome.WAIT)
        self.assertIsNone(result.signal)
        self.assertEqual(self.repo.list_signals(), [])
        self.assertIsNone(self.repo.get_open_signal())

    def test_wait_leaves_database_unchanged_when_idle(self):
        before = self.repo.list_signals()
        self.assertEqual(self.engine.process(make_analysis("WAIT")).outcome, SignalOutcome.WAIT)
        self.assertEqual(self.repo.list_signals(), before)

    def test_wait_does_not_modify_existing_open_signal(self):
        created = self.engine.process(_long()).signal
        result = self.engine.process(make_analysis("WAIT"))
        self.assertEqual(result.outcome, SignalOutcome.WAIT)
        reloaded = self.repo.get_signal(created.id)
        self.assertEqual(reloaded.status, "PENDING_ENTRY")
        self.assertEqual(reloaded.entry, created.entry)
        self.assertEqual(len(self.repo.list_signals()), 1)


@pytest.mark.unit
class TestOneOpenInvariant(EngineTestCase):
    def test_open_blocks_new_long(self):
        self.engine.process(_long())
        result = self.engine.process(_long())
        self.assertEqual(result.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_open_blocks_new_short(self):
        self.engine.process(_long())
        result = self.engine.process(_short())
        self.assertEqual(result.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_open_signal_remains_unchanged_after_block(self):
        first = self.engine.process(
            _long(entry_price=61000.0, stop_loss=60000.0, take_profit=64000.0)
        ).signal
        blocked = self.engine.process(
            _long(entry_price=62000.0, stop_loss=61000.0, take_profit=65000.0)
        )
        self.assertEqual(blocked.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        reloaded = self.repo.get_signal(first.id)
        for attr in ("entry", "stop_loss", "take_profit", "direction",
                     "timeframe", "created_at", "opened_at"):
            self.assertEqual(getattr(reloaded, attr), getattr(first, attr), attr)
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_concurrent_creation_single_open(self):
        analysis = _long()
        barrier = threading.Barrier(2)

        def create(_):
            barrier.wait()
            return self.engine.process(analysis)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(create, range(2)))
        outcomes = sorted(r.outcome for r in results)
        self.assertEqual(outcomes, [SignalOutcome.BLOCKED_OPEN_SIGNAL, SignalOutcome.CREATED])
        self.assertEqual(len(self.repo.list_signals()), 1)
        self.assertIsNotNone(self.repo.get_active_signal())


@pytest.mark.unit
class TestAILock(EngineTestCase):
    def test_open_blocks_analysis_before_llm_call(self):
        self.engine.process(_long())
        analyzer = MagicMock(side_effect=AssertionError("LLM must not be invoked"))
        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        analyzer.assert_not_called()

    def test_architecture_gate_order_open_exists(self):
        self.engine.process(_long())
        analyzer = MagicMock()
        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        analyzer.assert_not_called()
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_analyzer_called_when_idle(self):
        analyzer = MagicMock(return_value=_long())
        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        analyzer.assert_called_once_with([], None)

    def test_race_after_analysis_still_blocked_before_persist(self):
        def analyzer(_candles, _indicators):
            self.engine.process(_long())
            return _long()

        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        self.assertEqual(len(self.repo.list_signals()), 1)


@pytest.mark.unit
class TestImmutability(EngineTestCase):
    def test_later_ai_result_cannot_modify_open_signal(self):
        first = self.engine.process(
            _long(entry_price=61000.0, stop_loss=60000.0, take_profit=64000.0)
        ).signal
        later = _long(entry_price=70000.0, stop_loss=69000.0, take_profit=71000.0)
        result = self.engine.process(later)
        self.assertEqual(result.outcome, SignalOutcome.BLOCKED_OPEN_SIGNAL)
        reloaded = self.repo.get_signal(first.id)
        self.assertEqual(reloaded.entry, Decimal("61000.0"))
        self.assertEqual(reloaded.stop_loss, Decimal("60000.0"))
        self.assertEqual(reloaded.take_profit, Decimal("64000.0"))
        self.assertEqual(reloaded.direction, "LONG")
        self.assertEqual(reloaded.timeframe, "1h")


@pytest.mark.unit
class TestNumericValidation(EngineTestCase):
    def test_high_precision_decimals_persisted_exactly(self):
        entry = Decimal("61000.123456789012345678901234")
        sl = Decimal("60888.000000000000000001")
        tp = Decimal("61234.9876543210987654321")
        result = self.engine.process(
            _long(entry_price=entry, stop_loss=sl, take_profit=tp)
        )
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        reloaded = self.repo.get_signal(result.signal.id)
        self.assertEqual(reloaded.entry, entry)
        self.assertEqual(reloaded.stop_loss, sl)
        self.assertEqual(reloaded.take_profit, tp)

    def test_zero_and_negative_rejected(self):
        for overrides in (
            {"entry_price": 0.0},
            {"stop_loss": 0.0},
            {"take_profit": -1.0},
            {"entry_price": -100.0},
        ):
            with self.subTest(**overrides):
                self.assertEqual(
                    self.engine.process(_long(**overrides)).outcome, SignalOutcome.REJECTED
                )
        self.assertEqual(self.repo.list_signals(), [])

    def test_nan_and_infinity_rejected(self):
        for overrides in (
            {"entry_price": float("nan")},
            {"stop_loss": float("inf")},
            {"take_profit": float("-inf")},
            {"entry_price": Decimal("NaN")},
        ):
            with self.subTest(**overrides):
                self.assertEqual(
                    self.engine.process(_long(**overrides)).outcome, SignalOutcome.REJECTED
                )
        self.assertEqual(self.repo.list_signals(), [])

    def test_invalid_decimal_rejected(self):
        for overrides in (
            {"entry_price": "abc"},
            {"stop_loss": "1,000"},
            {"take_profit": "abc"},
        ):
            with self.subTest(**overrides):
                self.assertEqual(
                    self.engine.process(_long(**overrides)).outcome, SignalOutcome.REJECTED
                )
        self.assertEqual(self.repo.list_signals(), [])

    def test_decimal_comparison_semantics_no_binary_float(self):
        near_order = _long(
            entry_price=Decimal("100"),
            stop_loss=Decimal("99.999999999999999999"),
            take_profit=Decimal("100.000000000000000001"),
        )
        created = self.engine.process(near_order)
        self.assertEqual(created.outcome, SignalOutcome.CREATED)
        self.repo.cancel_signal(created.signal.id, close_reason="reset")
        self.assertFalse(self.engine.state.is_open())
        self.assertFalse(self.engine.state.is_active())

        equal_levels = _long(
            entry_price=Decimal("100"),
            stop_loss=Decimal("100"),
            take_profit=Decimal("100.5"),
        )
        self.assertEqual(self.engine.process(equal_levels).outcome, SignalOutcome.REJECTED)


@pytest.mark.unit
class TestFailureSafety(EngineTestCase):
    def test_analyzer_failure_no_signal(self):
        def analyzer(_c, _i):
            raise RuntimeError("provider timeout")

        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.ERROR)
        self.assertIsNone(result.signal)
        self.assertEqual(self.repo.list_signals(), [])

    def test_insufficient_market_data_no_signal(self):
        from signal_engine import AnalysisContextError

        def analyzer(_c, _i):
            raise AnalysisContextError("no closed candles available")

        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.ERROR)
        self.assertEqual(self.repo.list_signals(), [])

    def test_malformed_ai_output_no_signal(self):
        created = self.engine.process(_long())
        self.assertEqual(created.outcome, SignalOutcome.CREATED)
        self.repo.cancel_signal(created.signal.id, close_reason="reset")
        self.assertIsNone(self.repo.get_open_signal())
        self.assertIsNone(self.repo.get_active_signal())
        pre_count = len(self.repo.list_signals())

        result = self.engine.process(object())
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertIsNone(result.signal)
        self.assertEqual(len(self.repo.list_signals()), pre_count)
        self.assertIsNone(self.repo.get_active_signal())

    def test_analyzer_returns_non_analysis_no_signal(self):
        analyzer = MagicMock(return_value={"decision": "LONG"})
        result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_invalid_decision_no_signal(self):
        result = self.engine.process(make_analysis("BUY"))
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_invalid_confidence_no_signal(self):
        result = self.engine.process(_long(confidence=150.0))
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])

    def test_database_failure_no_false_success(self):
        with patch.object(
            self.engine.repository,
            "create_signal",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = self.engine.process(_long())
        self.assertEqual(result.outcome, SignalOutcome.ERROR)
        self.assertIsNone(result.signal)
        self.assertEqual(self.repo.list_signals(), [])

    def test_validation_failure_no_signal(self):
        result = self.engine.process(_long(take_profit=None))
        self.assertEqual(result.outcome, SignalOutcome.REJECTED)
        self.assertEqual(self.repo.list_signals(), [])


@pytest.mark.unit
class TestMetadata(EngineTestCase):
    def test_signal_metadata_persisted(self):
        analysis = _long(
            confidence=83.5,
            reasoning="trend + momentum",
            provider="deepseek",
            model="deepseek-v4-flash",
            analysis_timestamp="2026-09-10T01:00:00.000Z",
            market_timestamp="2026-09-10T00:00:00.000Z",
            candle_close_price=61050.25,
        )
        result = self.engine.process(analysis)
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        signal = self.repo.get_signal(result.signal.id)
        self.assertEqual(signal.symbol, "BTCUSDT")
        self.assertEqual(signal.timeframe, "1h")
        self.assertEqual(signal.direction, "LONG")
        self.assertEqual(signal.status, "PENDING_ENTRY")
        self.assertEqual(signal.confidence, 84)  # deterministic half-up
        self.assertEqual(signal.provider, "deepseek")
        self.assertEqual(signal.model_name, "deepseek-v4-flash")
        self.assertEqual(signal.rationale, "trend + momentum")
        self.assertEqual(signal.analysis_timestamp, "2026-09-10T01:00:00.000Z")
        self.assertEqual(signal.market_timestamp, "2026-09-10T00:00:00.000Z")
        self.assertEqual(signal.candle_close_price, Decimal("61050.25"))
        self.assertTrue(signal.created_at.endswith("Z"))
        # A PENDING_ENTRY has no open timestamp yet.
        self.assertIsNone(signal.opened_at)
        self.assertIsNone(signal.strategy_name)
        self.assertIsNone(signal.strategy_version)
        self.assertIsNone(signal.result)

    def test_confidence_default_normalization(self):
        result = self.engine.process(_long(confidence=80.0))
        self.assertEqual(self.repo.get_signal(result.signal.id).confidence, 80)


@pytest.mark.unit
class TestNoExecution(EngineTestCase):
    def test_process_places_no_binance_orders(self):
        with patch("binance.client.BinanceFuturesClient") as client_cls:
            result = self.engine.process(_long())
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        client_cls.assert_not_called()

    def test_analyze_and_create_places_no_binance_orders(self):
        analyzer = MagicMock(return_value=_long())
        with patch("binance.client.BinanceFuturesClient") as client_cls:
            result = self.engine.analyze_and_create([], None, analyzer)
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        client_cls.assert_not_called()

    def test_engine_exposes_no_scheduler_entry_points(self):
        for attr in ("run_forever", "run_loop", "schedule", "monitor"):
            self.assertFalse(hasattr(self.engine, attr), attr)


if __name__ == "__main__":
    unittest.main()
