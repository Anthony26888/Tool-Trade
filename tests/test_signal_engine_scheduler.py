"""Phase 10 unit tests for the 1H analysis scheduler (signal_engine/scheduler.py).

Covers every required behaviour: candle detection, one-analysis-per-candle
duplicate prevention (incl. across restarts and concurrent ticks), the active
signals gate (PENDING_ENTRY / OPEN / terminal), WAIT, LONG/SHORT -> PENDING_ENTRY,
data/indicator/context/LLM/database failure safety, timezone enforcement, and
the interaction with the independent TP/SL monitor.
"""

from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from binance.market_data import Candle
from database.database import (
    CandleLogRepository,
    Database,
    SchedulerStateRepository,
    SignalRepository,
)
from database.models import (
    STATUS_CANCELLED,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    SignalValidationError,
)
from signal_engine import (
    MonitorOutcome,
    OneHourScheduler,
    SchedulerOutcome,
    SignalEngine,
    SignalMonitor,
    SignalState,
)
from signal_engine.analysis import SignalAnalysisError
from signal_engine.context import AnalysisContextError
from signal_engine.scheduler import DEFAULT_WINDOW_CANDLES, SCHEDULER_INTERVAL
from tests.signal_engine_test_helpers import (
    DEFAULT_START_MS,
    INTERVAL_MS,
    TempSignalDb,
    make_analysis,
    make_candles,
)

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def latest_ms(candles) -> int:
    return candles[-1].timestamp


class FakeSchedulerData:
    """Deterministic replacement for ``fetch_closed_klines`` (no network)."""

    def __init__(self, *candles, error=None) -> None:
        self.candles = list(candles)
        self.error = error
        self.calls: list[tuple[str, str, int, int | None]] = []

    def fetch_closed_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        end_time_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[Candle]:
        self.calls.append((symbol, interval, limit, now_ms))
        if self.error is not None:
            raise self.error
        return [c for c in self.candles if c.is_closed]


def make_analyzer(decision="WAIT", calls=None):
    """Build an analyzer recording invocations; ``decision`` may be an exception."""

    def analyzer(candles, indicators):
        calls.append((list(candles), getattr(indicators, "shape", None)))
        if isinstance(decision, Exception):
            raise decision
        return make_analysis(decision)

    return analyzer


def _monitor_candle(high: float, low: float) -> Candle:
    mid = (high + low) / 2.0
    return Candle(
        timestamp=1_000_000,
        open=mid,
        high=high,
        low=low,
        close=mid,
        volume=1.0,
        close_time=1_059_999,
        is_closed=True,
    )


class FakeMonitorData:
    """Replacement for the monitor's ``fetch_klines``."""

    def __init__(self, *candles: Candle) -> None:
        self.candles = list(candles)

    def fetch_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
        return list(self.candles)


class SchedulerTestCase(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.candles = make_candles(DEFAULT_WINDOW_CANDLES)
        self.md = FakeSchedulerData(*self.candles)

    def make_scheduler(self, md=None, analyzer=None, **kwargs):
        kwargs.setdefault("market_data", md if md is not None else self.md)
        kwargs.setdefault("engine", self.harness.engine)
        kwargs.setdefault("state", self.harness.state)
        return OneHourScheduler(self.harness.repository, analyzer=analyzer, **kwargs)


# ── A. Candle detection ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestCandleDetection(SchedulerTestCase):
    def test_same_candle_is_already_processed(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        first = sched.tick(now=NOW)
        self.assertEqual(first.outcome, SchedulerOutcome.WAIT)
        second = sched.tick(now=NOW)
        self.assertEqual(second.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(second.candle_timestamp, latest_ms(self.candles))
        self.assertEqual(len(calls), 1)

    def test_market_behind_marker_is_no_new_candle(self):
        future = latest_ms(self.candles) + INTERVAL_MS
        SchedulerStateRepository(self.harness.db).record_processed_candle(
            "BTCUSDT", "1h", future, "2026-09-10T12:00:00.000Z"
        )
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.NO_NEW_CANDLE)
        self.assertEqual(len(calls), 0)

    def test_new_candle_after_marker_is_analyzed(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        advanced = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES + 1))
        sched2 = self.make_scheduler(md=advanced, analyzer=make_analyzer("WAIT", calls))
        result = sched2.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(result.candle_timestamp, latest_ms(advanced.candles))
        self.assertEqual(len(calls), 2)

    def test_forming_1h_candle_never_analyzed(self):
        raw_rows = []
        for i, (open_, close, high, low, vol) in enumerate(
            _rows_for(make_candles(DEFAULT_WINDOW_CANDLES))
        ):
            ts = DEFAULT_START_MS + i * INTERVAL_MS
            raw_rows.append(
                [
                    ts,
                    open_,
                    high,
                    low,
                    close,
                    vol,
                    ts + INTERVAL_MS - 1,
                    0.0,
                    1,
                    0.0,
                    0.0,
                    "0",
                ]
            )
        fresh_ts = DEFAULT_START_MS + DEFAULT_WINDOW_CANDLES * INTERVAL_MS
        raw_rows.append(
            [
                fresh_ts,
                62000.0,
                68000.0,
                55000.0,
                62000.0,
                100.0,
                fresh_ts + INTERVAL_MS - 1,
                0.0,
                1,
                0.0,
                0.0,
                "0",
            ]
        )
        # now = the fresh candle's close time exactly -> it is still forming.
        now_ms = fresh_ts + INTERVAL_MS - 1
        calls: list = []

        class ParsingData:
            def fetch_closed_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
                from binance.market_data import parse_klines

                candles = parse_klines(raw_rows, now_ms)
                return [c for c in candles if c.is_closed]

        sched = self.make_scheduler(md=ParsingData(), analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc))
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(len(calls), 1)
        analyzed_candles, _ = calls[0]
        self.assertEqual(len(analyzed_candles), DEFAULT_WINDOW_CANDLES)
        self.assertTrue(all(c.is_closed for c in analyzed_candles))

    def test_forming_candle_excluded_at_detection(self):
        forming = Candle(
            timestamp=latest_ms(self.candles) + INTERVAL_MS,
            open=62000.0,
            high=68000.0,
            low=55000.0,
            close=62000.0,
            volume=100.0,
            close_time=latest_ms(self.candles) + 2 * INTERVAL_MS - 1,
            is_closed=False,
        )
        md = FakeSchedulerData(forming)
        calls: list = []
        sched = self.make_scheduler(md=md, analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
        self.assertEqual(len(calls), 0)


def _rows_for(candles):
    for c in candles:
        yield c.open, c.close, c.high, c.low, c.volume


# ── B. Active-signal gate ────────────────────────────────────────────────────


@pytest.mark.unit
class TestActiveGate(SchedulerTestCase):
    def _seed_signal_at(self, decision="LONG", candle_index=-2, prices=None):
        """Persist a signal whose analysis candle is OLDER than the newest.

        When ``prices`` is given it overrides ``entry_price``/``stop_loss``/
        ``take_profit`` (used for SHORT so the ordering rule holds).
        """
        ts = self.candles[candle_index].timestamp
        kwargs: dict = {"market_timestamp": iso_from_ms(ts)}
        if prices:
            kwargs.update(prices)
        return self.harness.engine.process(make_analysis(decision, **kwargs)).signal

    def _seed_and_close(self, decision, candle_index, to_status, price):
        """Create a signal, promote PENDING_ENTRY -> OPEN, then close it."""
        signal = self._seed_signal_at(decision, candle_index)
        self.harness.state.transition(signal.id, STATUS_OPEN)
        return self.harness.state.transition(signal.id, to_status, close_price=price)

    def test_no_active_allows_analysis(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(len(calls), 1)

    def test_pending_entry_blocks_ai(self):
        signal = self._seed_signal_at()
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(result.signal.id, signal.id)
        self.assertEqual(len(calls), 0)

    def test_open_blocks_ai(self):
        signal = self._seed_signal_at()
        self.harness.state.transition(signal.id, STATUS_OPEN)
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(len(calls), 0)

    def test_active_gate_precedes_analysis_fetch(self):
        signal = self._seed_signal_at()
        self.harness.state.transition(signal.id, STATUS_OPEN)

        class FailOnSecondFetch:
            def __init__(self) -> None:
                self.calls = 0

            def fetch_closed_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
                self.calls += 1
                if self.calls == 1:
                    return [c for c in self.candles if c.is_closed]
                raise RuntimeError("analysis fetch must not happen")

        fail_md = FailOnSecondFetch()
        fail_md.candles = self.candles
        calls: list = []
        sched = self.make_scheduler(md=fail_md, analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(fail_md.calls, 1)
        self.assertEqual(len(calls), 0)

    def test_tp_hit_then_next_candle_allows_ai(self):
        signal = self._seed_and_close("LONG", -2, STATUS_TP_HIT, "64000")
        self.assertEqual(self.harness.repository.get_signal(signal.id).status, STATUS_TP_HIT)
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(len(calls), 1)

    def test_sl_hit_then_next_candle_allows_ai(self):
        signal = self._seed_and_close("LONG", -2, STATUS_SL_HIT, "60000")
        self.assertEqual(self.harness.repository.get_signal(signal.id).status, STATUS_SL_HIT)
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(len(calls), 1)

    def test_seed_from_signal_avoids_reanalysis_on_upgrade(self):
        # Legacy DB: a signal exists but no scheduler_state row. The scheduler
        # must seed its marker from the signal and NOT re-analyze that candle.
        signal = self._seed_and_close("LONG", -1, STATUS_SL_HIT, "60000")
        self.assertEqual(self.harness.repository.get_signal(signal.id).status, STATUS_SL_HIT)
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(len(calls), 0)


# ── C. WAIT / LONG / SHORT ───────────────────────────────────────────────────


@pytest.mark.unit
class TestDecisions(SchedulerTestCase):
    def test_wait_creates_no_signal_and_marks_processed(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertIsNone(result.signal)
        self.assertEqual(self.harness.repository.list_signals(), [])
        self.assertEqual(
            SchedulerStateRepository(self.harness.db).last_processed_candle(
                "BTCUSDT", "1h"
            ),
            latest_ms(self.candles),
        )

    def test_long_persists_pending_entry_with_metadata(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        signal = self.harness.repository.get_active_signal()
        self.assertIsNotNone(signal)
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)
        self.assertEqual(signal.direction, "LONG")
        self.assertEqual(signal.entry, Decimal("61000"))
        self.assertEqual(signal.stop_loss, Decimal("60000"))
        self.assertEqual(signal.take_profit, Decimal("64000"))
        self.assertEqual(signal.confidence, 80)
        self.assertEqual(signal.market_timestamp, "2026-09-10T00:00:00.000Z")
        self.assertEqual(signal.candle_close_price, Decimal("61050.5"))
        self.assertIsNone(signal.opened_at)
        self.assertEqual(result.candle_timestamp, latest_ms(self.candles))

    def test_short_persists_pending_entry(self):
        sched = self.make_scheduler(
            analyzer=lambda c, i: make_analysis(
                "SHORT",
                entry_price=59000.0,
                stop_loss=60000.0,
                take_profit=57700.0,  # RR 1.3 >= Phase B minimum 1.2
            )
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        signal = self.harness.repository.get_active_signal()
        self.assertEqual(signal.direction, "SHORT")
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)


# ── D. Data / indicator / context / LLM / DB failures ───────────────────────


@pytest.mark.unit
class TestFailureSafety(SchedulerTestCase):
    def test_detection_failure_is_data_unavailable_no_ai(self):
        from binance.client import BinanceConnectionError

        md = FakeSchedulerData(error=BinanceConnectionError("timeout"))
        calls: list = []
        sched = self.make_scheduler(md=md, analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
        self.assertEqual(len(calls), 0)
        self.assertIsNone(
            SchedulerStateRepository(self.harness.db).last_processed_candle("BTCUSDT", "1h")
        )

    def test_short_window_is_data_unavailable(self):
        few = make_candles(50)
        sched = self.make_scheduler(md=FakeSchedulerData(*few))
        calls: list = []
        calls_holder = calls
        sched = self.make_scheduler(
            md=FakeSchedulerData(*few), analyzer=make_analyzer("LONG", calls_holder)
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
        self.assertEqual(len(calls_holder), 0)

    def test_indicator_error_is_data_unavailable(self):
        from binance.indicators import EmptyIndicatorDataError

        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        with patch(
            "signal_engine.scheduler.compute_indicator_matrix",
            side_effect=EmptyIndicatorDataError("boom"),
        ):
            result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
        self.assertEqual(len(calls), 0)

    def test_context_error_is_data_unavailable(self):
        calls: list = []
        sched = self.make_scheduler(
            analyzer=make_analyzer(AnalysisContextError("no candles"), calls)
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
        self.assertEqual(len(calls), 1)

    def test_llm_error_is_error_and_not_marked(self):
        calls: list = []
        sched = self.make_scheduler(
            analyzer=make_analyzer(SignalAnalysisError("LLM failed"), calls)
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ERROR)
        self.assertIsNone(result.signal)
        self.assertIsNone(
            SchedulerStateRepository(self.harness.db).last_processed_candle("BTCUSDT", "1h")
        )
        retry = sched.tick(now=NOW)
        self.assertEqual(retry.outcome, SchedulerOutcome.ERROR)
        self.assertEqual(len(calls), 2)

    def test_breaker_opens_after_n_consecutive_errors(self):
        # Phase E: after N ERRORs the breaker skips LLM work (no new calls)
        # until the cooldown expires.
        calls: list = []
        sched = self.make_scheduler(
            analyzer=make_analyzer(SignalAnalysisError("provider 500"), calls)
        )
        with patch.dict(
            os.environ,
            {"BTCUSDT_ERROR_BREAKER_N": "3", "BTCUSDT_ERROR_BREAKER_COOLDOWN_S": "600"},
        ):
            for _ in range(3):
                self.assertEqual(
                    sched.tick(now=NOW).outcome, SchedulerOutcome.ERROR
                )
            self.assertEqual(len(calls), 3)
            skipped = sched.tick(now=NOW)
            self.assertEqual(skipped.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
            self.assertIn("circuit breaker", skipped.message)
            self.assertEqual(len(calls), 3)

    def test_breaker_success_resets_streak(self):
        outcomes = [
            SignalAnalysisError("flaky"),
            SignalAnalysisError("flaky"),
            "WAIT",
            SignalAnalysisError("flaky"),
            SignalAnalysisError("flaky"),
        ]
        calls: list = []
        states = list(outcomes)

        def flapping(candles, indicators):
            calls.append((candles, indicators))
            decision = states.pop(0)
            if isinstance(decision, Exception):
                raise decision
            return make_analysis(decision)

        from tests.signal_engine_test_helpers import make_candles as _fresh_candles

        sched = self.make_scheduler(analyzer=flapping)
        with patch.dict(os.environ, {"BTCUSDT_ERROR_BREAKER_N": "3"}):
            for step, _ in enumerate(outcomes):
                # Each tick sees a newer candle (ERROR/WAIT never share one).
                sched.market_data = FakeSchedulerData(
                    *_fresh_candles(DEFAULT_WINDOW_CANDLES + step)
                )
                result = sched.tick(now=NOW)
                expected = (
                    SchedulerOutcome.ERROR
                    if isinstance(outcomes[step], Exception)
                    else SchedulerOutcome.WAIT
                )
                self.assertEqual(result.outcome, expected)
            # Streak reset by WAIT: no breaker trip, all 5 analyses ran.
            self.assertEqual(len(calls), 5)

    def test_breaker_half_open_retries_after_cooldown(self):
        calls: list = []
        sched = self.make_scheduler(
            analyzer=make_analyzer(SignalAnalysisError("down"), calls)
        )
        env = {"BTCUSDT_ERROR_BREAKER_N": "2", "BTCUSDT_ERROR_BREAKER_COOLDOWN_S": "600"}
        with patch.dict(os.environ, env):
            t0 = NOW
            self.assertEqual(sched.tick(now=t0).outcome, SchedulerOutcome.ERROR)
            t1 = t0 + timedelta(seconds=30)
            self.assertEqual(sched.tick(now=t1).outcome, SchedulerOutcome.ERROR)
            # Breaker open until t1+600s: skipped tick makes no LLM call.
            t2 = t0 + timedelta(seconds=40)
            skipped = sched.tick(now=t2)
            self.assertEqual(skipped.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
            self.assertIn("circuit breaker", skipped.message)
            self.assertEqual(len(calls), 2)
            # Past cooldown: half-open tick calls the LLM again.
            t3 = t0 + timedelta(seconds=640)
            self.assertEqual(sched.tick(now=t3).outcome, SchedulerOutcome.ERROR)
            self.assertEqual(len(calls), 3)

    def test_breaker_disabled_at_zero(self):
        calls: list = []
        sched = self.make_scheduler(
            analyzer=make_analyzer(SignalAnalysisError("down"), calls)
        )
        with patch.dict(os.environ, {"BTCUSDT_ERROR_BREAKER_N": "0"}):
            for _ in range(6):
                self.assertEqual(
                    sched.tick(now=NOW).outcome, SchedulerOutcome.ERROR
                )
            self.assertEqual(len(calls), 6)

    def test_breaker_settings_fallback(self):
        from signal_engine.scheduler import _breaker_settings

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_breaker_settings(), (5, 600.0))
        with patch.dict(
            os.environ,
            {"BTCUSDT_ERROR_BREAKER_N": "3", "BTCUSDT_ERROR_BREAKER_COOLDOWN_S": "60"},
        ):
            self.assertEqual(_breaker_settings(), (3, 60.0))
        with patch.dict(os.environ, {"BTCUSDT_ERROR_BREAKER_N": "0"}):
            self.assertEqual(_breaker_settings(), (0, 0.0))
        with patch.dict(os.environ, {"BTCUSDT_ERROR_BREAKER_N": "many"}):
            self.assertEqual(_breaker_settings(), (5, 600.0))

    def test_rejected_analysis_is_marked_no_retry(self):
        # Phase A token guard: a validation failure records the candle as
        # processed so the same candle never burns another LLM call.
        calls: list = []

        def bad_analyzer(candles, indicators):
            calls.append((candles, indicators))
            return make_analysis(
                "LONG", entry_price=59000.0, stop_loss=60000.0, take_profit=64000.0
            )

        sched = self.make_scheduler(analyzer=bad_analyzer)
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.REJECTED)
        self.assertIsNone(result.signal)
        self.assertFalse(self.harness.repository.list_signals())
        self.assertEqual(
            SchedulerStateRepository(self.harness.db).last_processed_candle(
                "BTCUSDT", "1h"
            ),
            latest_ms(self.candles),
        )
        retry = sched.tick(now=NOW)
        self.assertEqual(retry.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(len(calls), 1)

    def test_rejected_marker_write_failure_is_error(self):
        def bad_analyzer(candles, indicators):
            return make_analysis(
                "LONG", entry_price=59000.0, stop_loss=60000.0, take_profit=64000.0
            )

        sched = self.make_scheduler(analyzer=bad_analyzer)
        with patch.object(
            SchedulerStateRepository,
            "record_processed_candle",
            side_effect=RuntimeError("disk readonly"),
        ):
            result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ERROR)

    def test_rate_limited_is_marked_no_retry(self):
        # A provider 429 must skip the candle instead of retrying every poll.
        from signal_engine.analysis import RateLimitedError

        calls: list = []
        sched = self.make_scheduler(
            analyzer=make_analyzer(
                RateLimitedError("LLM rate limit: Error code 429"), calls
            )
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.RATE_LIMITED)
        self.assertIsNone(result.signal)
        self.assertEqual(
            SchedulerStateRepository(self.harness.db).last_processed_candle(
                "BTCUSDT", "1h"
            ),
            latest_ms(self.candles),
        )
        retry = sched.tick(now=NOW)
        self.assertEqual(retry.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(len(calls), 1)

    def test_rate_limited_marker_write_failure_is_error(self):
        from signal_engine.analysis import RateLimitedError

        sched = self.make_scheduler(
            analyzer=make_analyzer(RateLimitedError("429"), calls=[])
        )
        with patch.object(
            SchedulerStateRepository,
            "record_processed_candle",
            side_effect=RuntimeError("disk readonly"),
        ):
            result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ERROR)

    def test_wait_marker_write_failure_is_error_not_wait(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        with patch.object(
            SchedulerStateRepository,
            "record_processed_candle",
            side_effect=RuntimeError("disk readonly"),
        ):
            result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ERROR)
        self.assertNotEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(calls), 1)

    def test_created_marker_write_failure_still_reports_created(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        with patch.object(
            SchedulerStateRepository,
            "record_processed_candle",
            side_effect=RuntimeError("disk readonly"),
        ):
            result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertIsNotNone(result.signal)
        self.assertEqual(len(self.harness.repository.list_signals()), 1)


# ── E. Restart safety ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRestartSafety(SchedulerTestCase):
    def _reopen(self):
        reopened = Database(self.harness.path)
        reopened.initialize()
        repo2 = SignalRepository(reopened)
        return reopened, repo2

    def test_restart_does_not_reanalyze_created_candle(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.CREATED)
        reopened, repo2 = self._reopen()
        sched2 = OneHourScheduler(
            repo2,
            market_data=self.md,
            engine=SignalEngine(repo2, SignalState(repo2)),
            analyzer=make_analyzer("LONG", calls),
        )
        result = sched2.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(len(calls), 1)
        reopened.connect().close()

    def test_restart_does_not_reanalyze_wait_candle(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        reopened, repo2 = self._reopen()
        sched2 = OneHourScheduler(
            repo2,
            market_data=self.md,
            engine=SignalEngine(repo2, SignalState(repo2)),
            analyzer=make_analyzer("WAIT", calls),
        )
        result = sched2.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(len(calls), 1)
        reopened.connect().close()

    def test_restart_then_new_candle_analyzes(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        reopened, repo2 = self._reopen()
        md2 = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES + 1))
        sched2 = OneHourScheduler(
            repo2,
            market_data=md2,
            engine=SignalEngine(repo2, SignalState(repo2)),
            analyzer=make_analyzer("LONG", calls),
        )
        result = sched2.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(repo2.list_signals()), 1)
        reopened.connect().close()


# ── F. Concurrency ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestConcurrency(SchedulerTestCase):
    def test_two_concurrent_ticks_same_candle_one_analysis(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        barrier = threading.Barrier(2, timeout=10)

        def tick(_i):
            barrier.wait(timeout=10)
            return sched.tick(now=NOW)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(tick, range(2)))
        outcomes = sorted(r.outcome for r in results)
        self.assertEqual(
            outcomes, [SchedulerOutcome.ALREADY_PROCESSED, SchedulerOutcome.CREATED]
        )
        self.assertEqual(len(calls), 1)
        signals = self.harness.repository.list_signals()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].status, STATUS_PENDING_ENTRY)


# ── G. Monitor interaction ───────────────────────────────────────────────────


@pytest.mark.unit
class TestMonitorInteraction(SchedulerTestCase):
    def test_pending_to_open_blocks_then_close_unlocks(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        created = sched.tick(now=NOW)
        self.assertEqual(created.outcome, SchedulerOutcome.CREATED)
        signal = created.signal
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)

        # Independent monitor promotes PENDING_ENTRY -> OPEN at the entry touch.
        promoted = SignalMonitor(
            self.harness.repository, FakeMonitorData(_monitor_candle(61100, 60700))
        ).poll()
        self.assertEqual(promoted.outcome, MonitorOutcome.ENTRY_HIT)

        # The scheduler must now be locked even on the next new candle.
        md_next = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES + 1))
        calls.clear()
        sched2 = self.make_scheduler(md=md_next, analyzer=make_analyzer("LONG", calls))
        blocked = sched2.tick(now=NOW)
        self.assertEqual(blocked.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(len(calls), 0)

        # Monitor closes OPEN at TP.
        closed = SignalMonitor(
            self.harness.repository, FakeMonitorData(_monitor_candle(65000, 60500))
        ).poll()
        self.assertEqual(closed.outcome, MonitorOutcome.TP_HIT)

        # A further new candle may now be analyzed again.
        md_final = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES + 2))
        sched3 = self.make_scheduler(md=md_final, analyzer=make_analyzer("LONG", calls))
        unlocked = sched3.tick(now=NOW)
        self.assertEqual(unlocked.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(sched3.repository.list_signals()), 2)


# ── H. API / timezone / init contract ───────────────────────────────────────


@pytest.mark.unit
class TestSchedulerContract(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)

    def test_naive_now_rejected(self):
        sched = OneHourScheduler(self.harness.repository, market_data=FakeSchedulerData())
        with self.assertRaises(ValueError):
            sched.tick(now=datetime(2026, 9, 10, 12, 0, 0))

    def test_naive_as_epoch_ms_rejected(self):
        from signal_engine.scheduler import as_epoch_ms

        with self.assertRaises(ValueError):
            as_epoch_ms(datetime(2026, 9, 10, 12, 0, 0))

    def test_invalid_limits_rejected(self):
        from binance.indicators import MIN_REQUIRED_CANDLES

        for bad in (MIN_REQUIRED_CANDLES - 1, 0, "200", 200.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                OneHourScheduler(
                    self.harness.repository,
                    market_data=FakeSchedulerData(),
                    candle_limit=bad,
                )
        for bad in (0, -1, None, "5", 5.0):
            with self.subTest(detection=bad), self.assertRaises(ValueError):
                OneHourScheduler(
                    self.harness.repository,
                    market_data=FakeSchedulerData(),
                    detection_limit=bad,
                )

    def test_scheduler_uses_1h_fetch_closed_klines_only(self):
        md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
        calls: list = []
        sched = OneHourScheduler(self.harness.repository, market_data=md, state=self.harness.state, engine=self.harness.engine, analyzer=make_analyzer("WAIT", calls))
        sched.tick(now=NOW)
        self.assertTrue(md.calls)
        for symbol, interval, _, _ in md.calls:
            self.assertEqual(symbol, "BTCUSDT")
            # 1H analysis candles plus the Phase P2 4H bias fetch (closed
            # klines only, never orders); nothing else may hit market data.
            self.assertIn(interval, (SCHEDULER_INTERVAL, "4h"))

    def test_no_real_client_and_no_order_paths(self):
        with patch("signal_engine.scheduler.BinanceMarketData") as cls:
            md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
            calls: list = []
            sched = OneHourScheduler(self.harness.repository, market_data=md, state=self.harness.state, engine=self.harness.engine, analyzer=make_analyzer("WAIT", calls))
            sched.tick(now=NOW)
        cls.assert_not_called()
        import signal_engine.scheduler as mod

        with open(mod.__file__) as f:
            src = f.read()
        for needle in ("/fapi/v1/order", "/fapi/v1/balance", "client.get("):
            self.assertNotIn(needle, src)

    def test_run_forever_interval_validated(self):
        sched = OneHourScheduler(self.harness.repository, market_data=FakeSchedulerData())
        with self.assertRaises(ValueError):
            sched.run_forever(interval=0)
        with self.assertRaises(ValueError):
            sched.run_forever(interval=-1)


@pytest.mark.unit
class TestSymbolResolution(SchedulerTestCase):
    def test_string_symbol_is_normalized(self):
        sched = self.make_scheduler(symbol="ethusdt", analyzer=make_analyzer("WAIT", []))
        self.assertEqual(sched.symbol, "ETHUSDT")

    def test_malformed_or_empty_symbol_rejected(self):
        for bad in ("SOL/USD", "", "  "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make_scheduler(symbol=bad, analyzer=make_analyzer("WAIT", []))

    def test_callable_symbol_resolved_per_tick(self):
        calls: list = []
        md = FakeSchedulerData(*self.candles)
        sched = self.make_scheduler(
            md=md, symbol="ETHUSDT", analyzer=make_analyzer("WAIT", calls)
        )
        self.assertEqual(sched.symbol, "ETHUSDT")
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(md.calls[0][0], "ETHUSDT")
        self.assertEqual(md.calls[-1][0], "ETHUSDT")

    def test_processed_markers_are_per_symbol(self):
        calls: list = []
        btc = self.make_scheduler(analyzer=make_analyzer("WAIT", calls))
        self.assertEqual(btc.tick(now=NOW).outcome, SchedulerOutcome.WAIT)

        eth_calls: list = []
        eth = self.make_scheduler(symbol="ETHUSDT", analyzer=make_analyzer("WAIT", eth_calls))
        result = eth.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(eth_calls), 1)

    def test_cold_start_seed_is_scoped_to_own_symbol(self):
        # A closed BTCUSDT signal must not seed the ETHUSDT marker on cold
        # start (regression: previously seeded from the latest signal of ANY
        # symbol). The active-signal gate is intentionally global.
        repo = SignalRepository(self.harness.db)
        signal = repo.create_signal(
            "BTCUSDT", "1h", "LONG", "100", "99", "101",
            market_timestamp=iso_from_ms(latest_ms(self.candles)),
        )
        repo.transition_signal(signal.id, STATUS_CANCELLED)
        calls: list = []
        eth = self.make_scheduler(symbol="ETHUSDT", analyzer=make_analyzer("WAIT", calls))
        result = eth.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(calls), 1)
        self.assertEqual(eth.tick(now=NOW).outcome, SchedulerOutcome.ALREADY_PROCESSED)

    def test_switch_symbol_resolves_fresh_between_ticks(self):
        # A new callable is picked up on the next tick (daemon symbol switch
        # without restart).
        md = FakeSchedulerData(*self.candles)
        current = {"symbol": "BTCUSDT"}

        def symbol():
            return current["symbol"]

        calls: list = []
        sched = self.make_scheduler(md=md, symbol=symbol, analyzer=make_analyzer("WAIT", calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        self.assertEqual(md.calls[-1][0], "BTCUSDT")

        current["symbol"] = "ETHUSDT"
        advanced = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES + 1))
        sched.market_data = advanced
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        self.assertEqual(advanced.calls[-1][0], "ETHUSDT")


@pytest.mark.unit
class TestSchedulerStateRepository(unittest.TestCase):
    def test_record_and_read_roundtrip(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        store = SchedulerStateRepository(harness.db)
        self.assertIsNone(store.last_processed_candle("BTCUSDT", "1h"))
        store.record_processed_candle("btcusdt", "1h", 1234567890, "2026-09-10T12:00:00.000Z")
        self.assertEqual(store.last_processed_candle("BTCUSDT", "1h"), 1234567890)
        # Symbol/timeframe normalization: case- and whitespace-insensitive keys.
        self.assertEqual(store.last_processed_candle("BTCUSDT", "1h"), 1234567890)
        store.record_processed_candle("BTCUSDT", "1h", 1234567899, "2026-09-10T12:01:00.000Z")
        self.assertEqual(store.last_processed_candle("BTCUSDT", "1h"), 1234567899)

    def test_validation(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        store = SchedulerStateRepository(harness.db)
        for bad in (0, -1, True, 1.5, "5"):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                store.record_processed_candle("BTCUSDT", "1h", bad, "t")
        with self.assertRaises(SignalValidationError):
            store.record_processed_candle("BTCUSDT", "1h", 1, "")


# ── F. Per-candle daemon activity log (Phase 16) ─────────────────────────────


@pytest.mark.unit
class TestCandleLog(SchedulerTestCase):
    """One diagnostic row per analysed candle: WAIT/CREATED/ERROR/BLOCKED are
    recorded; NO_NEW_CANDLE / ALREADY_PROCESSED are not."""

    def build(self, **kwargs):
        kwargs.setdefault("candle_log", CandleLogRepository(self.harness.db))
        return self.make_scheduler(**kwargs)

    def rows(self):
        return CandleLogRepository(self.harness.db).list(limit=100)

    def test_wait_is_logged_with_decision_confidence_and_closed_at(self):
        calls: list = []
        sched = self.build(analyzer=make_analyzer("WAIT", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertEqual(entry.outcome, "WAIT")
        self.assertEqual(entry.decision, "WAIT")
        self.assertEqual(entry.confidence, 80)
        self.assertEqual(entry.candle_timestamp_ms, latest_ms(self.candles))
        self.assertEqual(entry.symbol, "BTCUSDT")
        self.assertEqual(entry.timeframe, "1h")
        self.assertTrue(entry.recorded_at)
        self.assertTrue(entry.closed_at)
        self.assertIsNotNone(entry.reasoning)
        self.assertIsNone(entry.signal_id)
        self.assertIsNone(entry.error_notes)

    def test_quota_recorded_from_analysis(self):
        import dataclasses

        def analyzer(candles, indicators):
            base = make_analysis("WAIT")
            return dataclasses.replace(
                base, llm_calls=5, prompt_tokens=10500,
                completion_tokens=1800, total_tokens=12300,
            )

        sched = self.build(analyzer=analyzer)
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        entry = self.rows()[0]
        self.assertEqual(entry.llm_calls, 5)
        self.assertEqual(entry.prompt_tokens, 10500)
        self.assertEqual(entry.completion_tokens, 1800)
        self.assertEqual(entry.total_tokens, 12300)
        self.assertFalse(entry.tokens_estimated)

    def test_quota_estimated_flag_recorded(self):
        import dataclasses

        def analyzer(candles, indicators):
            base = make_analysis("WAIT")
            return dataclasses.replace(
                base, llm_calls=1, prompt_tokens=1600,
                completion_tokens=120, total_tokens=1720,
                tokens_estimated=True,
            )

        sched = self.build(analyzer=analyzer)
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        entry = self.rows()[0]
        self.assertEqual(entry.prompt_tokens, 1600)
        self.assertTrue(entry.tokens_estimated)

    def test_quota_defaults_single_call_without_tokens(self):
        calls: list = []
        sched = self.build(analyzer=make_analyzer("WAIT", calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        entry = self.rows()[0]
        self.assertEqual(entry.llm_calls, 1)
        self.assertIsNone(entry.prompt_tokens)
        self.assertIsNone(entry.total_tokens)

    def test_error_row_records_zero_calls(self):
        calls: list = []
        sched = self.build(
            analyzer=make_analyzer(SignalAnalysisError("LLM failed"), calls)
        )
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.ERROR)
        entry = self.rows()[0]
        self.assertEqual(entry.llm_calls, 0)
        self.assertIsNone(entry.prompt_tokens)

    def test_created_is_logged_with_signal_id_and_prices(self):
        calls: list = []
        sched = self.build(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertEqual(entry.outcome, "CREATED")
        self.assertEqual(entry.decision, "LONG")
        self.assertEqual(entry.signal_id, result.signal.id)
        self.assertEqual(entry.entry, "61000.0")
        self.assertEqual(entry.stop_loss, "60000.0")
        self.assertEqual(entry.take_profit, "64000.0")
        self.assertIsNotNone(entry.indicators_json)
        import json
        snapshot = json.loads(entry.indicators_json)
        self.assertIn("close", snapshot)

    def test_active_signal_block_is_logged_without_calling_ai(self):
        repo = SignalRepository(self.harness.db)
        # A signal opened on the PREVIOUS candle, still active; the newest
        # candle N closes and must be skipped without any AI call.
        repo.create_signal(
            "BTCUSDT", "1h", "LONG", "61000", "60000", "64000",
            market_timestamp=iso_from_ms(latest_ms(self.candles) - INTERVAL_MS),
        )
        calls: list = []
        sched = self.build(analyzer=make_analyzer("LONG", calls))
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(len(calls), 0)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertEqual(entry.outcome, "BLOCKED_ACTIVE_SIGNAL")
        self.assertEqual(entry.decision, "NONE")
        self.assertIsNotNone(entry.error_notes)

    def test_data_unavailable_is_logged(self):
        from binance.client import BinanceConnectionError

        calls: list = []
        sched = self.build(
            md=FakeSchedulerData(error=BinanceConnectionError("timeout")),
            analyzer=make_analyzer("LONG", calls),
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.DATA_UNAVAILABLE)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertEqual(entry.outcome, "DATA_UNAVAILABLE")
        self.assertEqual(entry.decision, "NONE")
        self.assertIn("BinanceConnectionError", entry.error_notes)

    def test_ai_error_is_logged(self):
        calls: list = []
        sched = self.build(
            analyzer=make_analyzer(SignalAnalysisError("LLM failed"), calls)
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.ERROR)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].outcome, "ERROR")
        self.assertEqual(rows[0].decision, "NONE")
        self.assertIn("LLM failed", rows[0].error_notes)

    def test_retry_of_failed_candle_keeps_one_row(self):
        calls: list = []
        sched = self.build(analyzer=make_analyzer(SignalAnalysisError("LLM failed"), calls))
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.ERROR)
        # Also run the scheduler without a candle log to prove the guard.
        plain = self.make_scheduler(analyzer=make_analyzer(SignalAnalysisError("x"), calls))
        self.assertEqual(plain.tick(now=NOW).outcome, SchedulerOutcome.ERROR)
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.ERROR)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].outcome, "ERROR")

    def test_no_new_candle_and_already_processed_are_not_logged(self):
        calls: list = []
        sched = self.build(analyzer=make_analyzer("WAIT", calls))
        sched.tick(now=NOW)
        # Same candle -> ALREADY_PROCESSED, no extra row.
        sched.tick(now=NOW)
        # A marker ahead of the market -> NO_NEW_CANDLE, no extra row.
        future = latest_ms(self.candles) + INTERVAL_MS
        SchedulerStateRepository(self.harness.db).record_processed_candle(
            "BTCUSDT", "1h", future, "2026-09-10T12:00:00.000Z"
        )
        sched.tick(now=NOW)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].outcome, "WAIT")

    def test_candle_log_disabled_by_default(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=make_analyzer("LONG", calls))
        sched.tick(now=NOW)
        rows = self.rows()
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
