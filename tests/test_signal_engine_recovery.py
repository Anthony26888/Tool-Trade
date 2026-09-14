"""Phase 9 unit tests for startup recovery (signal_engine/recovery.py).

Covers every scenario: empty DB, OPEN LONG/SHORT recovery, repeated restarts,
closed states staying closed, immutability, no-AI guarantees, monitor resuming
after recovery, TP/SL/AMBIGUOUS/data-unavailable outcomes, Binance safety,
persistence across reopen, concurrent recovery, malformed/inconsistent DB, and
clear structured errors without leaking raw exceptions.
"""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from binance.market_data import BinanceError, Candle
from database.database import Database, SignalRepository
from database.models import (
    STATUS_CANCELLED,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    Signal,
)
from signal_engine import (
    MonitorOutcome,
    RecoveryOutcome,
    RecoveryService,
)
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis


def _candle(high, low, *, closed: bool = True, ts: int = 1_000_000) -> Candle:
    """Build a fake candle with a controlled closed flag (no network)."""
    high_f, low_f = float(high), float(low)
    mid = (high_f + low_f) / 2.0
    close_time = ts + (59_999 if closed else 60_000)
    return Candle(
        timestamp=ts,
        open=mid,
        high=high_f,
        low=low_f,
        close=mid,
        volume=1.0,
        close_time=close_time,
        is_closed=closed,
    )


class FakeMarketData:
    """Deterministic replacement for BinanceMarketData (never touches network)."""

    def __init__(self, *candles: Candle) -> None:
        self.candles = list(candles)
        self.error: Exception | None = None

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        end_time_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[Candle]:
        if self.error is not None:
            raise self.error
        return list(self.candles)


class RecoveryTestCase(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)

    def recover(self):
        return RecoveryService(self.harness.repository).recover()

    def reopen(self) -> SignalRepository:
        """Simulate a full process restart: new DB + repository on the same file."""
        db = Database(self.harness.path)
        db.initialize()
        return SignalRepository(db)

    def _create_long(self, **overrides):
        result = self.harness.engine.process(make_analysis("LONG", **overrides))
        self.assertEqual(result.outcome.value, "CREATED")
        return result.signal

    def _create_short(self, **overrides):
        defaults = {"entry_price": 59000.0, "stop_loss": 60000.0, "take_profit": 58000.0}
        defaults.update(overrides)
        result = self.harness.engine.process(make_analysis("SHORT", **defaults))
        self.assertEqual(result.outcome.value, "CREATED")
        return result.signal

    def _promote_to_open(self, signal) -> Signal:
        return self.harness.state.transition(signal.id, STATUS_OPEN)

    def _open_long(self, **overrides) -> Signal:
        """Create and promote to OPEN (the entry was touched)."""
        return self._promote_to_open(self._create_long(**overrides))

    def _open_short(self, **overrides) -> Signal:
        return self._promote_to_open(self._create_short(**overrides))


# ── A. No signal ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNoSignal(RecoveryTestCase):
    def test_empty_database_returns_no_open_signal(self):
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)
        self.assertIsNone(result.signal)
        self.assertEqual(result.errors, ())

    def test_nothing_is_created_or_mutated(self):
        self.recover()
        signals = self.harness.repository.list_signals()
        self.assertEqual(len(signals), 0)
        self.assertIsNone(self.harness.repository.get_open_signal())


# ── B. WAIT does not persist as OPEN ───────────────────────────────────────


@pytest.mark.unit
class TestWaitRecovery(RecoveryTestCase):
    def test_wait_leaves_no_open_signal(self):
        result = self.harness.engine.process(make_analysis("WAIT"))
        self.assertEqual(result.outcome.value, "WAIT")
        recovery_result = self.recover()
        self.assertEqual(recovery_result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)
        self.assertIsNone(recovery_result.signal)


# ── C/D. OPEN LONG / SHORT recovered exactly ───────────────────────────────


@pytest.mark.unit
class TestOpenRecovery(RecoveryTestCase):
    def test_open_long_recovered_exactly(self):
        sig = self._open_long()
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        recovered = result.signal
        self.assertEqual(recovered.id, sig.id)
        self.assertEqual(recovered.symbol, sig.symbol)
        self.assertEqual(recovered.direction, "LONG")
        self.assertEqual(recovered.entry, sig.entry)
        self.assertEqual(recovered.stop_loss, sig.stop_loss)
        self.assertEqual(recovered.take_profit, sig.take_profit)
        self.assertEqual(recovered.opened_at, sig.opened_at)
        self.assertEqual(recovered.status, STATUS_OPEN)

    def test_open_short_recovered_exactly(self):
        sig = self._open_short()
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        recovered = result.signal
        self.assertEqual(recovered.id, sig.id)
        self.assertEqual(recovered.direction, "SHORT")
        self.assertEqual(recovered.entry, sig.entry)
        self.assertEqual(recovered.stop_loss, sig.stop_loss)
        self.assertEqual(recovered.take_profit, sig.take_profit)
        self.assertEqual(recovered.opened_at, sig.opened_at)

    def test_recovered_signal_has_all_immutable_fields(self):
        sig = self._open_long()
        recovered = self.recover().signal
        for name in (
            "id",
            "symbol",
            "direction",
            "entry",
            "stop_loss",
            "take_profit",
            "opened_at",
        ):
            self.assertEqual(getattr(recovered, name), getattr(sig, name))


# ── D/G. Repeated restarts return the same signal ──────────────────────────


@pytest.mark.unit
class TestRepeatedRestart(RecoveryTestCase):
    def test_repeat_recover_same_signal_no_duplicates(self):
        self._open_long()
        seen_ids = set()
        for _ in range(10):
            result = self.recover()
            self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
            seen_ids.add(result.signal.id)
        self.assertEqual(len(seen_ids), 1)
        open_list = self.harness.repository.list_signals(status=STATUS_OPEN)
        self.assertEqual(len(open_list), 1)


# ── C2. PENDING_ENTRY recovery ──────────────────────────────────────────────


@pytest.mark.unit
class TestPendingRecovery(RecoveryTestCase):
    def test_pending_long_recovered_exactly(self):
        sig = self._create_long()
        self.assertEqual(sig.status, STATUS_PENDING_ENTRY)
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        recovered = result.signal
        self.assertEqual(recovered.id, sig.id)
        self.assertEqual(recovered.status, STATUS_PENDING_ENTRY)
        self.assertEqual(recovered.direction, "LONG")
        self.assertEqual(recovered.entry, sig.entry)
        # Recovery never invents an OPEN moment for a pending signal.
        self.assertIsNone(recovered.opened_at)

    def test_pending_exists_no_open_and_no_overwrite(self):
        sig = self._create_long()
        result = self.recover()
        # The single active signal is recovered verbatim; nothing changes.
        self.assertEqual(result.signal.id, sig.id)
        after = self.harness.repository.get_signal(sig.id)
        self.assertEqual(after.status, STATUS_PENDING_ENTRY)
        self.assertEqual(after.entry, sig.entry)
        self.assertIsNone(self.harness.repository.get_open_signal())

    def test_recovered_pending_resumes_entry_monitoring(self):
        sig = self._create_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(61100, 60700))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_recovered_pending_then_entry_then_tp(self):
        sig = self._create_long()
        self.recover()
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(61100, 60700))
        )
        monitor.poll()
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)
        r = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(65000, 60500))
        ).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(r.signal.id, sig.id)

    def test_recovered_pending_ignores_exit_candles_until_entry(self):
        sig = self._create_long()
        self.recover()
        # A candle below the entry that dips into SL territory must NOT close
        # the signal: the exit is only evaluated once the position is OPEN.
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(60800, 59000))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_PENDING_ENTRY)

    def test_pending_survives_reopen(self):
        sig = self._create_long()
        reopened = self.reopen()
        result = RecoveryService(reopened).recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertEqual(result.signal.id, sig.id)
        self.assertEqual(result.signal.status, STATUS_PENDING_ENTRY)

    def test_corrupted_pending_with_opened_at_is_error(self):
        from database.database import iso_utc_now

        sig = self._create_long()
        with self.harness.db.transaction() as conn:
            conn.execute(
                "UPDATE signals SET opened_at = ? WHERE id = ?",
                (iso_utc_now(), sig.id),
            )
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERY_ERROR)
        self.assertIn("opened_at", result.message)
        # The corrupted tracking state is not silently "fixed".
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_PENDING_ENTRY)


# ── E/F. Closed states stay closed ─────────────────────────────────────────


@pytest.mark.unit
class TestClosedRecovery(RecoveryTestCase):
    def _close(self, new_status):
        sig = self._open_long()
        close_price = sig.take_profit if new_status == STATUS_TP_HIT else sig.stop_loss
        return self.harness.repository.transition_signal(
            sig.id, new_status, close_price=close_price
        )

    def test_tp_hit_stays_closed(self):
        closed = self._close(STATUS_TP_HIT)
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(self.harness.repository.get_signal(closed.id).status, STATUS_TP_HIT)

    def test_sl_hit_stays_closed(self):
        closed = self._close(STATUS_SL_HIT)
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(self.harness.repository.get_signal(closed.id).status, STATUS_SL_HIT)

    def test_cancelled_stays_closed(self):
        closed = self._close(STATUS_CANCELLED)
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(self.harness.repository.get_signal(closed.id).status, STATUS_CANCELLED)


# ── F. Immutability: recovery changes nothing ──────────────────────────────


@pytest.mark.unit
class TestImmutability(RecoveryTestCase):
    def test_persisted_row_unchanged_after_recovery(self):
        sig = self._open_long()
        before = self.harness.repository.get_signal(sig.id)
        self.recover()
        after = self.harness.repository.get_signal(sig.id)
        self.assertEqual(before, after)
        self.assertEqual(after.entry, sig.entry)
        self.assertEqual(after.stop_loss, sig.stop_loss)
        self.assertEqual(after.take_profit, sig.take_profit)

    def test_no_columns_touched(self):
        sig = self._open_long()
        row_before = dict(self._raw_row(sig.id))
        self.recover()
        row_after = dict(self._raw_row(sig.id))
        self.assertEqual(row_before, row_after)

    def _raw_row(self, signal_id):
        with self.harness.db.read() as conn:
            return conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()


# ── G. No AI during OPEN recovery ──────────────────────────────────────────


@pytest.mark.unit
class TestNoAI(RecoveryTestCase):
    def test_open_recovery_never_invokes_ai_entries(self):
        self._open_long()
        with (
            patch(
                "signal_engine.analysis.analyze_signal",
                side_effect=AssertionError("analyze_signal must not be called"),
            ) as analyze_signal,
            patch(
                "signal_engine.engine.SignalEngine.analyze_and_create",
                side_effect=AssertionError("analyze_and_create must not be called during recovery"),
            ) as analyze_and_create,
            patch(
                "signal_engine.engine.SignalEngine.process",
                side_effect=AssertionError("process must not be called during recovery"),
            ) as process,
        ):
            result = self.recover()
            self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
            analyze_signal.assert_not_called()
            analyze_and_create.assert_not_called()
            process.assert_not_called()

    def test_no_llm_import_path_in_recovery_module(self):
        import inspect

        import signal_engine.recovery as recovery_module

        source = inspect.getsource(recovery_module)
        for marker in (
            "from .analysis",
            "from .llm",
            "import analysis",
            "import llm",
            "from database.select",
        ):
            self.assertNotIn(marker, source)


# ── H. Monitor resumes on the recovered OPEN signal ────────────────────────


@pytest.mark.unit
class TestMonitorResumption(RecoveryTestCase):
    def test_recovered_long_then_tp_hit(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(65000, 60500))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(result.signal.status, STATUS_TP_HIT)
        self.assertEqual(result.signal.close_price, sig.take_profit)

    def test_recovered_long_then_sl_hit(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(60500, 59000))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.SL_HIT)
        self.assertEqual(result.signal.status, STATUS_SL_HIT)
        self.assertEqual(result.signal.close_price, sig.stop_loss)

    def test_recovered_short_then_tp_hit(self):
        sig = self._open_short()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(59500, 57000))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(result.signal.status, STATUS_TP_HIT)

    def test_ambiguous_stays_open(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(64500, 59000))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.AMBIGUOUS)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_unresolved_candle_stays_open(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(63000, 60500))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)


# ── I/J. Data unavailable / malformed data leaves signal OPEN ──────────────


@pytest.mark.unit
class TestDataFailure(RecoveryTestCase):
    def test_binance_error_leaves_open(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        data = FakeMarketData()
        data.error = BinanceError("connection reset")
        monitor = RecoveryService(self.harness.repository).resume_monitor(data)
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_malformed_ohlc_leaves_open(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        bad = _candle(float("nan"), 59000)
        monitor = RecoveryService(self.harness.repository).resume_monitor(FakeMarketData(bad))
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_only_forming_candle_leaves_open(self):
        sig = self._open_long()
        self.assertEqual(self.recover().signal.id, sig.id)
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(65000, 59000, closed=False))
        )
        result = monitor.poll()
        self.assertEqual(result.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)


# ── J. Persistence across a reopened connection ────────────────────────────


@pytest.mark.unit
class TestPersistenceReopen(RecoveryTestCase):
    def test_recover_after_reopen(self):
        sig = self._open_long()
        reopened = self.reopen()
        result = RecoveryService(reopened).recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertEqual(result.signal.id, sig.id)
        self.assertEqual(result.signal.entry, sig.entry)
        self.assertEqual(result.signal.stop_loss, sig.stop_loss)
        self.assertEqual(result.signal.take_profit, sig.take_profit)

    def test_signal_survives_several_reopens(self):
        sig = self._open_long()
        last = None
        for _ in range(5):
            repo = self.reopen()
            last = RecoveryService(repo).recover()
            self.assertEqual(last.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertEqual(last.signal.id, sig.id)


# ── K. Concurrent recovery ─────────────────────────────────────────────────


@pytest.mark.unit
class TestConcurrency(RecoveryTestCase):
    def test_concurrent_recovery_same_signal_no_duplicates(self):
        sig = self._open_long()
        results = []

        def worker():
            repo = self.reopen()
            return RecoveryService(repo).recover()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: worker(), range(8)))

        for result in results:
            self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
            self.assertEqual(result.signal.id, sig.id)
        open_list = self.harness.repository.list_signals(status=STATUS_OPEN)
        self.assertEqual(len(open_list), 1)

    def test_concurrent_recovery_with_no_signal(self):
        results = []

        def worker():
            repo = self.reopen()
            return RecoveryService(repo).recover()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: worker(), range(8)))

        for result in results:
            self.assertEqual(result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)


# ── L. Errors: failure and inconsistent state ──────────────────────────────


@pytest.mark.unit
class TestRecoveryErrors(RecoveryTestCase):
    def test_database_failure_returns_structured_error(self):
        self._open_long()
        with patch.object(
            self.harness.repository, "list_signals", side_effect=RuntimeError("disk full")
        ):
            result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERY_ERROR)
        self.assertIsNone(result.signal)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("disk full", result.message)

    def test_no_raw_exception_leaked(self):
        self._open_long()
        with patch.object(
            self.harness.repository, "list_signals", side_effect=RuntimeError("boom")
        ):
            result = self.recover()
        self.assertNotIsInstance(result, RuntimeError)
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERY_ERROR)

    def test_database_failure_does_not_close_or_create(self):
        self._open_long()
        with patch.object(
            self.harness.repository, "list_signals", side_effect=RuntimeError("boom")
        ):
            result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERY_ERROR)
        still_open = self.harness.repository.get_open_signal()
        self.assertIsNotNone(still_open)


# ── Binance safety ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBinanceSafety(RecoveryTestCase):
    def test_recovery_never_imports_or_uses_binance(self):
        import inspect

        import signal_engine.recovery as recovery_module

        source = inspect.getsource(recovery_module)
        for marker in ("import binance", "from binance"):
            self.assertNotIn(marker, source)

    def test_recovery_returns_without_touching_market_data(self):
        self._open_long()
        bands = RecoveryService(self.harness.repository)
        monitor = bands.resume_monitor(FakeMarketData(_candle(65000, 60500)))
        result = self.recover()  # recovery itself performs no fetch
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        # The monitor (created but not polled) has not fetched anything yet.
        self.assertEqual(monitor.market_data.candles, [_candle(65000, 60500)])

    def test_monitor_after_recovery_is_public_market_data_only(self):
        self._open_long()
        monitor = RecoveryService(self.harness.repository).resume_monitor(
            FakeMarketData(_candle(65000, 60500))
        )
        poll = monitor.poll()
        self.assertEqual(poll.outcome, MonitorOutcome.TP_HIT)
        self.assertIsNone(monitor.market_data.error)


# ── Recovery API contract used by a later scheduler ────────────────────────


@pytest.mark.unit
class TestStartupContract(RecoveryTestCase):
    def test_symbol_and_timeframe_present_in_result(self):
        self._open_long()
        result = self.recover()
        self.assertEqual(result.signal.symbol, "BTCUSDT")
        self.assertEqual(result.signal.timeframe, "1h")

    def test_no_open_means_caller_may_schedule_analysis(self):
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)
        self.assertTrue(
            self.harness.engine.state.can_open(),
            "no OPEN signal means analysis scheduling may proceed later",
        )

    def test_open_means_ai_must_not_run(self):
        self._open_long()
        result = self.recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertFalse(
            self.harness.engine.state.can_open(),
            "recovered OPEN signal locks the AI",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
