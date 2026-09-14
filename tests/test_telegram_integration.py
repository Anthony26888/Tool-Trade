"""Phase 11 integration: Telegram hooks fire exactly on successful transitions.

Verifies the wiring end-to-end without the network: the scheduler announces a
newly persisted PENDING_ENTRY once; the monitor announces the OPEN, TP, SL, and
ambiguous events only when the corresponding repository transition succeeds;
failures never roll back; concurrent polls emit exactly one notification; and
recovery never announces a position again after a restart.
"""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import pytest

from database.models import (
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
)
from signal_engine import (
    DEFAULT_WINDOW_CANDLES,
    MonitorOutcome,
    OneHourScheduler,
    RecoveryOutcome,
    RecoveryService,
    SchedulerOutcome,
    SignalMonitor,
    TelegramConfig,
    TelegramNotifier,
    TelegramSendResult,
)
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis, make_candles
from tests.test_signal_engine_scheduler import (
    NOW,
    FakeMonitorData,
    FakeSchedulerData,
    _monitor_candle,
    latest_ms,
    make_analyzer,
)


class RecordingNotifier:
    """Neutral fake notifier that records every dispatch (thread-safe)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _record(self, entry):
        self.calls.append(entry)

    def notify_signal_created(self, signal, *, candle_ts=None):
        self._record(("created", signal.id, candle_ts))

    def notify_signal_opened(self, signal):
        self._record(("opened", signal.id))

    def notify_signal_tp(self, signal):
        self._record(("tp", signal.id))

    def notify_signal_sl(self, signal):
        self._record(("sl", signal.id))

    def notify_ambiguous(self, signal, reason, *, candle_ts=None):
        self._record(("ambiguous", signal.id, reason, candle_ts))


class RaisingNotifier:
    """Simulates a hard Telegram outage; must never break the transition."""

    def notify_signal_created(self, signal, *, candle_ts=None):
        raise RuntimeError("telegram outage (created)")

    def notify_signal_opened(self, signal):
        raise RuntimeError("telegram outage (opened)")

    def notify_signal_tp(self, signal):
        raise RuntimeError("telegram outage (tp)")

    def notify_signal_sl(self, signal):
        raise RuntimeError("telegram outage (sl)")

    def notify_ambiguous(self, signal, reason, *, candle_ts=None):
        raise RuntimeError("telegram outage (ambiguous)")


class _CountingTelegramClient:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_message(self, text: str) -> TelegramSendResult:
        self.sent.append(text)
        return TelegramSendResult(ok=True, message_id=1)


class TelegramIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)

    def _create_long(self, **overrides):
        result = self.harness.engine.process(make_analysis("LONG", **overrides))
        self.assertEqual(result.outcome.value, "CREATED")
        return result.signal

    def _open_long(self, **overrides):
        sig = self._create_long(**overrides)
        return self.harness.state.transition(sig.id, STATUS_OPEN)


# ── A. Scheduler: PENDING_ENTRY created ──────────────────────────────────────


@pytest.mark.unit
class TestSchedulerNotifications(TelegramIntegrationTestCase):
    def test_scheduler_created_notifies_exactly_once(self):
        notifier = RecordingNotifier()
        md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=make_analyzer("LONG", []),
            notifier=notifier,
        )
        first = sched.tick(now=NOW)
        self.assertEqual(first.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(
            notifier.calls,
            [("created", first.signal.id, latest_ms(md.candles))],
        )
        # The same candle is never re-analyzed, so never re-announced.
        second = sched.tick(now=NOW)
        self.assertEqual(second.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(
            notifier.calls,
            [("created", first.signal.id, latest_ms(md.candles))],
        )

    def test_scheduler_wait_never_notifies(self):
        notifier = RecordingNotifier()
        md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=make_analyzer("WAIT", []),
            notifier=notifier,
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(notifier.calls, [])

    def test_scheduler_blocked_active_signal_never_notifies(self):
        # market_timestamp BEFORE the fake 2024 candle window: the seeded marker
        # stays older than the detected candle, so the gate — not detection —
        # decides this tick.
        self._create_long(market_timestamp="2024-06-01T00:00:00.000Z")
        notifier = RecordingNotifier()
        md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=make_analyzer("LONG", []),
            notifier=notifier,
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(notifier.calls, [])

    def test_notifier_param_is_optional(self):
        md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=make_analyzer("WAIT", []),
        )
        self.assertIsNone(sched.notifier)
        self.assertEqual(sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)

    def test_scheduler_notification_failure_does_not_fail_creation(self):
        md = FakeSchedulerData(*make_candles(DEFAULT_WINDOW_CANDLES))
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=make_analyzer("LONG", []),
            notifier=RaisingNotifier(),
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        self.assertEqual(result.signal.status, STATUS_PENDING_ENTRY)


# ── B. Monitor: OPEN / TP / SL / AMBIGUOUS ───────────────────────────────────


@pytest.mark.unit
class TestMonitorNotifications(TelegramIntegrationTestCase):
    def test_entry_hit_opens_and_notifies_once(self):
        sig = self._create_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(61100, 60700))
        result = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(
            self.harness.repository.get_signal(sig.id).status, STATUS_OPEN
        )
        self.assertEqual(notifier.calls, [("opened", sig.id)])

    def test_tp_hit_closes_and_notifies_once(self):
        sig = self._open_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(65000, 60500))
        result = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(
            self.harness.repository.get_signal(sig.id).status, STATUS_TP_HIT
        )
        self.assertEqual(notifier.calls, [("tp", sig.id)])

    def test_sl_hit_closes_and_notifies_once(self):
        sig = self._open_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(60500, 59000))
        result = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.SL_HIT)
        self.assertEqual(
            self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT
        )
        self.assertEqual(notifier.calls, [("sl", sig.id)])

    def test_pending_entry_ambiguous_notifies_and_stays_pending(self):
        sig = self._create_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(64000, 59900))
        result = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.AMBIGUOUS)
        self.assertEqual(
            self.harness.repository.get_signal(sig.id).status, STATUS_PENDING_ENTRY
        )
        self.assertEqual(len(notifier.calls), 1)
        kind, sid, reason, candle_ts = notifier.calls[0]
        self.assertEqual((kind, sid, candle_ts), ("ambiguous", sig.id, 1_000_000))
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)

    def test_open_ambiguous_notifies_and_stays_open(self):
        sig = self._open_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(65000, 59000))
        result = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.AMBIGUOUS)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)
        self.assertEqual(notifier.calls[0][0], "ambiguous")

    def test_monitoring_candle_never_notifies(self):
        self._open_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(60500, 60700))
        result = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(notifier.calls, [])

    def test_repeated_ambiguous_polls_send_once(self):
        # Real notifier (counting client): dedup on (signal id, candle ts).
        client = _CountingTelegramClient()
        notifier = TelegramNotifier(
            TelegramConfig(
                bot_token="123:TEST",
                chat_id="-1001",
                enabled=True,
                timeout=1.0,
            ),
            client=client,
        )
        self._open_long()
        md = FakeMonitorData(_monitor_candle(65000, 59000))
        monitor = SignalMonitor(
            self.harness.repository, md, notifier=notifier
        )
        monitor.poll()
        monitor.poll()
        self.assertEqual(len(client.sent), 1)

    def test_telegram_outage_never_rolls_back_open(self):
        sig = self._create_long()
        md = FakeMonitorData(_monitor_candle(61100, 60700))
        result = SignalMonitor(
            self.harness.repository, md, notifier=RaisingNotifier()
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_telegram_outage_never_rolls_back_tp(self):
        sig = self._open_long()
        md = FakeMonitorData(_monitor_candle(65000, 60500))
        result = SignalMonitor(
            self.harness.repository, md, notifier=RaisingNotifier()
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_TP_HIT)

    def test_telegram_outage_never_rolls_back_sl(self):
        sig = self._open_long()
        md = FakeMonitorData(_monitor_candle(60500, 59000))
        result = SignalMonitor(
            self.harness.repository, md, notifier=RaisingNotifier()
        ).poll()
        self.assertEqual(result.outcome, MonitorOutcome.SL_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT)


# ── C. Concurrency: exactly one notification per transition ──────────────────


@pytest.mark.unit
class TestConcurrentNotifications(TelegramIntegrationTestCase):
    def test_concurrent_entry_promotion_single_notification(self):
        sig = self._create_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(61100, 60700))
        barrier = threading.Barrier(2, timeout=10)

        def poll(_i):
            barrier.wait(timeout=10)
            return SignalMonitor(
                self.harness.repository, md, notifier=notifier
            ).poll()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(poll, range(2)))
        outcomes = {r.outcome for r in results}
        self.assertIn(MonitorOutcome.ENTRY_HIT, outcomes)
        self.assertTrue(outcomes <= {MonitorOutcome.ENTRY_HIT, MonitorOutcome.MONITORING})
        self.assertEqual(self.harness.repository.get_active_signal().status, STATUS_OPEN)
        self.assertEqual(len(self.harness.repository.list_signals()), 1)
        # Only the winning transition announces the position.
        self.assertEqual(notifier.calls, [("opened", sig.id)])

    def test_concurrent_tp_polls_single_notification(self):
        sig = self._open_long()
        notifier = RecordingNotifier()
        md = FakeMonitorData(_monitor_candle(65000, 60500))
        barrier = threading.Barrier(2, timeout=10)

        def poll(_i):
            barrier.wait(timeout=10)
            return SignalMonitor(
                self.harness.repository, md, notifier=notifier
            ).poll()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(poll, range(2)))
        outcomes = sorted(r.outcome for r in results)
        self.assertEqual(
            outcomes, [MonitorOutcome.NO_OPEN_SIGNAL, MonitorOutcome.TP_HIT]
        )
        self.assertEqual(notifier.calls, [("tp", sig.id)])


# ── D. Recovery: never re-announces an existing position ─────────────────────


@pytest.mark.unit
class TestRecoveryNoNotifications(TelegramIntegrationTestCase):
    def test_recovery_does_not_notify(self):
        sig = self._open_long()
        recovered = RecoveryService(self.harness.repository).recover()
        self.assertEqual(recovered.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertEqual(recovered.signal.id, sig.id)

    def test_resumed_monitor_has_no_notifier(self):
        self._create_long()
        service = RecoveryService(self.harness.repository)
        recovered = service.recover()
        self.assertEqual(recovered.outcome, RecoveryOutcome.RECOVERED_OPEN)
        resume = service.resume_monitor(market_data=None)
        self.assertIsNone(resume.notifier)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
