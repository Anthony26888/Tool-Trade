"""Plan B' Phase B1 tests: per-symbol one-active invariant (plan B-prime).

The global single-active rule becomes one slot PER SYMBOL: a BTCUSDT
position never blocks ETHUSDT analysis, while two actives on the SAME
symbol stay impossible (including under concurrent creation). Covers the
repository invariant, state/scheduler/engine gates, and scoped recovery.
Single-symbol behavior is pinned by the pre-existing suite (must stay green
untouched); these tests only add the multi-symbol dimension.
"""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from database.models import (
    STATUS_PENDING_ENTRY,
    SignalExistsError,
)
from signal_engine import (
    OneHourScheduler,
    SchedulerOutcome,
    SignalEngine,
    SignalState,
)
from signal_engine.recovery import RecoveryOutcome, RecoveryService
from tests.signal_engine_test_helpers import (
    TempSignalDb,
    make_analysis,
    make_candles,
)

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def make_pending(harness: TempSignalDb, symbol: str, direction: str = "LONG"):
    analysis = make_analysis(
        direction,
        symbol=symbol,
        entry_price=61000.0,
        stop_loss=60000.0,
        take_profit=64000.0,
    )
    result = harness.engine.process(analysis)
    assert result.outcome.value == "CREATED", result
    return result.signal


@pytest.mark.unit
class TestPerSymbolInvariant(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.repo = self.harness.repository

    def test_two_symbols_hold_one_slot_each(self):
        btc = make_pending(self.harness, "BTCUSDT")
        eth = make_pending(self.harness, "ETHUSDT")
        self.assertEqual(btc.status, STATUS_PENDING_ENTRY)
        self.assertEqual(eth.status, STATUS_PENDING_ENTRY)
        actives = self.repo.list_active_signals()
        self.assertEqual(
            sorted(s.symbol for s in actives), ["BTCUSDT", "ETHUSDT"]
        )

    def test_second_active_same_symbol_rejected(self):
        from decimal import Decimal

        make_pending(self.harness, "BTCUSDT")
        # Engine gate blocks first (no LLM burned)...
        again = self.harness.engine.process(make_analysis("LONG", symbol="BTCUSDT"))
        self.assertEqual(again.outcome.value, "BLOCKED_OPEN_SIGNAL")
        # ...and the repository transaction is the atomic backstop.
        with self.assertRaises(SignalExistsError):
            self.repo.create_signal(
                "BTCUSDT", "1h", "LONG", Decimal("61000"), Decimal("60000"), Decimal("64000")
            )
        # ...while another symbol is still free.
        make_pending(self.harness, "ETHUSDT")

    def test_per_symbol_reads(self):
        make_pending(self.harness, "BTCUSDT")
        state = SignalState(self.repo)
        self.assertIsNotNone(state.active_signal_for("BTCUSDT"))
        self.assertIsNone(state.active_signal_for("ETHUSDT"))
        self.assertIsNotNone(state.active_signal())  # global legacy intact
        self.assertIsNone(state.open_signal_for("BTCUSDT"))
        self.assertIsNone(state.open_signal_for("ETHUSDT"))

    def test_malformed_symbol_fails_safe_to_blocked(self):
        make_pending(self.harness, "BTCUSDT")
        state = SignalState(self.repo)
        # Never raises: falls back to the global lookup (blocks, never creates).
        self.assertIsNotNone(state.active_signal_for("!!!not-a-symbol!!!"))
        empty = TempSignalDb()
        self.addCleanup(empty.close)
        self.assertIsNone(SignalState(empty.repository).active_signal_for("!!!"))

    def test_concurrent_same_symbol_creates_exactly_one(self):
        from decimal import Decimal

        barrier = threading.Barrier(2, timeout=10)
        outcomes: list = []

        def create(_i):
            barrier.wait(timeout=10)
            try:
                self.repo.create_signal(
                    "BTCUSDT", "1h", "LONG", Decimal("61000"), Decimal("60000"),
                    Decimal("64000"),
                )
                return "CREATED"
            except SignalExistsError:
                return "REJECTED"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = sorted(pool.map(create, range(2)))
        self.assertEqual(outcomes, ["CREATED", "REJECTED"])
        actives = [
            s
            for s in self.repo.list_active_signals()
            if s.symbol == "BTCUSDT"
        ]
        self.assertEqual(len(actives), 1)

    def test_concurrent_different_symbols_both_create(self):
        barrier = threading.Barrier(2, timeout=10)

        def create(symbol):
            barrier.wait(timeout=10)
            make_pending(self.harness, symbol)
            return symbol

        with ThreadPoolExecutor(max_workers=2) as pool:
            done = sorted(pool.map(create, ["BTCUSDT", "ETHUSDT"]))
        self.assertEqual(done, ["BTCUSDT", "ETHUSDT"])


@pytest.mark.unit
class TestPerSymbolGates(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)

    def _scheduler(self, symbol, analyzer):
        from tests.test_signal_engine_scheduler import FakeSchedulerData

        # Candles ending just before NOW: newer than the cold-start marker
        # seeded from the pending signal, so the gate (not detection) decides.
        start_ms = int(NOW.timestamp() * 1000) - 219 * 3_600_000
        candles = make_candles(220, start_ms=start_ms)
        return OneHourScheduler(
            self.harness.repository,
            market_data=FakeSchedulerData(*candles),
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=analyzer,
            symbol=symbol,
        )

    def test_engine_blocks_same_symbol_only(self):
        make_pending(self.harness, "BTCUSDT")
        engine = SignalEngine(self.harness.repository, self.harness.state)
        blocked = engine.process(make_analysis("LONG", symbol="BTCUSDT"))
        self.assertEqual(blocked.outcome.value, "BLOCKED_OPEN_SIGNAL")
        created = engine.process(
            make_analysis(
                "SHORT",
                symbol="ETHUSDT",
                entry_price=59000.0,
                stop_loss=59500.0,
                take_profit=58000.0,
            )
        )
        self.assertEqual(created.outcome.value, "CREATED")

    def test_scheduler_lock_is_per_symbol(self):
        from tests.test_signal_engine_scheduler import make_analyzer

        make_pending(self.harness, "BTCUSDT")
        btc_calls: list = []
        btc = self._scheduler("BTCUSDT", make_analyzer("WAIT", btc_calls))
        self.assertEqual(
            btc.tick(now=NOW).outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL
        )
        self.assertEqual(btc_calls, [])
        eth_calls: list = []
        eth = self._scheduler("ETHUSDT", make_analyzer("WAIT", eth_calls))
        self.assertEqual(eth.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(eth_calls), 1)


@pytest.mark.unit
class TestScopedRecovery(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)

    def test_global_recover_still_strict(self):
        make_pending(self.harness, "BTCUSDT")
        make_pending(self.harness, "ETHUSDT")
        result = RecoveryService(self.harness.repository).recover()
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERY_ERROR)

    def test_scoped_recover_picks_own_symbol(self):
        make_pending(self.harness, "BTCUSDT")
        make_pending(self.harness, "ETHUSDT")
        service = RecoveryService(self.harness.repository)
        btc = service.recover(symbol="BTCUSDT")
        self.assertEqual(btc.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertEqual(btc.signal.symbol, "BTCUSDT")
        eth = service.recover(symbol="ETHUSDT")
        self.assertEqual(eth.outcome, RecoveryOutcome.RECOVERED_OPEN)
        self.assertEqual(eth.signal.symbol, "ETHUSDT")
        idle = service.recover(symbol="XAUUSDT")
        self.assertEqual(idle.outcome, RecoveryOutcome.NO_OPEN_SIGNAL)

    def test_scoped_recover_still_rejects_double_slot(self):
        make_pending(self.harness, "BTCUSDT")
        repo = self.harness.repository
        # Simulate a corrupted double slot (bypasses the guarded create path).
        with repo.database.transaction() as conn:
            conn.execute(
                "INSERT INTO signals (symbol, timeframe, direction, entry, "
                "stop_loss, take_profit, status, created_at) VALUES "
                "('BTCUSDT', '1h', 'SHORT', '60000', '61000', '59000', "
                "'PENDING_ENTRY', '2026-09-10T00:00:00.000Z')"
            )
        result = RecoveryService(repo).recover(symbol="BTCUSDT")
        self.assertEqual(result.outcome, RecoveryOutcome.RECOVERY_ERROR)


@pytest.mark.unit
class TestMonitorScoping(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.repo = self.harness.repository

    def _monitor(self, symbol=None, candles=()):
        from signal_engine import SignalMonitor

        class PerSymbolFeed:
            def __init__(self, rows):
                self.rows = list(rows)
                self.seen: list = []

            def fetch_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
                self.seen.append(symbol)
                return list(self.rows)

        return SignalMonitor(self.repo, market_data=PerSymbolFeed(candles), symbol=symbol)

    def test_scoped_monitor_ignores_sibling_symbol(self):
        from binance.market_data import Candle
        from signal_engine import MonitorOutcome

        btc = make_pending(self.harness, "BTCUSDT")
        self.repo.transition_signal(btc.id, "OPEN")
        eth = make_pending(self.harness, "ETHUSDT")
        # ETH monitor with entry-touching candles promotes ETH only.
        entry_touch = Candle(
            timestamp=2_000_000, open=61000.0, high=61100.0, low=60900.0,
            close=61050.0, volume=1.0, close_time=2_059_999, is_closed=True,
        )
        eth_mon = self._monitor("ETHUSDT", [entry_touch])
        result = eth_mon.poll()
        self.assertEqual(result.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(result.signal.id, eth.id)
        self.assertEqual(self.repo.get_signal(btc.id).status, "OPEN")
        # BTC monitor with TP-touching candles closes BTC only.
        tp_touch = Candle(
            timestamp=3_000_000, open=63900.0, high=64100.0, low=63800.0,
            close=64000.0, volume=1.0, close_time=3_059_999, is_closed=True,
        )
        btc_mon = self._monitor("BTCUSDT", [tp_touch])
        closed = btc_mon.poll()
        self.assertEqual(closed.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(self.repo.get_signal(btc.id).status, "TP_HIT")
        self.assertEqual(self.repo.get_signal(eth.id).status, "OPEN")

    def test_unscoped_monitor_keeps_legacy_global(self):
        from signal_engine import MonitorOutcome

        make_pending(self.harness, "BTCUSDT")
        mon = self._monitor(None, [])
        # No market data needed: idle check happens before any fetch... but
        # an active signal exists, so it fetches; empty feed -> DATA_UNAVAILABLE.
        result = mon.poll()
        self.assertEqual(result.outcome, MonitorOutcome.DATA_UNAVAILABLE)

    def test_invalid_symbol_rejected_at_construction(self):
        from signal_engine import SignalMonitor

        with self.assertRaises(ValueError):
            SignalMonitor(self.repo, symbol="!!!")

    def test_unresolvable_callable_goes_idle(self):
        from signal_engine import MonitorOutcome, SignalMonitor

        make_pending(self.harness, "BTCUSDT")

        def boom():
            raise RuntimeError("config broken")

        mon = SignalMonitor(self.repo, symbol=boom)
        result = mon.poll()
        self.assertEqual(result.outcome, MonitorOutcome.NO_ACTIVE_SIGNAL)


@pytest.mark.unit
class TestSharedMoney(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.repo = self.harness.repository
        from demo.executor import DemoExecutor

        self.executor = DemoExecutor(self.harness.db)

    def _open_position(self, symbol, direction="LONG"):
        sig = make_pending(self.harness, symbol)
        self.repo.transition_signal(sig.id, "OPEN")
        position = self.executor.open_position(self.repo.get_signal(sig.id))
        assert position is not None
        return self.repo.get_signal(sig.id)

    def test_concurrent_closes_keep_every_pnl(self):
        eth = self._open_position("ETHUSDT")
        btc = self._open_position("BTCUSDT")
        # Both TP: BTC +1000 gross-ish, ETH +1000 gross-ish on default sizing.
        self.repo.transition_signal(
            btc.id, "TP_HIT", close_price="64000", result="WIN"
        )
        self.repo.transition_signal(
            eth.id, "TP_HIT", close_price="64000", result="WIN"
        )
        barrier = threading.Barrier(2, timeout=10)

        def close(signal):
            barrier.wait(timeout=10)
            return self.executor.close_position(self.repo.get_signal(signal.id))

        with ThreadPoolExecutor(max_workers=2) as pool:
            trades = list(
                pool.map(close, [self.repo.get_signal(btc.id), self.repo.get_signal(eth.id)])
            )
        self.assertTrue(all(t is not None for t in trades))
        account = self.executor.account()
        expected = round(1000.0 + float(trades[0].net_pnl) + float(trades[1].net_pnl), 2)
        self.assertAlmostEqual(float(account.balance), expected, places=2)
        self.assertTrue(all(t is not None for t in trades))

    def test_open_cap_refuses_beyond_cap(self):
        import unittest.mock

        import demo.executor as executor_mod

        self._open_position("BTCUSDT")
        self._open_position("ETHUSDT")
        xau = make_pending(self.harness, "XAUUSDT")
        self.repo.transition_signal(xau.id, "OPEN")
        # Two OPEN positions already; with the cap lowered to 2 the third
        # open is refused even though its own slot is valid.
        with unittest.mock.patch.object(executor_mod, "MAX_OPEN_POSITIONS", 2):
            refused = self.executor.open_position(self.repo.get_signal(xau.id))
        self.assertIsNone(refused)


if __name__ == "__main__":
    unittest.main()
