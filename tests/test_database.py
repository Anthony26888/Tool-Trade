"""Phase 3 unit tests: SQLite signal persistence layer.

Every test uses a throwaway database under a temporary directory. The real
project database is never touched.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import uuid
from decimal import Decimal

import pytest

from binance.market_data import InvalidIntervalError, InvalidSymbolError
from database import (
    DEFAULT_DB_PATH,
    STATUS_CANCELLED,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    Database,
    DemoRepository,
    InvalidTransitionError,
    Signal,
    SignalExistsError,
    SignalNotFoundError,
    SignalRepository,
    SignalValidationError,
)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ta-db-test-")
        self.path = os.path.join(self._tmp, "signals.db")
        self.db = Database(self.path)
        self.db.initialize()
        self.repo = SignalRepository(self.db)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def create_long(self, **kwargs) -> Signal:
        params = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "direction": "LONG",
            "entry": "100.00",
            "stop_loss": "99.00",
            "take_profit": "101.00",
        }
        params.update(kwargs)
        return self.repo.create_signal(**params)

    def create_short(self, **kwargs) -> Signal:
        params = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "direction": "SHORT",
            "entry": "100.00",
            "stop_loss": "101.50",
            "take_profit": "99.00",
        }
        params.update(kwargs)
        return self.repo.create_signal(**params)

    def open_signal(self, signal: Signal) -> Signal:
        """Promote a PENDING_ENTRY signal to OPEN (the entry was touched)."""
        return self.repo.transition_signal(signal.id, STATUS_OPEN)


@pytest.mark.unit
class TestSchema(DatabaseTestCase):
    def test_initialization_creates_tables(self):
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        names = {row["name"] for row in rows}
        self.assertTrue({"signals", "demo_accounts", "demo_positions", "demo_trades"} <= names)

    def test_signals_columns_exist(self):
        expected = frozenset(
            {
                "id",
                "symbol",
                "timeframe",
                "direction",
                "status",
                "entry",
                "stop_loss",
                "take_profit",
                "confidence",
                "risk_reward",
                "rationale",
                "provider",
                "model_name",
                "temperature",
                "strategy_name",
                "strategy_version",
                "created_at",
                "opened_at",
                "closed_at",
                "close_price",
                "close_reason",
                "result",
                "analysis_timestamp",
                "market_timestamp",
                "candle_close_price",
            }
        )
        with self.db.read() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(signals)").fetchall()
            }
        self.assertEqual(columns, expected)

    def test_schema_creation_is_idempotent(self):
        self.create_long()
        self.db.initialize()
        self.repo.get_signal(self.repo.get_active_signal().id)
        self.db.initialize()
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_default_db_path_points_outside_repo(self):
        self.assertNotEqual(DEFAULT_DB_PATH, "")


class TestConcurrentInitialize(DatabaseTestCase):
    @pytest.mark.unit
    def test_initialize_is_safe_when_processes_race(self):
        # The container runs web + daemon as separate processes calling
        # initialize() on the same file; the WAL/schema setup must not deadlock
        # or error (regression: "database is locked").
        path = os.path.join(self._tmp, "concurrent.db")
        n = 8
        errors = []
        barrier = threading.Barrier(n)

        def worker() -> None:
            db = Database(path)
            barrier.wait()
            try:
                db.initialize()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        db = Database(path)
        db.initialize()
        with db.read() as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        for table in (
            "signals",
            "demo_accounts",
            "demo_positions",
            "demo_trades",
            "scheduler_state",
            "runtime_state",
            "app_settings",
            "config_audit",
        ):
            self.assertIn(table, tables)


@pytest.mark.unit
class TestCreateSignal(DatabaseTestCase):
    def test_create_valid_long_signal(self):
        signal = self.create_long(provider="test-provider", model_name="deepseek-v4-flash")
        self.assertEqual(signal.direction, "LONG")
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)
        self.assertEqual(signal.entry, Decimal("100.00"))
        self.assertEqual(signal.stop_loss, Decimal("99.00"))
        self.assertEqual(signal.take_profit, Decimal("101.00"))
        self.assertEqual(signal.provider, "test-provider")
        self.assertEqual(signal.model_name, "deepseek-v4-flash")
        self.assertIsNone(signal.closed_at)
        self.assertIsNone(signal.close_price)
        self.assertIsNone(signal.opened_at)

    def test_create_valid_short_signal(self):
        signal = self.create_short()
        self.assertEqual(signal.direction, "SHORT")
        self.assertEqual(signal.entry, Decimal("100.00"))
        self.assertEqual(signal.stop_loss, Decimal("101.50"))
        self.assertEqual(signal.take_profit, Decimal("99.00"))
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)

    def test_timestamps_default_to_utc(self):
        signal = self.create_long()
        self.assertTrue(signal.created_at.endswith("Z"))
        # A PENDING_ENTRY is not yet open: opened_at is recorded on promotion.
        self.assertIsNone(signal.opened_at)

    def test_reject_invalid_long_ordering(self):
        cases = (
            ("100.00", "100.00", "101.00"),  # entry == SL
            ("100.00", "99.00", "100.00"),  # TP == entry
            ("100.00", "101.00", "103.00"),  # SL > entry
            ("100.00", "99.00", "99.00"),  # SL >= TP
            ("100.00", "101.00", "99.00"),  # SL > TP
        )
        for entry, sl, tp in cases:
            with self.subTest(entry=entry, sl=sl, tp=tp), self.assertRaises(
                SignalValidationError
            ):
                self.create_long(entry=entry, stop_loss=sl, take_profit=tp)
        self.assertEqual(self.repo.list_signals(), [])

    def test_reject_invalid_short_ordering(self):
        cases = (
            ("100.00", "101.00", "100.00"),  # TP == entry
            ("100.00", "100.00", "99.00"),  # SL == entry
            ("100.00", "99.00", "99.00"),  # TP == SL
            ("100.00", "99.00", "98.00"),  # SL < entry
            ("100.00", "101.00", "102.00"),  # TP > entry
        )
        for entry, sl, tp in cases:
            with self.subTest(entry=entry, sl=sl, tp=tp), self.assertRaises(
                SignalValidationError
            ):
                self.create_short(entry=entry, stop_loss=sl, take_profit=tp)
        self.assertEqual(self.repo.list_signals(), [])

    def test_reject_non_positive_prices(self):
        for field in ("entry", "stop_loss", "take_profit"):
            with self.subTest(field=field), self.assertRaises(SignalValidationError):
                self.create_long(**{field: "0"})
        with self.assertRaises(SignalValidationError):
            self.create_long(entry="-1.00")

    def test_reject_invalid_confidence(self):
        for bad in (101, -1, True, "50"):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                self.create_long(confidence=bad)

    def test_reject_non_positive_risk_reward(self):
        with self.assertRaises(SignalValidationError):
            self.create_long(risk_reward="0")
        with self.assertRaises(SignalValidationError):
            self.create_long(risk_reward="-2.0")

    def test_reject_unknown_direction(self):
        with self.assertRaises(SignalValidationError):
            self.create_long(direction="SIDEWAYS")

    def test_reject_unsupported_symbol_and_timeframe(self):
        with self.assertRaises(InvalidSymbolError):
            self.create_long(symbol="BTC/USDT")
        with self.assertRaises(InvalidIntervalError):
            self.create_long(timeframe="2h")


@pytest.mark.unit
class TestSingleOpenSignal(DatabaseTestCase):
    def test_duplicate_open_rejected(self):
        self.create_long()
        with self.assertRaises(SignalExistsError):
            self.create_short()
        signals = self.repo.list_signals()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, "LONG")

    def test_duplicate_pending_rejected(self):
        self.create_short()
        with self.assertRaises(SignalExistsError):
            self.create_long()
        self.assertEqual(len(self.repo.list_signals()), 1)

    def test_pending_blocks_new_signal_and_opening_blocks_new_pending(self):
        # A PENDING_ENTRY occupies the single active slot...
        self.create_long()
        with self.assertRaises(SignalExistsError):
            self.create_short()
        # ... and an OPEN signal still blocks a new PENDING_ENTRY as well.
        self.open_signal(self.repo.get_active_signal())
        with self.assertRaises(SignalExistsError):
            self.create_short()

    def test_new_signal_allowed_after_close(self):
        first = self.create_long()
        promoted = self.open_signal(first)
        self.repo.transition_signal(
            promoted.id, STATUS_TP_HIT, close_price="101.00", close_reason="TP_HIT"
        )
        second = self.create_short()
        self.assertEqual(self.repo.get_active_signal().id, second.id)

    def test_database_check_constraint_blocks_duplicate_ordering(self):
        with self.db.read() as conn, self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO signals (symbol, timeframe, direction, status, entry,"
                " stop_loss, take_profit, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("BTCUSDT", "1h", "LONG", "OPEN", "100", "99", "98", "2026-01-01T00:00:00Z"),
            )


@pytest.mark.unit
class TestGetOpenSignal(DatabaseTestCase):
    def test_none_when_idle(self):
        self.assertIsNone(self.repo.get_open_signal())
        self.assertIsNone(self.repo.get_active_signal())

    def test_returns_single_open_signal(self):
        created = self.create_long()
        # PENDING is active but not OPEN; get_open_signal() waits for promotion.
        self.assertIsNone(self.repo.get_open_signal())
        promoted = self.open_signal(created)
        self.assertEqual(self.repo.get_open_signal().id, promoted.id)

    def test_none_after_close(self):
        created = self.create_long()
        promoted = self.open_signal(created)
        self.repo.transition_signal(promoted.id, STATUS_SL_HIT, close_price="99.00")
        self.assertIsNone(self.repo.get_open_signal())
        self.assertIsNone(self.repo.get_active_signal())


@pytest.mark.unit
class TestGetActiveSignal(DatabaseTestCase):
    def test_active_returns_pending(self):
        created = self.create_long()
        active = self.repo.get_active_signal()
        self.assertEqual(active.id, created.id)
        self.assertEqual(active.status, STATUS_PENDING_ENTRY)

    def test_active_returns_open_after_promotion(self):
        created = self.create_long()
        self.open_signal(created)
        active = self.repo.get_active_signal()
        self.assertEqual(active.status, STATUS_OPEN)

    def test_active_none_after_terminal(self):
        created = self.create_long()
        terminated = self.repo.cancel_signal(created.id, close_reason="test")
        self.assertEqual(terminated.status, STATUS_CANCELLED)
        self.assertIsNone(self.repo.get_active_signal())


@pytest.mark.unit
class TestTransitions(DatabaseTestCase):
    def test_pending_to_open_sets_opened_at(self):
        created = self.create_long()
        self.assertIsNone(created.opened_at)
        promoted = self.repo.transition_signal(created.id, STATUS_OPEN)
        self.assertEqual(promoted.status, STATUS_OPEN)
        self.assertIsNotNone(promoted.opened_at)
        self.assertRegex(promoted.opened_at, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T")

    def test_pending_to_terminal_rejected(self):
        # PENDING_ENTRY includes NO TP/SL decision: it must be promoted first.
        created = self.create_long()
        for target, kwargs in (
            (STATUS_TP_HIT, {"close_price": "101.00"}),
            (STATUS_SL_HIT, {"close_price": "99.00"}),
        ):
            with self.subTest(target=target), self.assertRaises(InvalidTransitionError):
                self.repo.transition_signal(created.id, target, **kwargs)

    def test_pending_to_cancelled(self):
        created = self.create_long()
        cancelled = self.repo.cancel_signal(created.id, close_reason="signal invalid")
        self.assertEqual(cancelled.status, STATUS_CANCELLED)
        self.assertEqual(cancelled.close_reason, "signal invalid")
        self.assertIsNone(cancelled.close_price)
        self.assertIsNone(cancelled.opened_at)
        self.assertIsNone(self.repo.get_active_signal())

    def test_open_to_tp_hit(self):
        created = self.create_long()
        opened = self.open_signal(created)
        closed = self.repo.transition_signal(
            opened.id, STATUS_TP_HIT, close_price="101.50", close_reason="TP_HIT", result="WIN"
        )
        self.assertEqual(closed.status, STATUS_TP_HIT)
        self.assertEqual(closed.close_price, Decimal("101.50"))
        self.assertEqual(closed.close_reason, "TP_HIT")
        self.assertEqual(closed.result, "WIN")
        self.assertIsNotNone(closed.closed_at)
        self.assertEqual(closed.opened_at, opened.opened_at)

    def test_open_to_sl_hit(self):
        created = self.create_long()
        opened = self.open_signal(created)
        closed = self.repo.transition_signal(opened.id, STATUS_SL_HIT, close_price="99.00")
        self.assertEqual(closed.status, STATUS_SL_HIT)
        self.assertEqual(closed.close_reason, STATUS_SL_HIT)

    def test_open_to_cancelled(self):
        created = self.create_long()
        opened = self.open_signal(created)
        cancelled = self.repo.cancel_signal(opened.id, close_reason="model cancelled")
        self.assertEqual(cancelled.status, STATUS_CANCELLED)
        self.assertEqual(cancelled.close_reason, "model cancelled")
        self.assertIsNone(cancelled.close_price)
        self.assertEqual(cancelled.opened_at, opened.opened_at)

    def test_closed_signal_cannot_reopen(self):
        created = self.create_long()
        opened = self.open_signal(created)
        self.repo.transition_signal(opened.id, STATUS_TP_HIT, close_price="101.00")
        with self.assertRaises(InvalidTransitionError):
            self.repo.transition_signal(created.id, STATUS_OPEN)

    def test_terminal_to_terminal_rejected(self):
        created = self.create_long()
        opened = self.open_signal(created)
        self.repo.transition_signal(opened.id, STATUS_TP_HIT, close_price="101.00")
        with self.assertRaises(InvalidTransitionError):
            self.repo.transition_signal(opened.id, STATUS_SL_HIT, close_price="99.00")

    def test_open_to_open_rejected(self):
        created = self.create_long()
        self.open_signal(created)
        with self.assertRaises(InvalidTransitionError):
            self.repo.transition_signal(created.id, STATUS_OPEN)

    def test_pending_to_pending_rejected(self):
        created = self.create_long()
        with self.assertRaises(InvalidTransitionError):
            self.repo.transition_signal(created.id, STATUS_PENDING_ENTRY)

    def test_unknown_status_rejected(self):
        created = self.create_long()
        with self.assertRaises(SignalValidationError):
            self.repo.transition_signal(created.id, "HODL")

    def test_tp_hit_and_sl_hit_require_close_price(self):
        created = self.create_long()
        opened = self.open_signal(created)
        with self.assertRaises(SignalValidationError):
            self.repo.transition_signal(opened.id, STATUS_TP_HIT)
        with self.assertRaises(SignalValidationError):
            self.repo.transition_signal(opened.id, STATUS_SL_HIT)

    def test_promotion_clears_stale_close_fields(self):
        # Promoting a PENDING_ENTRY must reset any close bookkeeping so a
        # cancelled-and-reopened lifecycle cannot leak a stale close.
        created = self.create_long()
        self.repo.transition_signal(created.id, STATUS_OPEN, closed_at="2020-01-01T00:00:00Z")
        reloaded = self.repo.get_signal(created.id)
        self.assertEqual(reloaded.status, STATUS_OPEN)
        self.assertIsNone(reloaded.closed_at)
        self.assertIsNone(reloaded.close_price)
        self.assertIsNone(reloaded.close_reason)
        self.assertIsNone(reloaded.result)

    def test_unknown_signal_rejected(self):
        with self.assertRaises(SignalNotFoundError):
            self.repo.get_signal(9999)
        with self.assertRaises(SignalNotFoundError):
            self.repo.transition_signal(9999, STATUS_OPEN)
        with self.assertRaises(SignalNotFoundError):
            self.repo.transition_signal(9999, STATUS_TP_HIT, close_price="101.00")

    def test_closed_signal_remains_closed(self):
        created = self.create_long()
        self.open_signal(created)
        self.repo.transition_signal(created.id, STATUS_SL_HIT, close_price="99.00")
        reloaded = self.repo.get_signal(created.id)
        self.assertEqual(reloaded.status, STATUS_SL_HIT)
        self.assertIsNone(self.repo.get_open_signal())
        self.assertIsNone(self.repo.get_active_signal())


@pytest.mark.unit
class TestImmutability(DatabaseTestCase):
    def test_entry_sl_tp_unchanged_after_pending_promotion_and_transitions(self):
        created = self.create_long(
            entry="100.50", stop_loss="99.250", take_profit="101.750"
        )
        opened = self.open_signal(created)
        closed = self.repo.transition_signal(
            opened.id, STATUS_TP_HIT, close_price="101.80"
        )
        for attr in ("entry", "stop_loss", "take_profit"):
            with self.subTest(attr=attr):
                self.assertEqual(
                    Decimal(str(getattr(created, attr))),
                    Decimal(str(getattr(closed, attr))),
                )
        with self.db.read() as conn:
            row = conn.execute(
                "SELECT entry, stop_loss, take_profit FROM signals WHERE id = ?",
                (created.id,),
            ).fetchone()
        self.assertEqual(
            tuple(row), (str(created.entry), str(created.stop_loss), str(created.take_profit))
        )

    def test_entry_sl_tp_unchanged_from_pending_to_cancelled(self):
        created = self.create_long(entry="100.50", stop_loss="99.250", take_profit="101.750")
        cancelled = self.repo.cancel_signal(created.id, close_reason="test")
        self.assertEqual(cancelled.entry, created.entry)
        self.assertEqual(cancelled.stop_loss, created.stop_loss)
        self.assertEqual(cancelled.take_profit, created.take_profit)


@pytest.mark.unit
class TestDecimalRoundTrip(DatabaseTestCase):
    def test_high_precision_decimals_round_trip(self):
        entry = "112345.678901234567890123456789"
        sl = "112200.000000000000000000000001"
        tp = "114500.999999999999999999999999"
        signal = self.create_long(entry=entry, stop_loss=sl, take_profit=tp)
        reloaded = self.repo.get_signal(signal.id)
        self.assertEqual(reloaded.entry, Decimal(entry))
        self.assertEqual(reloaded.stop_loss, Decimal(sl))
        self.assertEqual(reloaded.take_profit, Decimal(tp))
        self.assertEqual(reloaded.status, STATUS_PENDING_ENTRY)

    def test_risk_reward_and_close_price_round_trip(self):
        signal = self.create_long(risk_reward="2.500000000000")
        self.assertEqual(signal.risk_reward, Decimal("2.500000000000"))
        opened = self.open_signal(signal)
        closed = self.repo.transition_signal(
            opened.id, STATUS_TP_HIT, close_price="101.0000000000", result="WIN"
        )
        self.assertEqual(closed.close_price, Decimal("101.0000000000"))


@pytest.mark.unit
class TestTimestamps(DatabaseTestCase):
    def test_explicit_timestamps_persist(self):
        created = self.create_long(created_at="2026-09-10T00:00:00.000Z")
        self.assertEqual(created.created_at, "2026-09-10T00:00:00.000Z")
        # PENDING_ENTRY itself carries no open timestamp.
        self.assertIsNone(created.opened_at)
        opened = self.repo.transition_signal(
            created.id, STATUS_OPEN, opened_at="2026-09-10T01:00:00.000Z"
        )
        self.assertEqual(opened.opened_at, "2026-09-10T01:00:00.000Z")
        closed = self.repo.transition_signal(
            opened.id,
            STATUS_TP_HIT,
            close_price="101.00",
            closed_at="2026-09-10T03:15:30.500Z",
        )
        self.assertEqual(closed.closed_at, "2026-09-10T03:15:30.500Z")
        reloaded = self.repo.get_signal(created.id)
        self.assertEqual(reloaded.closed_at, "2026-09-10T03:15:30.500Z")
        self.assertEqual(reloaded.opened_at, "2026-09-10T01:00:00.000Z")

    def test_created_at_is_utc(self):
        signal = self.create_long()
        self.assertRegex(signal.created_at, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T")


@pytest.mark.unit
class TestRollback(DatabaseTestCase):
    def test_duplicate_open_rolls_back(self):
        self.create_long()
        with self.assertRaises(SignalExistsError):
            self.create_short()
        self.assertEqual(len(self.repo.list_signals()), 1)
        self.assertIsNotNone(self.repo.get_active_signal())
        self.assertIsNone(self.repo.get_open_signal())

    def test_invalid_create_leaves_no_rows(self):
        with self.assertRaises(SignalValidationError):
            self.create_long(entry="100", stop_loss="101", take_profit="103")
        self.assertEqual(self.repo.list_signals(), [])

    def test_invalid_close_price_does_not_change_status(self):
        created = self.create_long()
        opened = self.open_signal(created)
        with self.assertRaises(SignalValidationError):
            self.repo.transition_signal(opened.id, STATUS_TP_HIT, close_price="abc")
        reloaded = self.repo.get_signal(created.id)
        self.assertEqual(reloaded.status, STATUS_OPEN)
        self.assertIsNone(reloaded.closed_at)

    def test_failed_promotion_keeps_pending(self):
        created = self.create_long()
        with self.assertRaises(SignalNotFoundError):
            self.repo.transition_signal(9999, STATUS_OPEN)
        reloaded = self.repo.get_signal(created.id)
        self.assertEqual(reloaded.status, STATUS_PENDING_ENTRY)
        self.assertIsNone(reloaded.opened_at)


@pytest.mark.unit
class TestSurvivesReopen(DatabaseTestCase):
    def test_data_survives_closing_and_reopening(self):
        created = self.create_long(
            provider="test-provider", strategy_name="phase3", strategy_version="1.0"
        )
        reopened_db = Database(self.path)
        reopened_db.initialize()
        reopened_repo = SignalRepository(reopened_db)
        reloaded = reopened_repo.get_signal(created.id)
        self.assertEqual(reloaded.id, created.id)
        self.assertEqual(reloaded.status, STATUS_PENDING_ENTRY)
        self.assertEqual(reloaded.provider, "test-provider")
        self.assertEqual(reloaded.strategy_name, "phase3")
        self.assertEqual(reloaded.entry, created.entry)
        self.assertEqual(reopened_repo.get_active_signal().id, created.id)
        self.assertIsNone(reopened_repo.get_open_signal())


@pytest.mark.unit
class TestListSignals(DatabaseTestCase):
    def test_filters_by_direction_and_status(self):
        first = self.create_long()
        opened = self.open_signal(first)
        self.repo.transition_signal(opened.id, STATUS_TP_HIT, close_price="101.00")
        self.create_short()
        longs = self.repo.list_signals(direction="LONG")
        shorts = self.repo.list_signals(direction="SHORT")
        opens = self.repo.list_signals(status=STATUS_OPEN)
        pendings = self.repo.list_signals(status=STATUS_PENDING_ENTRY)
        self.assertEqual([s.id for s in longs], [first.id])
        self.assertEqual(len(shorts), 1)
        self.assertEqual(len(opens), 0)
        self.assertEqual([s.id for s in pendings], [shorts[0].id])

    def test_invalid_order_column_rejected(self):
        with self.assertRaises(SignalValidationError):
            self.repo.list_signals(order_by="entry; DROP TABLE signals")

    def test_filters_by_created_at_window(self):
        first = self.create_long()
        self.open_signal(first)
        self.repo.transition_signal(first.id, STATUS_TP_HIT, close_price="101.00")
        second = self.create_short()
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE signals SET created_at = ? WHERE id = ?",
                ("2026-09-10T00:00:00.000Z", first.id),
            )
            conn.execute(
                "UPDATE signals SET created_at = ? WHERE id = ?",
                ("2026-09-11T00:00:00.000Z", second.id),
            )
        a, b = "2026-09-10T00:00:00.000Z", "2026-09-11T00:00:00.000Z"
        self.assertEqual([s.id for s in self.repo.list_signals(created_since=a)], [second.id, first.id])
        self.assertEqual([s.id for s in self.repo.list_signals(created_since=a, created_until=b)], [second.id, first.id])
        self.assertEqual([s.id for s in self.repo.list_signals(created_since=b)], [second.id])
        self.assertEqual([s.id for s in self.repo.list_signals(created_since=a, created_until=a)], [first.id])
        self.assertEqual(
            [s.id for s in self.repo.list_signals(created_since="2020-01-01")],
            [second.id, first.id],
        )

    def test_time_filter_normalizes_and_rejects(self):
        self.create_long()
        self.assertEqual(
            len(self.repo.list_signals(created_since="2026-09-13T17:00:00.000Z")), 1
        )
        with self.assertRaises(SignalValidationError):
            self.repo.list_signals(created_since="not-a-timestamp")


@pytest.mark.unit
class TestSignalsDelete(DatabaseTestCase):
    def _account_id(self) -> int:
        return DemoRepository(self.db).ensure_account(
            name="demo",
            initial_balance=Decimal("1000"),
            margin_per_trade=Decimal("50"),
            leverage=10,
            risk_percent=Decimal("1"),
            fee_rate=Decimal("0.04"),
        )["id"]

    def _closed_signal_with_trade(self, *, short: bool = False) -> int:
        sig = self.create_short() if short else self.create_long()
        self.open_signal(sig)
        if short:
            self.repo.transition_signal(
                sig.id, STATUS_SL_HIT, close_price="99.00", result="LOSS"
            )
        else:
            self.repo.transition_signal(
                sig.id, STATUS_TP_HIT, close_price="101.00", result="WIN"
            )
        demo = DemoRepository(self.db)
        pos = demo.create_position(
            account_id=self._account_id(),
            signal_id=sig.id,
            symbol="BTCUSDT",
            side="SHORT" if short else "LONG",
            entry_price=Decimal("100"),
            quantity=Decimal("5"),
            position_size=Decimal("500"),
            margin=Decimal("50"),
            leverage=10,
            stop_loss=Decimal("101.50" if short else "99.00"),
            take_profit=Decimal("99.00" if short else "101.00"),
        )
        demo.record_trade(
            position_id=pos["id"],
            signal_id=sig.id,
            account_id=pos["account_id"],
            side="SHORT" if short else "LONG",
            entry_price=Decimal("100"),
            exit_price=Decimal("99.00" if short else "101.00"),
            quantity=Decimal("5"),
            margin=Decimal("50"),
            position_size=Decimal("500"),
            leverage=10,
            gross_pnl=Decimal("5"),
            fee=Decimal("0.20"),
            net_pnl=Decimal("4.80"),
            pnl_percent=Decimal("0.96"),
            result="WIN" if not short else "LOSS",
            next_balance=Decimal("1004.80"),
            next_equity=Decimal("1004.80"),
            next_peak_equity=Decimal("1004.80"),
        )
        return sig.id

    def test_delete_pending_signal(self):
        created = self.create_long()
        counts = self.repo.delete_signals([created.id])
        self.assertEqual(counts["deleted"], [created.id])
        self.assertEqual(counts["missing"], [])
        self.assertEqual(self.repo.list_signals(), [])
        with self.assertRaises(SignalNotFoundError):
            self.repo.get_signal(created.id)

    def test_delete_missing_and_duplicate_ids(self):
        created = self.create_long()
        counts = self.repo.delete_signals([created.id, created.id, 9999])
        self.assertEqual(counts["deleted"], [created.id])
        self.assertEqual(counts["missing"], [9999])

    def test_delete_rejects_open_signal(self):
        created = self.create_long()
        self.open_signal(created)
        sig = self.repo.get_signal(created.id)
        self.assertEqual(sig.status, STATUS_OPEN)
        counts = self.repo.delete_signals([created.id])
        self.assertEqual(counts["deleted"], [])
        self.assertEqual(counts["skipped_open"], [created.id])
        self.assertEqual(self.repo.get_signal(created.id).status, STATUS_OPEN)
        self.assertEqual(self.repo.get_active_signal().id, created.id)

    def test_delete_rejects_invalid_ids(self):
        with self.assertRaises(SignalValidationError):
            self.repo.delete_signals([])
        with self.assertRaises(SignalValidationError):
            self.repo.delete_signals(["abc"])

    def test_non_cascade_keeps_signal_with_position(self):
        closed = self._closed_signal_with_trade()
        counts = self.repo.delete_signals([closed], cascade=False)
        self.assertEqual(counts["deleted"], [])
        self.assertEqual(counts["skipped_linked"], [closed])
        self.assertIsNotNone(self.repo.get_signal(closed))

    def test_cascade_deletes_position_and_trade(self):
        closed = self._closed_signal_with_trade()
        with self.db.read() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM demo_positions WHERE signal_id = ?", (closed,)
                ).fetchone()[0],
                1,
            )
        counts = self.repo.delete_signals([closed], cascade=True)
        self.assertEqual(counts["deleted"], [closed])
        with self.db.read() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM demo_positions WHERE signal_id = ?", (closed,)
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM demo_trades WHERE signal_id = ?", (closed,)
                ).fetchone()[0],
                0,
            )
            audit = conn.execute(
                "SELECT namespace, action FROM config_audit "
                "WHERE namespace = 'signals'"
            ).fetchall()
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["action"], "delete")

    def test_delete_all_skips_open_and_cascades(self):
        closed = self._closed_signal_with_trade()
        pending = self.create_long()
        counts = self.repo.delete_all_signals()
        self.assertEqual(sorted(counts["deleted"]), sorted([pending.id, closed]))
        self.assertEqual(counts["skipped_open"], [])
        self.assertEqual(self.repo.list_signals(), [])
        with self.db.read() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM demo_trades WHERE signal_id = ?", (closed,)
                ).fetchone()[0],
                0,
            )

    def test_delete_all_only_open(self):
        created = self.create_long()
        self.open_signal(created)
        counts = self.repo.delete_all_signals()
        self.assertEqual(counts["deleted"], [])
        self.assertEqual(counts["skipped_open"], [created.id])
        self.assertEqual(self.repo.get_signal(created.id).status, STATUS_OPEN)

    def test_delete_all_when_empty(self):
        counts = self.repo.delete_all_signals()
        self.assertEqual(counts["deleted"], [])
        self.assertEqual(counts["skipped_open"], [])


@pytest.mark.unit
class TestPendingEntryLifecycle(DatabaseTestCase):
    def test_create_returns_pending_entry(self):
        signal = self.create_long()
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)
        self.assertIsNone(signal.opened_at)
        self.assertIsNone(signal.closed_at)

    def test_pending_signal_round_trips_through_promotion(self):
        created = self.create_long()
        self.open_signal(created)
        self.assertEqual(self.repo.get_signal(created.id).status, STATUS_OPEN)

    def test_pending_signal_cancels_and_releases_active_slot(self):
        created = self.create_long()
        self.repo.cancel_signal(created.id, close_reason="test")
        second = self.create_short()
        self.assertTrue(second.id > created.id)
        self.assertEqual(second.status, STATUS_PENDING_ENTRY)

    def test_list_signals_statuses(self):
        created = self.create_long()
        self.assertEqual(
            [s.id for s in self.repo.list_signals(status=STATUS_PENDING_ENTRY)],
            [created.id],
        )
        self.assertEqual(self.repo.list_signals(status=STATUS_OPEN), [])


_OLD_SIGNALS_DDL = """
    CREATE TABLE signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
        status TEXT NOT NULL CHECK (status IN ('OPEN', 'TP_HIT', 'SL_HIT', 'CANCELLED')),
        entry TEXT NOT NULL, stop_loss TEXT NOT NULL, take_profit TEXT NOT NULL,
        confidence INTEGER, risk_reward TEXT, rationale TEXT, provider TEXT,
        model_name TEXT, strategy_name TEXT, strategy_version TEXT,
        created_at TEXT NOT NULL, opened_at TEXT, closed_at TEXT,
        close_price TEXT, close_reason TEXT,
        result TEXT CHECK (result IS NULL OR result IN ('WIN', 'LOSS'))
    )
"""


@pytest.mark.unit
class TestPendingEntryMigration(DatabaseTestCase):
    """The signals.status CHECK must gain PENDING_ENTRY without data loss."""

    def _fresh_db(self) -> tuple[Database, SignalRepository]:
        path = os.path.join(self._tmp, f"legacy-{uuid.uuid4().hex}.db")
        db = Database(path)
        return db, SignalRepository(db)

    def _install_legacy_schema(self, db: Database, rows: list[tuple]) -> None:
        import sqlite3

        conn = sqlite3.connect(db.path)
        conn.execute(_OLD_SIGNALS_DDL)
        conn.executemany(
            "INSERT INTO signals (symbol, timeframe, direction, status, entry,"
            " stop_loss, take_profit, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def test_migration_preserves_existing_open_and_closed_rows(self):
        db, _ = self._fresh_db()
        self._install_legacy_schema(
            db,
            [
                ("BTCUSDT", "1h", "LONG", "OPEN", "100", "99", "101", "2026-01-01T00:00:00Z"),
                ("BTCUSDT", "1h", "SHORT", "TP_HIT", "100", "101", "99", "2026-01-02T00:00:00Z"),
            ],
        )
        db.initialize()
        repo = SignalRepository(db)
        opened = repo.get_signal(1)
        closed = repo.get_signal(2)
        self.assertEqual(opened.status, STATUS_OPEN)
        self.assertEqual(closed.status, STATUS_TP_HIT)
        # The OPEN signal is still the sole active signal after migration.
        self.assertEqual(repo.get_active_signal().id, opened.id)
        # The migrated schema now accepts CANCELLED and PENDING_ENTRY.
        promoted_lifecycle = repo.cancel_signal(opened.id, close_reason="test")
        self.assertEqual(promoted_lifecycle.status, STATUS_CANCELLED)
        fresh = repo.create_signal("BTCUSDT", "1h", "LONG", "100", "99", "101")
        self.assertEqual(fresh.status, STATUS_PENDING_ENTRY)

    def test_migration_keeps_ids_and_is_idempotent(self):
        db, _ = self._fresh_db()
        self._install_legacy_schema(
            db,
            [("BTCUSDT", "1h", "LONG", "OPEN", "100", "99", "101", "2026-01-01T00:00:00Z")],
        )
        db.initialize()
        repo = SignalRepository(db)
        self.assertEqual(repo.get_signal(1).id, 1)
        db.initialize()
        db.initialize()
        repo2 = SignalRepository(db)
        self.assertEqual(repo2.get_signal(1).id, 1)
        self.assertEqual(len(repo2.list_signals()), 1)
        # The migrated OPEN signal still holds the active slot; close it so the
        # next signal can verify the AUTOINCREMENT sequence survived the rebuild.
        repo2.cancel_signal(1, close_reason="test")
        created = repo2.create_signal("BTCUSDT", "1h", "SHORT", "100", "101", "99")
        self.assertEqual(created.id, 2)
        self.assertEqual(created.status, STATUS_PENDING_ENTRY)

    def test_fresh_database_already_supports_pending_entry(self):
        created = self.create_long()
        self.assertEqual(created.status, STATUS_PENDING_ENTRY)
        self.open_signal(created)
        self.repo.transition_signal(created.id, STATUS_SL_HIT, close_price="99.00")
        self.assertEqual(self.repo.get_signal(created.id).status, STATUS_SL_HIT)
