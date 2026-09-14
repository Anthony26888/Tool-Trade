"""Phase 6 metadata persistence and additive schema migration tests.

The ``signals`` table gained three nullable audit-meta columns. Databases
created by earlier phases must be upgraded in place without touching existing
rows (requirement 20/23: never destroy trading history on startup).
"""

from __future__ import annotations

import sqlite3
import unittest
from decimal import Decimal

import pytest

from database.database import Database, SignalRepository, iso_utc_now
from database.models import STATUS_OPEN, Signal, SignalValidationError
from tests.signal_engine_test_helpers import TempSignalDb

_OLD_SIGNALS_DDL = """
    CREATE TABLE signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
        status TEXT NOT NULL CHECK (status IN ('OPEN', 'TP_HIT', 'SL_HIT', 'CANCELLED')),
        entry TEXT NOT NULL,
        stop_loss TEXT NOT NULL,
        take_profit TEXT NOT NULL,
        confidence INTEGER CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 100),
        risk_reward TEXT,
        rationale TEXT,
        provider TEXT,
        model_name TEXT,
        strategy_name TEXT,
        strategy_version TEXT,
        created_at TEXT NOT NULL,
        opened_at TEXT,
        closed_at TEXT,
        close_price TEXT,
        close_reason TEXT,
        result TEXT CHECK (result IS NULL OR result IN ('WIN', 'LOSS'))
    )
"""


def _build_old_schema(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(_OLD_SIGNALS_DDL)
        conn.execute(
            """
            INSERT INTO signals (
                symbol, timeframe, direction, status, entry, stop_loss,
                take_profit, confidence, risk_reward, rationale, provider,
                model_name, strategy_name, strategy_version, created_at,
                opened_at, closed_at, close_price, close_reason, result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "1h",
                "LONG",
                STATUS_OPEN,
                "61000.0",
                "60000.0",
                "64000.0",
                80,
                "2.0",
                "legacy signal",
                "deepseek",
                "deepseek-v4-flash",
                None,
                None,
                "2026-08-01T00:00:00.000Z",
                "2026-08-01T00:00:00.000Z",
                None,
                None,
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.unit
class TestSignalMetadataPersistence(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.repo = self.harness.repository

    def test_all_metadata_persisted_and_round_tripped(self):
        signal = self.repo.create_signal(
            "BTCUSDT",
            "1h",
            "LONG",
            "61000.0",
            "60000.0",
            "64000.0",
            confidence=80,
            rationale="trend",
            provider="deepseek",
            model_name="deepseek-v4-flash",
            analysis_timestamp="2026-09-10T01:00:00.000Z",
            market_timestamp="2026-09-10T00:00:00.000Z",
            candle_close_price="61050.25",
        )
        reloaded = self.repo.get_signal(signal.id)
        self.assertEqual(reloaded.analysis_timestamp, "2026-09-10T01:00:00.000Z")
        self.assertEqual(reloaded.market_timestamp, "2026-09-10T00:00:00.000Z")
        self.assertEqual(reloaded.candle_close_price, Decimal("61050.25"))
        self.assertEqual(reloaded.provider, "deepseek")
        self.assertEqual(reloaded.model_name, "deepseek-v4-flash")

    def test_metadata_fields_nullable(self):
        signal = self.repo.create_signal(
            "BTCUSDT", "1h", "SHORT", "59000.0", "60000.0", "58000.0"
        )
        reloaded = self.repo.get_signal(signal.id)
        self.assertIsNone(reloaded.analysis_timestamp)
        self.assertIsNone(reloaded.market_timestamp)
        self.assertIsNone(reloaded.candle_close_price)
        self.assertIsInstance(reloaded, Signal)

    def test_every_decimal_precision_round_trip_including_candle_price(self):
        candle = Decimal("61050.1234567890123456789")
        signal = self.repo.create_signal(
            "BTCUSDT",
            "1h",
            "LONG",
            "61000.1234567890123456789",
            "60000.0000000000001",
            "64000.987654321",
            candle_close_price=candle,
        )
        reloaded = self.repo.get_signal(signal.id)
        self.assertEqual(reloaded.candle_close_price, candle)

    def test_blank_timestamp_rejected(self):
        for kwargs in ({"analysis_timestamp": "  "}, {"market_timestamp": ""}):
            with self.subTest(**kwargs), self.assertRaises(SignalValidationError):
                self.repo.create_signal(
                    "BTCUSDT",
                    "1h",
                    "LONG",
                    "61000.0",
                    "60000.0",
                    "64000.0",
                    **kwargs,
                )

    def test_invalid_candle_close_price_rejected(self):
        for value in ("abc", 0, -1, float("nan"), float("inf"), Decimal("NaN")):
            with self.subTest(value=value), self.assertRaises(SignalValidationError):
                self.repo.create_signal(
                    "BTCUSDT",
                    "1h",
                    "LONG",
                    "61000.0",
                    "60000.0",
                    "64000.0",
                    candle_close_price=value,
                )

    def test_metadata_survives_status_transition(self):
        signal = self.repo.create_signal(
            "BTCUSDT",
            "1h",
            "LONG",
            "61000.0",
            "60000.0",
            "64000.0",
            analysis_timestamp="2026-09-10T01:00:00.000Z",
            market_timestamp="2026-09-10T00:00:00.000Z",
            candle_close_price="61050.25",
        )
        closed = self.repo.transition_signal(
            signal.id, STATUS_OPEN
        )
        self.assertEqual(closed.status, STATUS_OPEN)
        closed = self.repo.transition_signal(
            closed.id, "TP_HIT", close_price="64000.0"
        )
        self.assertEqual(closed.analysis_timestamp, "2026-09-10T01:00:00.000Z")
        self.assertEqual(closed.market_timestamp, "2026-09-10T00:00:00.000Z")
        self.assertEqual(closed.candle_close_price, Decimal("61050.25"))


@pytest.mark.unit
class TestAdditiveMigration(unittest.TestCase):
    def test_old_schema_upgraded_and_data_preserved(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/legacy.db"
            _build_old_schema(path)
            db = Database(path)
            db.initialize()

            conn = db.connect()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
            conn.close()
            for column in ("analysis_timestamp", "market_timestamp", "candle_close_price"):
                self.assertIn(column, columns)

            repo = SignalRepository(db)
            legacy = repo.get_signal(1)
            self.assertEqual(legacy.entry, Decimal("61000.0"))
            self.assertEqual(legacy.direction, "LONG")
            self.assertEqual(legacy.status, STATUS_OPEN)
            self.assertIsNone(legacy.analysis_timestamp)
            self.assertIsNone(legacy.market_timestamp)
            self.assertIsNone(legacy.candle_close_price)
            self.assertEqual(repo.get_open_signal().id, 1)

    def test_initialize_is_idempotent_and_preserves_history(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/legacy.db"
            _build_old_schema(path)
            db = Database(path)
            db.initialize()
            db.initialize()
            db.initialize()
            repo = SignalRepository(db)
            signals = repo.list_signals(symbol="BTCUSDT", order_by="id", desc=False)
            self.assertEqual([s.entry for s in signals], [Decimal("61000.0")])

    def test_new_phase6_signal_writable_after_migration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/legacy.db"
            _build_old_schema(path)
            db = Database(path)
            db.initialize()
            repo = SignalRepository(db)
            repo.transition_signal(1, "CANCELLED", close_reason="migration test")
            new_signal = repo.create_signal(
                "BTCUSDT",
                "1h",
                "SHORT",
                "59000.0",
                "60000.0",
                "58000.0",
                confidence=75,
                analysis_timestamp=iso_utc_now(),
                market_timestamp=iso_utc_now(),
                candle_close_price="58999.5",
            )
            self.assertEqual(new_signal.candle_close_price, Decimal("58999.5"))
            self.assertEqual(len(repo.list_signals(symbol="BTCUSDT")), 2)

    def test_old_schema_without_signal_table_still_creatable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/fresh.db"
            db = Database(path)
            db.initialize()
            repo = SignalRepository(db)
            signal = repo.create_signal(
                "BTCUSDT", "1h", "LONG", "61000.0", "60000.0", "64000.0"
            )
            self.assertEqual(signal.candle_close_price, None)
