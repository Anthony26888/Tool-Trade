"""Per-candle daemon activity log (Phase 16) — database repository tests.

The candle_log table stores a single diagnostic row per (symbol, timeframe,
candle open time) explaining why the daemon produced LONG/SHORT/WAIT or skipped
a candle (locked, data unavailable, AI error). It is never read by the trading
logic; these tests pin the persisted format and the retention prune.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from database.database import (
    CandleLogRepository,
    Database,
    SignalRepository,
    SignalValidationError,
)

SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
CANDLE_MS = 1_720_000_000_000
NEXT_CANDLE_MS = CANDLE_MS + 3_600_000


@pytest.fixture()
def repo():
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "candle_log.db")
    db = Database(path)
    db.initialize()
    yield CandleLogRepository(db)
    tmp.cleanup()


class TestCandleLogRepository:
    def test_upsert_creates_one_row_per_candle(self, repo):
        sig = SignalRepository(repo.database).create_signal(
            SYMBOL, TIMEFRAME, "LONG", "61000", "60000", "64000",
            confidence=82, provider="deepseek", model_name="deepseek-v4-flash",
            temperature=0.2, rationale="uptrend continuation",
        )
        repo.upsert(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            candle_timestamp_ms=CANDLE_MS,
            outcome="CREATED",
            decision="LONG",
            confidence=82,
            entry="61000",
            stop_loss="60000",
            take_profit="64000",
            close_price="61050.5",
            signal_id=sig.id,
            provider="deepseek",
            model="deepseek-v4-flash",
            temperature=0.2,
            reasoning="uptrend continuation",
            indicators_json='{"rsi14": 55.5}',
        )
        rows = repo.list()
        assert len(rows) == 1
        entry = rows[0]
        assert entry.decision == "LONG"
        assert entry.outcome == "CREATED"
        assert entry.confidence == 82
        assert entry.entry == "61000"
        assert entry.stop_loss == "60000"
        assert entry.take_profit == "64000"
        assert entry.signal_id == sig.id
        assert entry.provider == "deepseek"
        assert entry.model == "deepseek-v4-flash"
        assert entry.reasoning == "uptrend continuation"
        assert entry.indicators_json == '{"rsi14": 55.5}'
        assert entry.id == 1

    def test_quota_columns_round_trip(self, repo):
        repo.upsert(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            candle_timestamp_ms=CANDLE_MS,
            outcome="CREATED",
            decision="LONG",
            llm_calls=5,
            prompt_tokens=10500,
            completion_tokens=1800,
            total_tokens=12300,
        )
        entry = repo.list()[0]
        assert entry.llm_calls == 5
        assert entry.prompt_tokens == 10500
        assert entry.completion_tokens == 1800
        assert entry.total_tokens == 12300

    def test_quota_columns_default_null(self, repo):
        repo.upsert(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            candle_timestamp_ms=CANDLE_MS,
            outcome="WAIT",
            decision="WAIT",
        )
        entry = repo.list()[0]
        assert entry.llm_calls is None
        assert entry.prompt_tokens is None
        assert entry.completion_tokens is None
        assert entry.total_tokens is None

    def test_quota_columns_reject_bad_values(self, repo):
        for bad in (-1, True, 1.5, "5"):
            with pytest.raises(SignalValidationError):
                repo.upsert(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    candle_timestamp_ms=CANDLE_MS,
                    outcome="ERROR",
                    decision="NONE",
                    llm_calls=bad,
                )

    def test_upsert_same_candle_updates_not_duplicates(self, repo):
        repo.upsert(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            candle_timestamp_ms=CANDLE_MS,
            outcome="ERROR",
            decision="NONE",
            error_notes="first attempt failed",
            recorded_at="2026-09-10T00:01:00.000Z",
        )
        repo.upsert(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            candle_timestamp_ms=CANDLE_MS,
            outcome="WAIT",
            decision="WAIT",
            reasoning="retry succeeded",
            recorded_at="2026-09-10T00:05:00.000Z",
        )
        rows = repo.list()
        assert len(rows) == 1
        entry = rows[0]
        assert entry.outcome == "WAIT"
        assert entry.decision == "WAIT"
        assert entry.reasoning == "retry succeeded"
        assert entry.recorded_at == "2026-09-10T00:05:00.000Z"

    def test_list_newest_first_and_limit_offset(self, repo):
        for i, ts in enumerate((CANDLE_MS, NEXT_CANDLE_MS, NEXT_CANDLE_MS + 3_600_000)):
            repo.upsert(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                candle_timestamp_ms=ts,
                outcome="WAIT",
                decision="WAIT",
                confidence=50,
                recorded_at=f"2026-09-10T00:0{i}:00.000Z",
            )
        assert [e.candle_timestamp_ms for e in repo.list(limit=2)] == [
            NEXT_CANDLE_MS + 3_600_000,
            NEXT_CANDLE_MS,
        ]
        assert [e.candle_timestamp_ms for e in repo.list(offset=1, limit=1)] == [
            NEXT_CANDLE_MS
        ]

    def test_list_filters_by_decision(self, repo):
        repo.upsert(
            symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=CANDLE_MS,
            outcome="WAIT", decision="WAIT",
        )
        repo.upsert(
            symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=NEXT_CANDLE_MS,
            outcome="WAIT", decision="LONG", confidence=80,
        )
        assert [e.decision for e in repo.list(decision="LONG")] == ["LONG"]
        assert [e.decision for e in repo.list(decision="NONE")] == []

    def test_list_rejects_unknown_decision(self, repo):
        with pytest.raises(SignalValidationError):
            repo.list(decision="ROCKET")

    def test_upsert_rejects_bad_confidence(self, repo):
        with pytest.raises(SignalValidationError):
            repo.upsert(
                symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=CANDLE_MS,
                outcome="CREATED", decision="LONG", confidence=101,
            )

    def test_upsert_rejects_non_positive_candle_ts(self, repo):
        with pytest.raises(SignalValidationError):
            repo.upsert(
                symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=0,
                outcome="ERROR",
            )

    def test_indicators_round_trip_as_json_string(self, repo):
        repo.upsert(
            symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=CANDLE_MS,
            outcome="CREATED", decision="SHORT", confidence=70,
            indicators_json='{"ema20": 60000, "rsi14": 44.2}',
        )
        raw = repo.list()[0].indicators_json
        assert '"ema20": 60000' in raw
        assert '"rsi14": 44.2' in raw

    def test_optional_prices_kept_as_text(self, repo):
        repo.upsert(
            symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=CANDLE_MS,
            outcome="BLOCKED_ACTIVE_SIGNAL", decision="NONE",
        )
        entry = repo.list()[0]
        assert entry.entry is None
        assert entry.stop_loss is None
        assert entry.take_profit is None
        assert entry.signal_id is None
        assert entry.error_notes is None

    def test_old_schema_migrated_and_rows_preserved(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy_candle.db")
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    """
                    CREATE TABLE candle_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        candle_timestamp_ms INTEGER NOT NULL,
                        closed_at TEXT,
                        recorded_at TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        decision TEXT,
                        confidence INTEGER,
                        entry TEXT,
                        stop_loss TEXT,
                        take_profit TEXT,
                        close_price TEXT,
                        signal_id INTEGER,
                        provider TEXT,
                        model TEXT,
                        temperature TEXT,
                        reasoning TEXT,
                        error_notes TEXT,
                        indicators_json TEXT,
                        UNIQUE (symbol, timeframe, candle_timestamp_ms)
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO candle_log (symbol, timeframe, candle_timestamp_ms,"
                    " recorded_at, outcome, decision) VALUES (?, ?, ?, ?, ?, ?)",
                    (SYMBOL, TIMEFRAME, CANDLE_MS, "2026-09-10T00:00:00.000Z", "WAIT", "WAIT"),
                )
                conn.commit()
            finally:
                conn.close()
            db = Database(path)
            db.initialize()
            migrated = CandleLogRepository(db)
            rows = migrated.list()
            assert len(rows) == 1
            assert rows[0].outcome == "WAIT"
            assert rows[0].llm_calls is None
            migrated.upsert(
                symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=NEXT_CANDLE_MS,
                outcome="CREATED", decision="LONG", llm_calls=1, total_tokens=1500,
            )
            assert migrated.list(limit=2)[0].total_tokens == 1500

    def test_clear_removes_all_rows(self, repo):
        repo.upsert(
            symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=CANDLE_MS,
            outcome="WAIT", decision="WAIT",
        )
        assert repo.clear() == 1
        assert repo.list() == []

    def test_retention_prunes_oldest_rows(self, monkeypatch):
        import database.database as db_module
        monkeypatch.setattr(db_module, "CANDLE_LOG_RETENTION", 3)
        tmp = tempfile.TemporaryDirectory()
        db = Database(os.path.join(tmp.name, "retention.db"))
        db.initialize()
        repo = CandleLogRepository(db)
        try:
            for i in range(5):
                repo.upsert(
                    symbol=SYMBOL, timeframe=TIMEFRAME,
                    candle_timestamp_ms=CANDLE_MS + i * 3_600_000,
                    outcome="WAIT", decision="WAIT",
                )
            rows = repo.list(limit=10)
            assert len(rows) == 3
            assert [e.candle_timestamp_ms for e in rows] == [
                CANDLE_MS + 4 * 3_600_000,
                CANDLE_MS + 3 * 3_600_000,
                CANDLE_MS + 2 * 3_600_000,
            ]
        finally:
            tmp.cleanup()

    def test_cross_instance_read(self, repo):
        sig = SignalRepository(repo.database).create_signal(
            SYMBOL, TIMEFRAME, "LONG", "61000", "60000", "64000", confidence=90,
        )
        repo.upsert(
            symbol=SYMBOL, timeframe=TIMEFRAME, candle_timestamp_ms=CANDLE_MS,
            outcome="CREATED", decision="LONG", confidence=90, signal_id=sig.id,
        )
        second = CandleLogRepository(repo.database)
        rows = second.list()
        assert len(rows) == 1
        assert rows[0].signal_id == sig.id
