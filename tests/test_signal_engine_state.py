"""Phase 6 unit tests: signal state gate (signal_engine/state.py).

The state gate never caches active (PENDING_ENTRY or OPEN) state in memory; it
reads SQLite through the Phase 3 repository on every call.
"""

from __future__ import annotations

import unittest

import pytest

from database.database import Database, SignalRepository
from database.models import (
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_TP_HIT,
    SignalNotFoundError,
    SignalValidationError,
)
from signal_engine import SignalOutcome, SignalState
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis


@pytest.mark.unit
class TestSignalState(unittest.TestCase):
    def test_idle_state(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        self.assertIsNone(harness.state.open_signal())
        self.assertIsNone(harness.state.active_signal())
        self.assertFalse(harness.state.is_open())
        self.assertFalse(harness.state.is_active())
        self.assertTrue(harness.state.can_open())

    def test_state_reflects_created_signal(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        result = harness.engine.process(make_analysis())
        self.assertEqual(result.outcome, SignalOutcome.CREATED)
        # A persisted signal is immediately ACTIVE but not yet OPEN.
        self.assertTrue(harness.state.is_active())
        self.assertFalse(harness.state.is_open())
        self.assertFalse(harness.state.can_open())
        self.assertEqual(harness.state.active_signal().id, result.signal.id)
        self.assertEqual(harness.state.active_signal().status, STATUS_PENDING_ENTRY)
        self.assertIsNone(harness.state.open_signal())

    def test_state_promotes_to_open_and_tracks_it(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        result = harness.engine.process(make_analysis())
        opened = harness.state.transition(result.signal.id, STATUS_OPEN)
        self.assertEqual(opened.status, STATUS_OPEN)
        self.assertTrue(harness.state.is_open())
        self.assertTrue(harness.state.is_active())
        self.assertEqual(harness.state.open_signal().id, result.signal.id)

    def test_state_reads_through_transition_not_cache(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        result = harness.engine.process(make_analysis())
        cancelled = harness.state.cancel(result.signal.id, close_reason="test")
        self.assertEqual(cancelled.status, "CANCELLED")
        self.assertIsNone(harness.state.open_signal())
        self.assertIsNone(harness.state.active_signal())
        self.assertFalse(harness.state.is_open())
        self.assertFalse(harness.state.is_active())

    def test_state_sees_updates_from_another_instance(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        result = harness.engine.process(make_analysis())
        reopened_db = Database(harness.path)
        reopened_db.initialize()
        reopened_state = SignalState(SignalRepository(reopened_db))
        self.assertFalse(reopened_state.is_open())
        self.assertTrue(reopened_state.is_active())
        self.assertEqual(reopened_state.active_signal().id, result.signal.id)

    def test_transition_delegates_to_phase3_rules(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        result = harness.engine.process(make_analysis())
        # PENDING_ENTRY -> OPEN first.
        opened = harness.state.transition(result.signal.id, STATUS_OPEN)
        self.assertEqual(opened.status, STATUS_OPEN)
        with self.assertRaises(SignalValidationError):
            harness.state.transition(result.signal.id, STATUS_TP_HIT)  # no close_price
        closed = harness.state.transition(
            result.signal.id, STATUS_TP_HIT, close_price="64000.0"
        )
        self.assertEqual(closed.status, STATUS_TP_HIT)
        self.assertFalse(harness.state.is_open())
        self.assertFalse(harness.state.is_active())

    def test_pending_entry_transition_to_open_rejected_when_idle(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        with self.assertRaises(SignalNotFoundError):
            harness.state.transition(9999, STATUS_OPEN)

    def test_active_signal_returns_fresh_read_every_call(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        result = harness.engine.process(make_analysis())
        first_read = harness.state.active_signal()
        harness.state.cancel(result.signal.id, close_reason="test")
        self.assertIsNone(harness.state.active_signal())
        self.assertIsNotNone(first_read)
