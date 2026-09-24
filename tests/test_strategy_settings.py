"""Settings Strategy Tuning tests (stored overlay > env > defaults, no restart).

Covers the validator overlay precedence, DB readers, engine use of stored
guardrails, the per-tick warn-hours override, the Settings service
(get/update/validate/reset), the web round-trip, and the UI markers.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.request
from decimal import Decimal

import pytest

from database.database import ConfigRepository, Database
from signal_engine.config import ConfigService, SettingsValidationError
from signal_engine.event_calendar import EventCalendar
from signal_engine.validator import (
    GuardrailConfig,
    SignalValidationError,
    default_guardrails,
    read_stored_strategy,
    read_stored_warn_hours,
    validate_analysis,
)
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis
from web.server import WebApplication, WebServer


def write_strategy(db, doc: dict) -> None:
    ConfigRepository(db).set("strategy", json.dumps(doc))


# ── overlay precedence ───────────────────────────────────────────────────


@pytest.mark.unit
class TestStoredOverlay(unittest.TestCase):
    def test_stored_beats_env_beats_default(self):
        env = {"BTCUSDT_MIN_CONFIDENCE": "70", "BTCUSDT_HTF_BIAS": "0"}
        self.assertEqual(GuardrailConfig.from_stored({}, env).min_confidence, 70)
        self.assertEqual(GuardrailConfig.from_stored(None, env).htf_bias, 0)
        guards = GuardrailConfig.from_stored(
            {"min_confidence": 55, "htf_bias": 1}, env
        )
        self.assertEqual(guards.min_confidence, 55)
        self.assertEqual(guards.htf_bias, 1)
        # Untouched keys keep env/default values.
        self.assertEqual(guards.min_risk_reward, Decimal("1.2"))

    def test_invalid_stored_values_fall_back(self):
        guards = GuardrailConfig.from_stored(
            {"min_confidence": "abc", "max_funding_rate": "zzz"}, {}
        )
        self.assertEqual(guards.min_confidence, 60)
        self.assertEqual(guards.max_funding_rate, Decimal("0.0005"))

    def test_unknown_stored_keys_ignored(self):
        guards = GuardrailConfig.from_stored({"nope": 1}, {})
        self.assertEqual(guards.min_confidence, 60)

    def test_default_guardrails_unchanged(self):
        self.assertEqual(default_guardrails({}).min_confidence, 60)


# ── DB readers ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStoredReaders(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "t.db"))
        self.db.initialize()

    def test_missing_or_broken_is_empty(self):
        self.assertEqual(read_stored_strategy(self.db), {})
        self.assertIsNone(read_stored_warn_hours(self.db))
        ConfigRepository(self.db).set("strategy", "not-json")
        self.assertEqual(read_stored_strategy(self.db), {})
        self.assertIsNone(read_stored_warn_hours(self.db))
        self.assertEqual(read_stored_strategy(None), {})

    def test_round_trip(self):
        write_strategy(self.db, {"min_confidence": 55, "warn_hours": 3})
        self.assertEqual(read_stored_strategy(self.db)["min_confidence"], 55)
        self.assertEqual(read_stored_warn_hours(self.db), 3.0)

    def test_bad_warn_hours_ignored(self):
        write_strategy(self.db, {"warn_hours": -2})
        self.assertIsNone(read_stored_warn_hours(self.db))
        write_strategy(self.db, {"warn_hours": "soon"})
        self.assertIsNone(read_stored_warn_hours(self.db))


# ── engine + scheduler wiring ────────────────────────────────────────────


@pytest.mark.unit
class TestEngineUsesStored(unittest.TestCase):
    def test_stored_confidence_applies_without_restart(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        analysis = make_analysis("LONG", confidence=55.0)
        # Env default (60) rejects; stored (50) creates.
        before = harness.engine.process(analysis)
        self.assertEqual(before.outcome.value, "REJECTED")
        write_strategy(harness.db, {"min_confidence": 50})
        after = harness.engine.process(analysis)
        self.assertEqual(after.outcome.value, "CREATED")

    def test_explicit_arg_still_wins(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        write_strategy(harness.db, {"min_confidence": 50})
        result = harness.engine.process(
            make_analysis("LONG", confidence=55.0),
            guardrails=GuardrailConfig(min_confidence=90),
        )
        self.assertEqual(result.outcome.value, "REJECTED")

    def test_validate_analysis_direct_unaffected(self):
        # No database in this path: explicit-only, backtests unchanged.
        with self.assertRaises(SignalValidationError):
            validate_analysis(make_analysis("LONG", confidence=55.0))

    def test_scheduler_applies_stored_warn_hours(self):
        from signal_engine import OneHourScheduler

        harness = TempSignalDb()
        self.addCleanup(harness.close)
        calendar = EventCalendar(
            fetcher=lambda url, timeout: (_ for _ in ()).throw(
                ConnectionError("offline")
            ),
            fallback_path="/nonexistent/ev.json",
        )
        sched = OneHourScheduler(
            harness.repository,
            engine=harness.engine,
            state=harness.state,
            event_calendar=calendar,
        )
        write_strategy(harness.db, {"warn_hours": 2})
        sched._event_watch(1_700_000_000_000)
        self.assertEqual(calendar.warn_hours, 2.0)


# ── settings service ─────────────────────────────────────────────────────


def make_service(env=None):
    tmp = tempfile.TemporaryDirectory()
    db = Database(os.path.join(tmp.name, "cfg.db"))
    db.initialize()
    service = ConfigService(
        db, secrets_path=os.path.join(tmp.name, "secrets.json"), env=env or {}
    )
    return tmp, service


@pytest.mark.unit
class TestStrategySettings(unittest.TestCase):
    def test_public_defaults(self):
        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        public = service.get_strategy_public()
        self.assertEqual(public["min_confidence"], 60)
        self.assertEqual(public["min_risk_reward"], "1.2")
        self.assertEqual(public["htf_bias"], 1)
        self.assertEqual(public["warn_hours"], 12.0)
        self.assertIsNone(public["updated_at"])

    def test_update_merges_and_validates(self):
        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        public = service.update_strategy({"min_confidence": 55, "htf_bias": 0})
        self.assertEqual(public["min_confidence"], 55)
        self.assertEqual(public["htf_bias"], 0)
        self.assertIsNotNone(public["updated_at"])
        # Partial update keeps the rest.
        public = service.update_strategy({"min_risk_reward": "2.0"})
        self.assertEqual(public["min_confidence"], 55)
        self.assertEqual(public["min_risk_reward"], "2.0")
        for bad in (
            {"min_confidence": 101},
            {"min_confidence": "high"},
            {"min_risk_reward": -1},
            {"htf_bias": 2},
            {"warn_hours": -1},
            {"nope": 1},
        ):
            with self.assertRaises(SettingsValidationError, msg=str(bad)):
                service.update_strategy(bad)

    def test_reset_clears_overlay(self):
        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        service.update_strategy({"min_confidence": 55})
        service.update_strategy({"reset": True})
        self.assertEqual(service.get_strategy_public()["min_confidence"], 60)

    def test_effective_overlay_end_to_end(self):
        tmp, service = make_service(env={"BTCUSDT_MIN_CONFIDENCE": "70"})
        self.addCleanup(tmp.cleanup)
        service.update_strategy({"min_risk_reward": "2.0"})
        effective = GuardrailConfig.from_stored(
            read_stored_strategy(service.database), service.env
        )
        self.assertEqual(effective.min_confidence, 70)
        self.assertEqual(effective.min_risk_reward, Decimal("2.0"))


# ── web ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStrategyWeb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "web.db"))
        self.db.initialize()
        self.service = ConfigService(
            self.db,
            secrets_path=os.path.join(self._tmp.name, "secrets.json"),
            env={},
        )
        self.app = WebApplication(self.db, config_service=self.service)
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.bound_port}"

    def _api(self, path, method="GET", body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_settings_round_trip(self):
        status, data = self._api("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["strategy"]["min_confidence"], 60)
        status, data = self._api(
            "/api/settings", "PUT", {"strategy": {"min_confidence": 55}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["strategy"]["min_confidence"], 55)
        # Effective on the daemon path with no restart.
        stored = read_stored_strategy(self.db)
        effective = GuardrailConfig.from_stored(stored, {})
        self.assertEqual(effective.min_confidence, 55)

    def test_strategy_ui_markers(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
        self.assertIn('id="strategyForm"', html)
        with urllib.request.urlopen(
            self.base + "/static/app.js", timeout=10
        ) as resp:
            js = resp.read().decode("utf-8")
        for marker in (
            "renderStrategyForm",
            "stConfidence",
            "stRiskReward",
            "stFunding",
            "stHtf",
            "saveStrategy",
            "resetStrategy",
        ):
            self.assertIn(marker, js)


if __name__ == "__main__":
    unittest.main()
