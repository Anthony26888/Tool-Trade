"""Supervisor tests (plan B', phase B3): combo Settings + process planning.

Covers the enabled-combo read/fallback, OPEN-position discovery, the pure
convergence planner, pidfile handling, and sync/stop/status orchestration
with injected process fakes (no real daemons are ever spawned here).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
import unittest.mock
from types import SimpleNamespace

import pytest

from database.database import Database
from signal_engine import supervisor
from signal_engine.supervisor import (
    daemon_pids,
    is_alive,
    open_symbols,
    plan,
    read_enabled_symbols,
    read_pid,
    status,
    stop_all,
    sync,
)


def make_db(symbols=None, open_syms=()):
    tmp = tempfile.TemporaryDirectory()
    db = Database(os.path.join(tmp.name, "sup.db"))
    db.initialize()
    if symbols is not None:
        conn = sqlite3.connect(db.path)
        conn.execute(
            "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("symbols", json.dumps({"symbols": symbols}), "2026-09-24T00:00:00.000Z"),
        )
        conn.commit()
        conn.close()
    if open_syms:
        from database.database import SignalRepository

        repo = SignalRepository(db)
        for symbol in open_syms:
            sig = repo.create_signal(
                symbol, "1h", "LONG", "100", "99", "101", confidence=80
            )
            repo.transition_signal(sig.id, "OPEN")
    return tmp, db


def ps_match(text):
    def _ps(pid):
        del pid
        return text

    return _ps


@pytest.mark.unit
class TestEnabledAndOpenReads(unittest.TestCase):
    def test_combo_and_fallbacks(self):
        tmp, db = make_db(["ETHUSDT", "BTCUSDT", "ETHUSDT", "DOGE"])
        self.addCleanup(tmp.cleanup)
        # Order preserved, deduped, unknowns dropped.
        self.assertEqual(read_enabled_symbols(db.path), ["ETHUSDT", "BTCUSDT"])
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        fresh = Database(os.path.join(tmp2.name, "fresh.db"))
        fresh.initialize()
        self.assertEqual(read_enabled_symbols(fresh.path), ["BTCUSDT"])
        self.assertEqual(read_enabled_symbols("/nonexistent/x.db"), ["BTCUSDT"])

    def test_bad_documents_fall_back(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "bad.db"))
        db.initialize()
        for bad in ("not-json", "[1, 2]", '{"symbols": []}', '{"symbols": "BTC"}'):
            conn = sqlite3.connect(db.path)
            conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value, updated_at)"
                " VALUES (?, ?, ?)",
                ("symbols", bad, "2026-09-24T00:00:00.000Z"),
            )
            conn.commit()
            conn.close()
            self.assertEqual(read_enabled_symbols(db.path), ["BTCUSDT"], msg=bad)

    def test_open_symbols(self):
        tmp, db = make_db(open_syms=("BTCUSDT", "ETHUSDT"))
        self.addCleanup(tmp.cleanup)
        self.assertEqual(open_symbols(db.path), {"BTCUSDT", "ETHUSDT"})
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        fresh = Database(os.path.join(tmp2.name, "fresh.db"))
        fresh.initialize()
        self.assertEqual(open_symbols(fresh.path), set())


@pytest.mark.unit
class TestPlan(unittest.TestCase):
    def test_start_missing_stop_extra(self):
        actions = plan(["BTCUSDT", "ETHUSDT"], {"BTCUSDT": 11}, set())
        self.assertEqual(actions["start"], ["ETHUSDT"])
        self.assertEqual(actions["stop"], [])
        self.assertEqual(actions["skipped_blocked"], [])

    def test_open_never_stopped(self):
        actions = plan(["BTCUSDT"], {"BTCUSDT": 11, "ETHUSDT": 22}, {"ETHUSDT"})
        self.assertEqual(actions["start"], [])
        self.assertEqual(actions["stop"], [])
        self.assertEqual(actions["skipped_blocked"], ["ETHUSDT"])

    def test_idle_extra_stops(self):
        actions = plan(["BTCUSDT"], {"BTCUSDT": 11, "XAUUSDT": 33}, set())
        self.assertEqual(actions["stop"], ["XAUUSDT"])

    def test_unknown_symbols_ignored(self):
        actions = plan(["BTCUSDT", "DOGE"], {}, set())
        self.assertEqual(actions["start"], ["BTCUSDT"])

    def test_empty_desired_stops_everything_idle(self):
        actions = plan([], {"BTCUSDT": 11}, set())
        self.assertEqual(actions["stop"], ["BTCUSDT"])


@pytest.mark.unit
class TestPidfiles(unittest.TestCase):
    def test_is_alive(self):
        self.assertTrue(is_alive(os.getpid()))
        self.assertFalse(is_alive(999999999))

    def test_stale_pidfile_ignored(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "daemon-BTCUSDT.pid"), "w") as handle:
            handle.write("999999999\n")
        self.assertEqual(
            daemon_pids(tmp.name, ps_fn=ps_match("signal_engine daemon")), {}
        )
        self.assertIsNone(read_pid(tmp.name, "daemon-BTCUSDT.pid"))

    def test_live_pidfile_accepted_with_cmdline_guard(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, "daemon-ETHUSDT.pid"), "w") as handle:
            handle.write(f"{os.getpid()}\n")
        self.assertEqual(
            daemon_pids(tmp.name, ps_fn=ps_match("python -m signal_engine daemon")),
            {"ETHUSDT": os.getpid()},
        )
        self.assertEqual(
            daemon_pids(tmp.name, ps_fn=ps_match("something else")),
            {},
        )


@pytest.mark.unit
class TestSyncOrchestration(unittest.TestCase):
    def setUp(self):
        self.tmp, self.db = make_db(["BTCUSDT", "ETHUSDT"])
        self.addCleanup(self.tmp.cleanup)
        self.rundir = tempfile.TemporaryDirectory()
        self.addCleanup(self.rundir.cleanup)
        self.spawned: list = []
        self.killed: list = []
        self.pid_counter = [40000]
        self.live: set = set()
        self.roles: dict = {}

        def spawn(python_bin, args, env, cwd, log_path):
            self.pid_counter[0] += 1
            pid = self.pid_counter[0]
            self.spawned.append((args, env.get("BTCUSDT_SYMBOL"), log_path))
            self.live.add(pid)
            self.roles[pid] = args[0]
            return pid

        def kill(pid):
            self.killed.append(pid)
            self.live.discard(pid)

        def ps(pid):
            if pid not in self.live:
                return ""
            return f"python -m signal_engine {self.roles.get(pid, 'daemon')}"

        self.spawn = spawn
        self.kill = kill
        self.ps = ps
        # Fake pids (40001+...) are "alive" exactly while registered: this
        # exercises pidfile + liveness logic without real processes.
        patcher = unittest.mock.patch.object(
            supervisor, "is_alive", lambda pid: pid in self.live
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _sync(self, **kwargs):
        return sync(
            db_path=self.db.path,
            rundir=self.rundir.name,
            web_port=8129,
            python_bin="/usr/bin/python3",
            spawn_fn=self.spawn,
            kill_fn=self.kill,
            ps_fn=self.ps,
            **kwargs,
        )

    def test_sync_starts_missing_daemons_and_web(self):
        report = self._sync()
        started = sorted(s for _, s, _ in self.spawned if s)
        self.assertEqual(started, ["BTCUSDT", "ETHUSDT"])
        self.assertTrue(any(a == ["web", "--host", "127.0.0.1", "--port", "8129"] for a, _, _ in self.spawned))
        self.assertEqual(report["started"], ["BTCUSDT", "ETHUSDT"])
        self.assertTrue(report["web_started"])
        self.assertEqual(report["exit_code"], 0)
        # Second sync is a no-op (pidfiles + live checks agree).
        spawned_before = len(self.spawned)
        report2 = sync(
            db_path=self.db.path,
            rundir=self.rundir.name,
            web_port=8129,
            python_bin="/usr/bin/python3",
            spawn_fn=self.spawn,
            kill_fn=self.kill,
            ps_fn=self.ps,
        )
        self.assertEqual(report2["started"], [])
        self.assertEqual(report2["stopped"], [])
        self.assertFalse(report2["web_started"])
        self.assertEqual(len(self.spawned), spawned_before)

    def test_sync_blocks_open_and_stops_idle_extra(self):
        from database.database import SignalRepository

        repo = SignalRepository(self.db)
        sig = repo.create_signal(
            "ETHUSDT", "1h", "LONG", "100", "99", "101", confidence=80
        )
        repo.transition_signal(sig.id, "OPEN")
        # Pretend both daemons already run (live pidfiles).
        for symbol in ("BTCUSDT", "ETHUSDT"):
            self.pid_counter[0] += 1
            pid = self.pid_counter[0]
            self.live.add(pid)
            with open(
                os.path.join(
                    self.rundir.name, f"daemon-{symbol}.pid"
                ),
                "w",
            ) as handle:
                handle.write(str(pid))
        # Desired drops ETHUSDT while it holds OPEN.
        conn = sqlite3.connect(self.db.path)
        conn.execute(
            "UPDATE app_settings SET value = ? WHERE key = 'symbols'",
            (json.dumps({"symbols": ["BTCUSDT"]}),),
        )
        conn.commit()
        conn.close()

        report = sync(
            db_path=self.db.path,
            rundir=self.rundir.name,
            web_port=8129,
            python_bin="/usr/bin/python3",
            spawn_fn=self.spawn,
            kill_fn=self.kill,
            ps_fn=self.ps,
        )
        self.assertEqual(report["stopped"], [])
        self.assertEqual(report["skipped_blocked"], ["ETHUSDT"])
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(self.killed, [])

    def test_stop_all_respects_open(self):
        from database.database import SignalRepository

        repo = SignalRepository(self.db)
        sig = repo.create_signal(
            "BTCUSDT", "1h", "LONG", "100", "99", "101", confidence=80
        )
        repo.transition_signal(sig.id, "OPEN")
        for symbol in ("BTCUSDT", "ETHUSDT"):
            self.pid_counter[0] += 1
            pid = self.pid_counter[0]
            self.live.add(pid)
            with open(
                os.path.join(self.rundir.name, f"daemon-{symbol}.pid"), "w"
            ) as handle:
                handle.write(str(pid))

        report = stop_all(
            db_path=self.db.path, rundir=self.rundir.name,
            kill_fn=self.kill, ps_fn=self.ps,
        )
        self.assertEqual(report["stopped"], ["ETHUSDT"])
        self.assertEqual(report["skipped_blocked"], ["BTCUSDT"])
        self.assertEqual(report["exit_code"], 1)

    def test_status_shape(self):
        report = status(
            db_path=self.db.path, rundir=self.rundir.name, ps_fn=lambda pid: ""
        )
        self.assertEqual(report["desired"], ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(
            report["daemons"]["BTCUSDT"], {"pid": None, "open": False}
        )
        self.assertIsNone(report["web_pid"])
        self.assertEqual(report["open_symbols"], [])


@pytest.mark.unit
class TestSymbolsSettings(unittest.TestCase):
    def test_update_and_public(self):
        import tempfile as _tmp

        from signal_engine.config import ConfigService

        tmp = _tmp.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "cfg.db"))
        db.initialize()
        service = ConfigService(
            db, secrets_path=os.path.join(tmp.name, "secrets.json"), env={}
        )
        public = service.get_symbols_public()
        self.assertEqual(public["symbols"], ["BTCUSDT"])
        self.assertEqual(public["supported"], ["BTCUSDT", "ETHUSDT", "XAUUSDT"])
        updated = service.update_symbols({"symbols": ["ETHUSDT", "BTCUSDT", "ETHUSDT", "ethusdt"]})
        self.assertEqual(updated["symbols"], ["ETHUSDT", "BTCUSDT"])
        self.assertIsNotNone(updated["updated_at"])
        dumped = json.dumps(public)
        self.assertNotIn("secret", dumped.lower())

    def test_update_rejects(self):
        import tempfile as _tmp

        from signal_engine.config import ConfigService, SettingsValidationError

        tmp = _tmp.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "cfg.db"))
        db.initialize()
        service = ConfigService(
            db, secrets_path=os.path.join(tmp.name, "secrets.json"), env={}
        )
        for bad in ({"symbols": []}, {"symbols": ["DOGE"]}, {"symbols": "BTCUSDT"}, {}):
            with self.assertRaises(SettingsValidationError, msg=str(bad)):
                service.update_symbols(bad)


@pytest.mark.unit
class TestSymbolsWeb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "web.db"))
        self.db.initialize()
        from signal_engine.config import ConfigService
        from web.server import WebApplication, WebServer

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
        import urllib.request

        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_settings_symbols_round_trip(self):
        status, data = self._api("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["symbols"]["symbols"], ["BTCUSDT"])
        status, data = self._api(
            "/api/settings", "PUT", {"symbols": {"symbols": ["BTCUSDT", "ETHUSDT"]}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            data["settings"]["symbols"]["symbols"], ["BTCUSDT", "ETHUSDT"]
        )

    def test_symbols_ui_markers(self):
        import urllib.request

        with urllib.request.urlopen(self.base + "/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
        self.assertIn('id="symbolsForm"', html)
        with urllib.request.urlopen(
            self.base + "/static/app.js", timeout=10
        ) as resp:
            js = resp.read().decode("utf-8")
        for marker in (
            "renderSymbolsForm",
            "sym-check",
            "saveSymbols",
            "set.symbolsTitle",
            "set.saveSymbols",
            "set.symbolDisplay",
        ):
            self.assertIn(marker, js)


@pytest.mark.unit
class TestDaemonSymbolResolve(unittest.TestCase):
    def _service(self, env=None):
        from signal_engine.config import ConfigService

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "cfg.db"))
        db.initialize()
        return ConfigService(
            db, secrets_path=os.path.join(tmp.name, "secrets.json"), env=env or {}
        )

    def test_single_symbol_legacy_unchanged(self):
        service = self._service(env={"BTCUSDT_SYMBOL": "ETHUSDT"})
        self.assertEqual(service.resolve_daemon_symbol(), "ETHUSDT")
        service.update_symbol({"symbol": "XAUUSDT"})
        # Stored single beats env, exactly like resolve_symbol.
        self.assertEqual(service.resolve_daemon_symbol(), "XAUUSDT")
        self.assertEqual(
            service.resolve_daemon_symbol(), service.resolve_symbol()
        )

    def test_multi_combo_hands_authority_to_env(self):
        service = self._service(env={"BTCUSDT_SYMBOL": "ETHUSDT"})
        service.update_symbol({"symbol": "BTCUSDT"})
        service.update_symbols({"symbols": ["BTCUSDT", "ETHUSDT"]})
        # Legacy single says BTCUSDT, but the daemon's own env (ETHUSDT)
        # wins so N daemons on one DB analyse N symbols.
        self.assertEqual(service.resolve_daemon_symbol(), "ETHUSDT")

    def test_multi_combo_bad_env_falls_back(self):
        service = self._service(env={"BTCUSDT_SYMBOL": "DOGE"})
        service.update_symbols({"symbols": ["BTCUSDT", "ETHUSDT"]})
        self.assertEqual(service.resolve_daemon_symbol(), "BTCUSDT")

    def test_runtime_wiring_uses_daemon_resolver(self):
        from signal_engine.runtime import Runtime

        service = self._service(env={"BTCUSDT_SYMBOL": "ETHUSDT"})
        service.update_symbol({"symbol": "BTCUSDT"})
        service.update_symbols({"symbols": ["BTCUSDT", "ETHUSDT"]})
        runtime = Runtime.__new__(Runtime)
        runtime.config_service = service
        runtime.config = SimpleNamespace(symbol="BTCUSDT")
        self.assertEqual(runtime._resolving_daemon_symbol()(), "ETHUSDT")
        # Display path untouched.
        self.assertEqual(service.resolve_symbol(), "BTCUSDT")
