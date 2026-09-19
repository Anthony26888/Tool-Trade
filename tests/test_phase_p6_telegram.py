"""Phase P6a tests: dual-channel Telegram (signals vs reports) + daily report.

Covers channel routing with fallback, legacy single-chat compatibility, the
Refreshable send_message fix, Settings resolve/persist/test per channel, the
07:00 (+07) exactly-once daily report, and the web route + UI markers.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import unittest.mock
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from database.database import (
    Database,
    RuntimeStateRepository,
    SignalRepository,
)
from signal_engine.config import ConfigService
from signal_engine.daily_report import (
    REPORT_LAST_DAILY_KEY,
    build_daily_report,
    report_window,
    should_send_daily,
)
from signal_engine.telegram import (
    RefreshableTelegramNotifier,
    TelegramConfig,
    TelegramConfigError,
    TelegramNotifier,
    TelegramSendResult,
    telegram_config_from_env,
)
from web.server import WebApplication, WebServer


def enabled_config(**overrides) -> TelegramConfig:
    base = {
        "bot_token": "123456789:TESTtoken",
        "chat_id_signals": "-100111",
        "chat_id_reports": "-100222",
        "enabled": True,
    }
    base.update(overrides)
    return TelegramConfig(**base)


class Recorder:
    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, text: str) -> TelegramSendResult:
        self.calls.append(text)
        return TelegramSendResult(ok=True, message_id=1)


# ── routing ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDualChannelRouting(unittest.TestCase):
    def _notifier(self, config):
        notifier = TelegramNotifier(config)
        sig, rep = Recorder(), Recorder()
        notifier._signal_client.send_message = sig
        notifier._report_client.send_message = rep
        return notifier, sig, rep

    def test_signal_lifecycle_goes_to_signals(self):
        from tests.test_telegram import _signal

        notifier, sig, rep = self._notifier(enabled_config())
        notifier.notify_signal_created(_signal(), candle_ts=1)
        notifier.notify_signal_opened(_signal(status="OPEN"))
        self.assertEqual(len(sig.calls), 2)
        self.assertEqual(rep.calls, [])

    def test_generic_send_defaults_to_reports(self):
        notifier, sig, rep = self._notifier(enabled_config())
        notifier.send_message("notice")
        self.assertEqual(rep.calls, ["notice"])
        self.assertEqual(sig.calls, [])
        notifier.send_message("trade", channel="signal")
        self.assertEqual(sig.calls, ["trade"])

    def test_missing_channel_falls_back(self):
        only_signals = TelegramNotifier(
            TelegramConfig(bot_token="1:A", chat_id_signals="-100111", enabled=True)
        )
        self.assertEqual(only_signals._report_client.config.chat_id, "-100111")
        only_reports = TelegramNotifier(
            TelegramConfig(bot_token="1:A", chat_id_reports="-100222", enabled=True)
        )
        self.assertEqual(only_reports._signal_client.config.chat_id, "-100222")

    def test_legacy_single_chat_fills_both(self):
        config = TelegramConfig(bot_token="1:A", chat_id="-100999", enabled=True)
        self.assertEqual(config.signal_chat, "-100999")
        self.assertEqual(config.report_chat, "-100999")

    def test_disabled_sends_nothing(self):
        notifier, sig, rep = self._notifier(TelegramConfig(enabled=False))
        self.assertFalse(notifier.send_message("x").ok)
        self.assertEqual(sig.calls, [])
        self.assertEqual(rep.calls, [])

    def test_env_vars(self):
        config = telegram_config_from_env(
            {
                "BTCUSDT_TELEGRAM_ENABLED": "true",
                "BTCUSDT_TELEGRAM_BOT_TOKEN": "1:A",
                "BTCUSDT_TELEGRAM_CHAT_ID_SIGNALS": "-100111",
                "BTCUSDT_TELEGRAM_CHAT_ID_REPORTS": "-100222",
            }
        )
        self.assertEqual(config.signal_chat, "-100111")
        self.assertEqual(config.report_chat, "-100222")
        legacy = telegram_config_from_env(
            {
                "BTCUSDT_TELEGRAM_ENABLED": "true",
                "BTCUSDT_TELEGRAM_BOT_TOKEN": "1:A",
                "BTCUSDT_TELEGRAM_CHAT_ID": "-100999",
            }
        )
        self.assertEqual(legacy.signal_chat, "-100999")
        self.assertEqual(legacy.report_chat, "-100999")

    def test_enabled_requires_token_and_a_chat(self):
        with self.assertRaises(TelegramConfigError):
            telegram_config_from_env(
                {"BTCUSDT_TELEGRAM_ENABLED": "true", "BTCUSDT_TELEGRAM_BOT_TOKEN": "1:A"}
            )
        with self.assertRaises(TelegramConfigError):
            telegram_config_from_env(
                {
                    "BTCUSDT_TELEGRAM_ENABLED": "true",
                    "BTCUSDT_TELEGRAM_CHAT_ID_SIGNALS": "-100111",
                }
            )

    def test_refreshable_send_message_delegates(self):
        inner = TelegramNotifier(enabled_config())
        rec = Recorder()
        inner._report_client.send_message = rec
        outer = RefreshableTelegramNotifier(lambda: inner)
        result = outer.send_message("event notice")
        self.assertTrue(result.ok)
        self.assertEqual(rec.calls, ["event notice"])
        dead = RefreshableTelegramNotifier(lambda: None)
        self.assertFalse(dead.send_message("x").ok)


# ── settings service ─────────────────────────────────────────────────────


def make_service():
    tmp = tempfile.TemporaryDirectory()
    db = Database(os.path.join(tmp.name, "cfg.db"))
    db.initialize()
    service = ConfigService(
        db, secrets_path=os.path.join(tmp.name, "secrets.json"), env={}
    )
    return tmp, service


@pytest.mark.unit
class TestTelegramSettings(unittest.TestCase):
    def test_update_and_public_two_chats(self):
        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        service.update_telegram(
            {
                "enabled": True,
                "chat_id_signals": "-100111",
                "chat_id_reports": "-100222",
                "bot_token": "711:AA-secret-token",
            }
        )
        public = service.get_telegram_public()
        self.assertTrue(public["chat_id_signals_configured"])
        self.assertTrue(public["chat_id_reports_configured"])
        self.assertEqual(public["chat_id_signals_masked"], "********")
        dumped = json.dumps(public)
        self.assertNotIn("AA-secret-token", dumped)
        self.assertNotIn("-100111", dumped)
        config = service.telegram_config_object()
        self.assertEqual(config.signal_chat, "-100111")
        self.assertEqual(config.report_chat, "-100222")

    def test_legacy_chat_id_fills_both(self):
        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        service.update_telegram(
            {"enabled": True, "chat_id": "-100999", "bot_token": "711:AA-x"}
        )
        config = service.telegram_config_object()
        self.assertEqual(config.signal_chat, "-100999")
        self.assertEqual(config.report_chat, "-100999")

    def test_enabled_requires_a_chat(self):
        from signal_engine.config import SettingsValidationError

        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        with self.assertRaises(SettingsValidationError):
            service.update_telegram({"enabled": True, "bot_token": "711:AA-x"})

    def test_test_telegram_per_channel(self):
        tmp, service = make_service()
        self.addCleanup(tmp.cleanup)
        service.update_telegram(
            {
                "enabled": True,
                "chat_id_signals": "-100111",
                "chat_id_reports": "-100222",
                "bot_token": "711:AA-x",
            }
        )
        seen: list = []

        def fake_sender(config):
            seen.append(config.chat_id)
            return TelegramSendResult(ok=True, message_id=3)

        service._telegram_sender = fake_sender
        self.assertTrue(service.test_telegram("signals")["success"])
        self.assertTrue(service.test_telegram("reports")["success"])
        self.assertEqual(seen, ["-100111", "-100222"])


# ── daily report ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDailyReport(unittest.TestCase):
    def test_should_send_daily(self):
        # 06:59 VN -> no; 07:00 VN -> yes; same-day flag -> no.
        before = datetime(2026, 9, 19, 23, 59, tzinfo=timezone.utc)
        at = datetime(2026, 9, 20, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(should_send_daily(None, before))
        self.assertTrue(should_send_daily(None, at))
        self.assertFalse(should_send_daily("2026-09-20", at))
        # Restart next morning with yesterday's flag -> yes again.
        later = datetime(2026, 9, 21, 0, 30, tzinfo=timezone.utc)
        self.assertTrue(should_send_daily("2026-09-20", later))

    def test_report_window(self):
        at = datetime(2026, 9, 20, 0, 30, tzinfo=timezone.utc)  # 07:30 VN 20/09
        start, end, label = report_window(at)
        self.assertEqual(label, "19/09")
        self.assertEqual(start, "2026-09-18T17:00:00Z")
        self.assertEqual(end, "2026-09-19T17:00:00Z")

    def test_build_content(self):
        signals = [
            SimpleNamespace(decision="LONG"),
            SimpleNamespace(decision="SHORT"),
            SimpleNamespace(decision="WAIT"),
        ]
        trades = [
            SimpleNamespace(net_pnl=Decimal("12.40")),
            SimpleNamespace(net_pnl=Decimal("-8.10")),
        ]
        rows = [
            SimpleNamespace(outcome="WAIT"),
            SimpleNamespace(outcome="WAIT"),
            SimpleNamespace(outcome="CREATED"),
            SimpleNamespace(outcome="EVENT_BLACKOUT"),
            SimpleNamespace(outcome="REJECTED"),
        ]
        text = build_daily_report(
            signals=signals,
            trades=trades,
            candle_rows=rows,
            balance=Decimal("1004.30"),
            initial_balance=Decimal("1000"),
            last_error=None,
            day_label="19/09",
        )
        self.assertIn("19/09", text)
        self.assertIn("LONG 1 / SHORT 1 / WAIT 1", text)
        self.assertIn("+$4.30", text)
        self.assertIn("$1,004.30", text)
        self.assertIn("EVENT_BLACKOUT 1", text)
        self.assertIn("REJECTED 1", text)
        self.assertIn("Lỗi: không", text)

    def test_build_with_error_and_empty(self):
        text = build_daily_report(
            signals=[],
            trades=[],
            candle_rows=[],
            balance=None,
            initial_balance=None,
            last_error="boom",
            day_label="19/09",
        )
        self.assertIn("Signal: 0", text)
        self.assertIn("Lỗi: boom", text)

    def test_build_never_raises(self):
        text = build_daily_report(
            signals=object(),
            trades=object(),
            candle_rows=object(),
            balance="xx",
            initial_balance=None,
            last_error=None,
            day_label="19/09",
        )
        self.assertIn("19/09", text)


# ── runtime hook ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRuntimeDailyHook(unittest.TestCase):
    def test_maybe_daily_report_sends_once(self):
        import signal_engine.runtime as runtime_mod
        from signal_engine.runtime import Runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "rt.db"))
        db.initialize()
        sent: list = []

        class FakeNotifier:
            def send_message(self, text, *args, **kwargs):
                sent.append(text)
                return TelegramSendResult(ok=True, message_id=1)

        runtime = Runtime.__new__(Runtime)
        runtime.database = db
        runtime.repository = SignalRepository(db)
        runtime.executor = SimpleNamespace(account=lambda: None)
        runtime.state_store = RuntimeStateRepository(db)
        runtime.notifier = FakeNotifier()
        runtime._report_check_at = 0.0
        at = datetime(2026, 9, 20, 0, 30, tzinfo=timezone.utc)
        with unittest.mock.patch.object(
            runtime_mod.time, "monotonic", return_value=1000.0
        ):
            runtime._maybe_daily_report(now_utc=at)
            runtime._maybe_daily_report(now_utc=at)
        self.assertEqual(len(sent), 1)
        self.assertIn("19/09", sent[0])
        self.assertEqual(
            runtime.state_store.get(REPORT_LAST_DAILY_KEY), "2026-09-20"
        )

    def test_maybe_daily_report_before_hour(self):
        import signal_engine.runtime as runtime_mod
        from signal_engine.runtime import Runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "rt.db"))
        db.initialize()
        runtime = Runtime.__new__(Runtime)
        runtime.database = db
        runtime.state_store = RuntimeStateRepository(db)
        runtime.notifier = None
        runtime._report_check_at = 0.0
        at = datetime(2026, 9, 19, 23, 0, tzinfo=timezone.utc)  # 06:00 VN
        with unittest.mock.patch.object(
            runtime_mod.time, "monotonic", return_value=1000.0
        ):
            runtime._maybe_daily_report(now_utc=at)
        self.assertIsNone(runtime.state_store.get(REPORT_LAST_DAILY_KEY))


# ── web ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebTelegramChannels(unittest.TestCase):
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

    def _post(self, path, body):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_channel_param_routed(self):
        seen: list = []

        def fake_sender(config):
            seen.append(config.chat_id)
            return TelegramSendResult(ok=True, message_id=9)

        self.service.update_telegram(
            {
                "enabled": True,
                "chat_id_signals": "-100111",
                "chat_id_reports": "-100222",
                "bot_token": "711:AA-x",
            }
        )
        self.service._telegram_sender = fake_sender
        self._post("/api/settings/test-telegram", {"channel": "signals"})
        self._post("/api/settings/test-telegram", {"channel": "reports"})
        self.assertEqual(seen, ["-100111", "-100222"])

    def test_settings_ui_markers(self):
        with urllib.request.urlopen(self.base + "/static/app.js", timeout=10) as resp:
            js = resp.read().decode("utf-8")
        for marker in (
            "tgChatSignals",
            "tgChatReports",
            "testTelegramSignals",
            "testTelegramReports",
            "{ channel }",
        ):
            self.assertIn(marker, js)


if __name__ == "__main__":
    unittest.main()
