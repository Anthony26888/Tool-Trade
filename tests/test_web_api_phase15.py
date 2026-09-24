"""Phase 15 tests for the Web Dashboard server + JSON API (web/server.py).

An in-process ThreadingHTTPServer on an ephemeral port is exercised end to end
with the stdlib ``urllib`` client: static SPA serving, the read-only dashboard/
signals/trades/statistics/health endpoints, the Settings GET/PUT cycle, secret
non-leakage over HTTP, the 409 demo-settings lock, and the live connection /
Telegram test endpoints behind injected fakes.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from binance.client import BinanceConnectionError
from binance.market_data import Candle
from database.database import (
    CandleLogRepository,
    Database,
    DemoRepository,
    RuntimeStateRepository,
    SignalRepository,
)
from database.models import STATUS_OPEN, STATUS_SL_HIT, STATUS_TP_HIT
from demo.executor import DemoExecutor
from signal_engine.config import ConfigService, SettingsValidationError
from web.server import WebApplication, WebServer


def _request(url: str, method: str = "GET", body=None, headers=None):
    data = body if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method, headers=headers or {}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, raw
    except urllib.error.HTTPError as exc:  # noqa: F821 - deferred import below
        raw = exc.read().decode("utf-8")
        return exc.code, raw


class FakeResponse:
    def __init__(self, status_code=200, payload=None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class WebApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self._tmp.name, "web.db"))
        self.db.initialize()
        self.config_service = ConfigService(
            self.db,
            secrets_path=os.path.join(self._tmp.name, "secrets.json"),
            env={},
            transport_get=lambda url, timeout: FakeResponse(
                200, {"models": [{"name": "qwen3:4b"}]}
            ),
            telegram_sender=lambda config: SimpleNamespace(
                ok=True, message_id=7, error=None
            ),
        )
        self.app = WebApplication(self.db, config_service=self.config_service)
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.bound_port}"

    def tearDown(self) -> None:
        self.server.stop()
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------------

    def get_json(self, path: str) -> tuple[int, dict]:
        status, raw = _request(self.base + path)
        self.assertEqual(status, 200)
        return status, json.loads(raw)

    def put_settings(self, group: dict) -> tuple[int, dict]:

        status, raw = _request(
            self.base + "/api/settings", method="PUT", body=group
        )
        return status, json.loads(raw)

    def seed_demo_account(self) -> None:
        DemoRepository(self.db).ensure_account(
            name="demo",
            initial_balance=Decimal("1000"),
            margin_per_trade=Decimal("50"),
            leverage=10,
            risk_percent=Decimal("1"),
            fee_rate=Decimal("0.04"),
        )

    def seed_active_signal(self) -> int:
        return SignalRepository(self.db).create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        ).id

    # -- static + routing ------------------------------------------------------

    def test_index_html_and_static_assets(self):
        status, raw = _request(self.base + "/", method="GET")
        self.assertEqual(status, 200)
        self.assertIn("<title>NEXTRA AI", raw)
        self.assertIn("/static/app.js", raw)

        status, raw = _request(self.base + "/static/app.js", method="GET")
        self.assertEqual(status, 200)
        self.assertIn("loadDashboard", raw)

        status, raw = _request(self.base + "/static/style.css", method="GET")
        self.assertEqual(status, 200)
        self.assertIn(".sidebar", raw)

    def test_brand_logo_served_as_png(self):
        resp = urllib.request.urlopen(self.base + "/static/ai.png", timeout=10)
        try:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get_content_type(), "image/png")
            self.assertGreater(len(resp.read()), 1000)
        finally:
            resp.close()

    def test_index_has_nextra_branding_and_favicon(self):
        status, raw = _request(self.base + "/", method="GET")
        self.assertEqual(status, 200)
        self.assertIn("NEXTRA AI", raw)
        self.assertIn('rel="icon"', raw)
        self.assertIn('class="brand-logo"', raw)
        self.assertIn("/static/ai.png", raw)

        status, raw = _request(self.base + "/nope", method="GET")
        self.assertEqual(status, 404)

        # Path traversal must be blocked.
        status, _ = _request(self.base + "/static/../pyproject.toml", method="GET")
        self.assertEqual(status, 404)

    def test_dashboard_spa_uses_flattened_api_shape(self):
        # Regression: the SPA must read top-level keys from /api/dashboard
        # (account/statistics/health/...), NOT a nested ".dashboard" object that
        # the API never returns (caused "reading 'account' of undefined").
        status, raw = _request(self.base + "/static/app.js", method="GET")
        self.assertEqual(status, 200)
        self.assertIn('const d = await api("/api/dashboard");', raw)
        self.assertNotIn('api("/api/dashboard")).dashboard', raw)

    # -- dashboard/health ------------------------------------------------------

    def test_health_endpoint(self):
        _, data = self.get_json("/api/health")
        health = data["health"]
        for key in ("state", "scheduler", "monitor", "scheduler_last_tick", "monitor_last_poll", "last_error", "last_error_at"):
            self.assertIn(key, health)

    def test_health_with_daemon_heartbeats_is_recent(self):
        # Regression: once the daemon has heartbeated, the web health endpoint
        # must NOT crash with now=None (TZ comparison) and must report RUNNING.
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        repo = RuntimeStateRepository(self.db)
        repo.set("scheduler.last_tick", now)
        repo.set("monitor.last_poll", now)
        _, data = self.get_json("/api/health")
        self.assertEqual(data["health"]["scheduler"], "RUNNING")
        self.assertEqual(data["health"]["monitor"], "RUNNING")

    def test_post_clear_last_error(self):
        repo = RuntimeStateRepository(self.db)
        repo.set("runtime.last_error", "stale boom")
        repo.set("runtime.last_error_at", "2026-09-15T00:00:00Z")
        status, raw = _request(
            self.base + "/api/runtime/clear-error", method="POST", body={}
        )
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data["health"]["last_error"], "")
        self.assertEqual(data["health"]["last_error_at"], "")
        self.assertEqual(repo.get("runtime.last_error"), "")
        self.assertEqual(repo.get("runtime.last_error_at"), "")

    def test_dashboard_lazily_creates_demo_account(self):
        # No account exists and the daemon never ran: the dashboard must
        # bootstrap the demo account at the resolved config instead of showing
        # an empty panel (AGENTS.md sections 17 and 23).
        self.config_service.update_demo(
            {
                "initial_balance": "500",
                "margin_per_trade": "25",
                "leverage": 5,
                "risk_percent": "1",
                "fee_rate": "0.0004",
            }
        )
        _, data = self.get_json("/api/dashboard")
        self.assertEqual(data["account"]["balance"], "500")
        self.assertEqual(data["account"]["margin_per_trade"], "25")
        self.assertEqual(data["account"]["leverage"], 5)
        self.assertIsNotNone(data["statistics"])
        self.assertEqual(data["statistics"]["total_trades"], 0)
        self.assertIsNone(data["active_signal"])

        # Idempotent: a second read must not reset the created account.
        _, again = self.get_json("/api/dashboard")
        self.assertEqual(again["account"]["balance"], "500")

        # Statistics participates in the same lazy bootstrap.
        _, stats = self.get_json("/api/statistics")
        self.assertEqual(stats["account"]["balance"], "500")

    def test_dashboard_preserves_existing_account(self):
        self.seed_demo_account()
        _, data = self.get_json("/api/dashboard")
        self.assertEqual(data["account"]["balance"], "1000")
        # Lazy bootstrap must never override an existing account balance.
        self.assertEqual(data["account"]["margin_per_trade"], "50")

    def test_dashboard_with_account_and_signal(self):
        self.seed_demo_account()
        signal_id = self.seed_active_signal()
        DemoRepository(self.db).create_position(
            account_id=1,
            signal_id=signal_id,
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("100.00"),
            quantity=Decimal("5.0"),
            position_size=Decimal("500.00"),
            margin=Decimal("50.00"),
            leverage=10,
            stop_loss=Decimal("99.00"),
            take_profit=Decimal("101.00"),
        )
        _, data = self.get_json("/api/dashboard")
        self.assertEqual(data["account"]["balance"], "1000")
        self.assertEqual(data["active_signal"]["direction"], "LONG")
        self.assertEqual(data["position"]["side"], "LONG")
        self.assertEqual(data["statistics"]["total_trades"], 0)

    # -- info endpoints ---------------------------------------------------------

    def test_signals_and_trades_endpoints(self):
        self.seed_demo_account()
        self.seed_active_signal()
        _, signals = self.get_json("/api/signals?limit=10")
        self.assertEqual(len(signals["signals"]), 1)
        self.assertEqual(signals["signals"][0]["status"], "PENDING_ENTRY")

    def test_dashboard_positions_list(self):
        self.seed_demo_account()
        sig_id = self.seed_active_signal()
        _, data = self.get_json("/api/dashboard")
        positions = data["positions"]
        self.assertEqual(len(positions), 1)
        item = positions[0]
        self.assertEqual(item["signal"]["id"], sig_id)
        self.assertEqual(item["signal"]["symbol"], "BTCUSDT")
        self.assertIsNone(item["position"])
        self.assertIsNone(item["unrealized"])
        # Backward-compatible singular keys stay intact.
        self.assertEqual(data["active_signal"]["id"], sig_id)
        # A second active row (post-B1 multi-symbol state, inserted raw)
        # appears in the list with its own position block.
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO signals (symbol, timeframe, direction, entry, "
                "stop_loss, take_profit, status, created_at) VALUES "
                "('ETHUSDT', '1h', 'SHORT', '3500', '3600', '3400', "
                "'PENDING_ENTRY', '2026-09-20T00:00:00.000Z')"
            )
        _, data = self.get_json("/api/dashboard")
        self.assertEqual(len(data["positions"]), 2)
        symbols = sorted(p["signal"]["symbol"] for p in data["positions"])
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])

        _, trades = self.get_json("/api/trades?limit=10")
        self.assertEqual(trades["count"], 0)
        self.assertEqual(trades["trades"], [])

        _, stats = self.get_json("/api/statistics")
        self.assertEqual(stats["account"]["leverage"], 10)
        self.assertIsNotNone(stats["statistics"])

    def test_trades_carry_position_symbol(self):
        from database.database import DemoRepository

        self.seed_demo_account()
        demo = DemoRepository(self.db)
        account = demo.get_account("demo")
        repo = SignalRepository(self.db)
        seen = {}
        for symbol in ("BTCUSDT", "ETHUSDT"):
            sig = repo.create_signal(
                symbol, "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
            )
            repo.transition_signal(sig.id, STATUS_OPEN)
            repo.transition_signal(
                sig.id, STATUS_TP_HIT, close_price="101.00", result="WIN"
            )
            pos = demo.create_position(
                account_id=account["id"],
                signal_id=sig.id,
                symbol=symbol,
                side="LONG",
                entry_price=Decimal("100"),
                quantity=Decimal("5"),
                position_size=Decimal("500"),
                margin=Decimal("50"),
                leverage=10,
                stop_loss=Decimal("99.00"),
                take_profit=Decimal("101.00"),
            )
            demo.record_trade(
                position_id=pos["id"],
                signal_id=sig.id,
                account_id=account["id"],
                side="LONG",
                entry_price=Decimal("100"),
                exit_price=Decimal("101.00"),
                quantity=Decimal("5"),
                margin=Decimal("50"),
                position_size=Decimal("500"),
                leverage=10,
                gross_pnl=Decimal("5"),
                fee=Decimal("0.20"),
                net_pnl=Decimal("4.80"),
                pnl_percent=Decimal("0.96"),
                result="WIN",
            )
            seen[sig.id] = symbol
        _, data = self.get_json("/api/trades?limit=10")
        got = {t["signal_id"]: t["symbol"] for t in data["trades"]}
        self.assertEqual(got, seen)
        for t in data["trades"]:
            self.assertEqual(t["leverage"], 10)
            self.assertTrue(t["opened_at"])
            self.assertTrue(t["closed_at"])

    def test_signals_and_daemon_rows_carry_symbol(self):
        repo = SignalRepository(self.db)
        sig_id = repo.create_signal(
            "ETHUSDT", "1h", "SHORT", Decimal("100.00"), Decimal("101.00"), Decimal("99.00")
        ).id
        _, signals = self.get_json("/api/signals?limit=10")
        row = next(s for s in signals["signals"] if s["id"] == sig_id)
        self.assertEqual(row["symbol"], "ETHUSDT")
        clog = CandleLogRepository(self.db)
        clog.upsert(
            symbol="ETHUSDT", timeframe="1h", candle_timestamp_ms=1_720_000_000_000,
            outcome="WAIT", decision="WAIT",
        )
        _, data = self.get_json("/api/daemon-log?limit=10")
        entry = next(r for r in data["rows"] if r["candle_timestamp_ms"] == 1_720_000_000_000)
        self.assertEqual(entry["symbol"], "ETHUSDT")

    # -- signals filters + delete -------------------------------------------------

    def test_signals_filter_params(self):
        repo = SignalRepository(self.db)
        long_id = repo.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        ).id
        repo.transition_signal(long_id, STATUS_OPEN)
        repo.transition_signal(long_id, STATUS_TP_HIT, close_price="101.00", result="WIN")
        repo.create_signal(
            "BTCUSDT", "1h", "SHORT", Decimal("100.00"), Decimal("101.50"), Decimal("99.00")
        )
        long_created = repo.get_signal(long_id).created_at
        _, only_long = self.get_json("/api/signals?direction=LONG")
        self.assertEqual([s["id"] for s in only_long["signals"]], [long_id])
        _, pending = self.get_json("/api/signals?status=PENDING_ENTRY")
        self.assertEqual(len(pending["signals"]), 1)
        _, empty = self.get_json("/api/signals?direction=SHORT&status=TP_HIT")
        self.assertEqual(empty["signals"], [])
        _, window = self.get_json(
            "/api/signals?since=" + long_created + "&until=" + long_created
        )
        self.assertEqual([s["id"] for s in window["signals"]], [long_id])

    def test_invalid_filter_status_returns_400(self):
        status, raw = _request(self.base + "/api/signals?status=BOGUS")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(raw))

    def test_signals_offset_pagination(self):
        repo = SignalRepository(self.db)
        ids = []
        for _ in range(3):
            sig_id = repo.create_signal(
                "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
            ).id
            repo.transition_signal(sig_id, STATUS_OPEN)
            repo.transition_signal(sig_id, STATUS_TP_HIT, close_price="101.00", result="WIN")
            ids.append(sig_id)
        _, page1 = self.get_json("/api/signals?limit=2&offset=0")
        _, page2 = self.get_json("/api/signals?limit=2&offset=2")
        self.assertEqual([s["id"] for s in page1["signals"]], ids[::-1][:2])
        self.assertEqual([s["id"] for s in page2["signals"]], ids[::-1][2:])
        _, negative = self.get_json("/api/signals?limit=2&offset=-5")
        self.assertEqual([s["id"] for s in negative["signals"]], ids[::-1][:2])

    def test_delete_signals_endpoint(self):
        repo = SignalRepository(self.db)
        signal_id = repo.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        ).id
        status, raw = _request(
            self.base + "/api/signals", method="DELETE", body={"ids": [signal_id]}
        )
        self.assertEqual(status, 200)
        counts = json.loads(raw)["counts"]
        self.assertEqual(counts["deleted"], [signal_id])
        _, data = self.get_json("/api/signals")
        self.assertEqual(data["signals"], [])

    def test_delete_all_skips_open(self):
        repo = SignalRepository(self.db)
        closed_id = repo.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        ).id
        repo.transition_signal(closed_id, STATUS_OPEN)
        repo.transition_signal(
            closed_id, STATUS_TP_HIT, close_price="101.00", result="WIN"
        )
        open_id = repo.create_signal(
            "BTCUSDT", "1h", "SHORT", Decimal("100.00"), Decimal("101.50"), Decimal("99.00")
        ).id
        repo.transition_signal(open_id, STATUS_OPEN)
        status, raw = _request(
            self.base + "/api/signals", method="DELETE", body={"all": True}
        )
        self.assertEqual(status, 200)
        counts = json.loads(raw)["counts"]
        self.assertEqual(counts["deleted"], [closed_id])
        self.assertEqual(counts["skipped_open"], [open_id])
        _, data = self.get_json("/api/signals")
        self.assertEqual([s["id"] for s in data["signals"]], [open_id])

    def test_delete_invalid_body_returns_400(self):
        status, raw = _request(
            self.base + "/api/signals", method="DELETE", body={"ids": []}
        )
        self.assertEqual(status, 400)
        status, raw = _request(
            self.base + "/api/signals", method="DELETE", body={}
        )
        self.assertEqual(status, 400)

    # -- settings cycle ---------------------------------------------------------

    def test_settings_get_never_leaks_secret_over_http(self):
        self.config_service.update_ai_provider(
            {"provider": "api", "preset": "deepseek", "model": "deepseek-chat",
             "api_key": "sk-WEB-SECRET-KEY"}
        )
        status, raw = _request(self.base + "/api/settings", method="GET")
        self.assertEqual(status, 200)
        self.assertNotIn("sk-WEB-SECRET-KEY", raw)
        data = json.loads(raw)["settings"]
        self.assertTrue(data["ai_provider"]["api_key_configured"])

    def test_put_ai_provider_applies_when_idle(self):
        status, data = self.put_settings(
            {"ai_provider": {"provider": "ollama", "model": "qwen3:4b"}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["settings"]["config_state"], "applied")
        self.assertEqual(data["settings"]["ai_provider"]["model"], "qwen3:4b")
        self.assertEqual(data["settings"]["ai_provider"]["analysis_mode"], "single")

    def test_put_ai_provider_analysis_mode_round_trip(self):
        status, data = self.put_settings(
            {"ai_provider": {
                "provider": "ollama",
                "model": "qwen3:4b",
                "analysis_mode": "multi",
            }}
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            data["settings"]["ai_provider"]["analysis_mode"], "multi"
        )
        status, data = self.put_settings(
            {"ai_provider": {
                "provider": "ollama",
                "model": "qwen3:4b",
                "analysis_mode": "committee",
            }}
        )
        self.assertEqual(status, 400)
        self.assertIs(data["ok"], False)

    def test_put_ai_provider_validation_error(self):
        status, data = self.put_settings(
            {"ai_provider": {"provider": "api", "preset": "custom", "model": ""}}
        )
        self.assertEqual(status, 400)
        self.assertIs(data["ok"], False)

    def test_put_demo_locked_returns_409(self):
        self.seed_active_signal()
        status, data = self.put_settings(
            {"demo": {
                "initial_balance": "1000",
                "margin_per_trade": "50",
                "leverage": 10,
                "risk_percent": "1",
                "fee_rate": "0.04",
            }}
        )
        self.assertEqual(status, 409)
        self.assertIn("locked", data["error"])

    def test_put_demo_updates_and_dashboard_uses_stored(self):
        self.seed_demo_account()
        status, data = self.put_settings(
            {"demo": {
                "initial_balance": "1000",
                "margin_per_trade": "75",
                "leverage": 5,
                "risk_percent": "2",
                "fee_rate": "0.05",
            }}
        )
        self.assertEqual(status, 200)
        row = DemoRepository(self.db).get_account("demo")
        self.assertEqual(row["margin_per_trade"], "75")
        self.assertEqual(row["leverage"], 5)
        # balance is immutable from the web layer
        self.assertEqual(row["balance"], "1000")

    def test_post_demo_reset_fresh_start(self):
        self.seed_demo_account()
        status, data = self.put_settings(
            {"demo": {
                "initial_balance": "2000",
                "margin_per_trade": "50",
                "leverage": 10,
                "risk_percent": "1",
                "fee_rate": "0.04",
            }}
        )
        self.assertEqual(status, 200)

        status, raw = _request(self.base + "/api/demo/reset", method="POST", body={})
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data["settings"]["initial_balance"], "2000")
        row = DemoRepository(self.db).get_account("demo")
        self.assertEqual(row["initial_balance"], "2000")
        self.assertEqual(row["balance"], "2000")
        self.assertEqual(row["equity"], "2000")
        self.assertEqual(row["peak_equity"], "2000")

    def test_post_demo_reset_locked_returns_409(self):
        self.seed_active_signal()
        status, raw = _request(self.base + "/api/demo/reset", method="POST", body={})
        self.assertEqual(status, 409)
        data = json.loads(raw)
        self.assertIn("locked", data["error"])

    def test_put_telegram_and_test_send(self):
        status, data = self.put_settings(
            {"telegram": {"enabled": True, "chat_id": "987654", "bot_token": "711:AA-web"}}
        )
        self.assertEqual(status, 200)
        self.assertTrue(data["settings"]["telegram"]["token_configured"])

        status, raw = _request(
            self.base + "/api/settings/test-telegram", method="POST", body={}
        )
        self.assertEqual(status, 200)
        result = json.loads(raw)["result"]
        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], 7)

    # -- connection tests -------------------------------------------------------

    def test_test_connection_ollama_via_fake_transport(self):
        status, data = self.get_json("/api/settings/ollama/models?base_url=http://localhost:11434")
        self.assertEqual(data["models"], ["qwen3:4b"])

        _, result = self.get_json("/api/settings")
        self.assertIsInstance(result["settings"]["audit"], list)

        status, raw = _request(
            self.base + "/api/settings/test-connection",
            method="POST",
            body={"provider": "ollama", "model": "qwen3:4b", "base_url": "http://localhost:11434"},
        )
        self.assertEqual(status, 200)
        conn = json.loads(raw)["result"]
        self.assertTrue(conn["success"])
        self.assertTrue(conn["model_available"])

    def test_unknown_api_route_is_404(self):
        status, _ = _request(self.base + "/api/does-not-exist", method="GET")
        self.assertEqual(status, 404)

    # -- TP/SL outcome preview --------------------------------------------------

    def test_dashboard_position_outcomes_when_open(self):
        repo = SignalRepository(self.db)
        signal = repo.create_signal(
            symbol="BTCUSDT",
            timeframe="1h",
            direction="LONG",
            entry="100.00",
            stop_loss="99.00",
            take_profit="101.00",
        )
        repo.transition_signal(signal.id, STATUS_OPEN)
        executor = DemoExecutor(self.db)
        executor.open_position(repo.get_signal(signal.id))

        _, data = self.get_json("/api/dashboard")
        outcomes = data["position_outcomes"]
        self.assertEqual(outcomes["take_profit"]["exit_price"], "101.00")
        self.assertEqual(outcomes["stop_loss"]["exit_price"], "99.00")
        self.assertEqual(outcomes["take_profit"]["result"], "WIN")
        self.assertEqual(outcomes["stop_loss"]["result"], "LOSS")

        # The preview must be internally consistent with the live account:
        # projected_balance = balance + net_pnl for both scenarios.
        balance = Decimal(data["account"]["balance"])
        for key in ("take_profit", "stop_loss"):
            self.assertEqual(
                Decimal(outcomes[key]["projected_balance"]),
                balance + Decimal(outcomes[key]["net_pnl"]),
            )
            self.assertIn("gross_pnl", outcomes[key])
            self.assertIn("fee", outcomes[key])
            self.assertIn("pnl_percent", outcomes[key])

    def test_dashboard_position_outcomes_absent_when_idle(self):
        status, data = self.get_json("/api/dashboard")
        self.assertIsNone(data["position_outcomes"])
        self.assertEqual(status, 200)

    def test_daemon_log_endpoint_list(self):
        signal_id = SignalRepository(self.db).create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("61000"), Decimal("60000"), Decimal("64000")
        ).id
        CandleLogRepository(self.db).upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_000_000_000,
            outcome="CREATED", decision="LONG", confidence=82,
            entry="61000", stop_loss="60000", take_profit="64000",
            signal_id=signal_id, provider="deepseek", model="deepseek-v4-flash",
            reasoning="uptrend", indicators_json='{"rsi14": 55.5}',
        )
        CandleLogRepository(self.db).upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_003_600_000,
            outcome="WAIT", decision="WAIT", confidence=60, reasoning="nothing",
        )
        _, data = self.get_json("/api/daemon-log")
        self.assertEqual(data["count"], 2)
        rows = {r["id"]: r for r in data["rows"]}
        newest = data["rows"][0]
        self.assertEqual(newest["decision"], "WAIT")
        self.assertEqual(newest["outcome"], "WAIT")
        created = data["rows"][1]
        self.assertEqual(created["decision"], "LONG")
        self.assertEqual(created["signal_id"], signal_id)
        self.assertEqual(created["entry"], "61000")
        self.assertEqual(created["indicators"]["rsi14"], 55.5)
        self.assertEqual(created["model"], "deepseek-v4-flash")
        self.assertIn(signal_id, rows)

    def test_daemon_log_filters_by_decision(self):
        repo = CandleLogRepository(self.db)
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_000_000_000,
            outcome="CREATED", decision="LONG", confidence=82,
        )
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_003_600_000,
            outcome="WAIT", decision="WAIT", confidence=60,
        )
        _, data = self.get_json("/api/daemon-log?decision=LONG&limit=10")
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["rows"][0]["decision"], "LONG")

    def test_daemon_log_empty(self):
        _, data = self.get_json("/api/daemon-log")
        self.assertEqual(data["count"], 0)
        self.assertEqual(data["rows"], [])

    def test_clear_daemon_log_endpoint(self):
        repo = CandleLogRepository(self.db)
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_000_000_000,
            outcome="WAIT", decision="WAIT",
        )
        status, raw = _request(self.base + "/api/daemon-log", method="DELETE")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["cleared"], 1)
        _, data = self.get_json("/api/daemon-log")
        self.assertEqual(data["count"], 0)

    def test_daemon_log_invalid_decision_returns_400(self):
        status, raw = _request(self.base + "/api/daemon-log?decision=ROCKET")
        self.assertEqual(status, 400)

    def test_delete_daemon_log_rows_by_ids(self):
        repo = CandleLogRepository(self.db)
        for ts in (1_720_000_000_000, 1_720_003_600_000, 1_720_007_200_000):
            repo.upsert(
                symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=ts,
                outcome="WAIT", decision="WAIT",
            )
        _, data = self.get_json("/api/daemon-log?limit=10")
        ids = sorted(r["id"] for r in data["rows"])
        self.assertEqual(len(ids), 3)
        status, raw = _request(
            self.base + "/api/daemon-log", method="DELETE",
            body={"ids": [ids[0], ids[2], 999999, -5, "x", True]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["deleted"], 2)
        _, data = self.get_json("/api/daemon-log?limit=10")
        self.assertEqual([r["id"] for r in data["rows"]], [ids[1]])
        # Empty selection is a no-op; missing body still clears all.
        status, raw = _request(
            self.base + "/api/daemon-log", method="DELETE", body={"ids": []}
        )
        self.assertEqual(json.loads(raw)["cleared"], 1)
        _, data = self.get_json("/api/daemon-log?limit=10")
        self.assertEqual(data["rows"], [])

    def test_daemon_log_indicator_json_is_parsed(self):
        repo = CandleLogRepository(self.db)
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_000_000_000,
            outcome="CREATED", decision="SHORT", confidence=70,
            indicators_json='{"ema20": 60000, "volume_ratio": 1.4}',
        )
        _, data = self.get_json("/api/daemon-log")
        indicators = data["rows"][0]["indicators"]
        self.assertEqual(indicators["ema20"], 60000)
        self.assertEqual(indicators["volume_ratio"], 1.4)

    def test_daemon_log_includes_quota_fields(self):
        repo = CandleLogRepository(self.db)
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_000_000_000,
            outcome="CREATED", decision="LONG", llm_calls=5,
            prompt_tokens=10500, completion_tokens=1800, total_tokens=12300,
        )
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_003_600_000,
            outcome="WAIT", decision="WAIT",
        )
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=1_720_007_200_000,
            outcome="CREATED", decision="SHORT", llm_calls=1,
            prompt_tokens=1600, completion_tokens=120, total_tokens=1720,
            tokens_estimated=True,
        )
        _, data = self.get_json("/api/daemon-log")
        by_ts = {r["candle_timestamp_ms"]: r for r in data["rows"]}
        created = by_ts[1_720_000_000_000]
        self.assertEqual(created["llm_calls"], 5)
        self.assertEqual(created["prompt_tokens"], 10500)
        self.assertEqual(created["completion_tokens"], 1800)
        self.assertEqual(created["total_tokens"], 12300)
        self.assertEqual(created["tokens_estimated"], 0)
        wait = by_ts[1_720_003_600_000]
        self.assertIsNone(wait["llm_calls"])
        self.assertIsNone(wait["total_tokens"])
        estimated = by_ts[1_720_007_200_000]
        self.assertEqual(estimated["tokens_estimated"], 1)
        self.assertEqual(estimated["prompt_tokens"], 1600)


def _sample_candles(count: int = 30) -> list[Candle]:
    start = 1_700_000_000_000
    step = 3_600_000
    return [
        Candle(
            timestamp=start + i * step,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
            close_time=start + i * step + step - 1,
            is_closed=True,
        )
        for i in range(count)
    ]


class FakeMarketData:
    def __init__(self, candles=None, error=None) -> None:
        self._candles = candles if candles is not None else _sample_candles()
        self._error = error
        self.calls: list[tuple[str, str, int]] = []

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        self.calls.append((symbol, interval, limit))
        if self._error is not None:
            raise self._error
        return self._candles[:limit]


class ChartApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self._tmp.name, "web.db"))
        self.db.initialize()
        self.market = FakeMarketData()
        self.app = WebApplication(
            self.db,
            market_data=self.market,
            config_service=ConfigService(
                self.db,
                secrets_path=os.path.join(self._tmp.name, "secrets.json"),
                env={},
                transport_get=lambda url, timeout: FakeResponse(
                    200, {"models": [{"name": "qwen3:4b"}]}
                ),
            ),
        )
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.bound_port}"

    def tearDown(self) -> None:
        self.server.stop()
        self._tmp.cleanup()

    def get_json(self, path: str) -> dict:
        status, raw = _request(self.base + path)
        self.assertEqual(status, 200)
        return json.loads(raw)

    def seed_active_signal(self) -> None:
        SignalRepository(self.db).create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        )

    # -- /api/chart -----------------------------------------------------------

    def test_chart_returns_candles_and_idle_overlay(self):
        data = self.get_json("/api/chart?interval=1h&limit=30")
        chart = data["chart"]
        self.assertEqual(len(chart["candles"]), 30)
        first = chart["candles"][0]
        self.assertEqual(first["t"], 1_700_000_000_000)
        self.assertEqual(first["o"], 100.0)
        self.assertEqual(first["h"], 101.0)
        self.assertEqual(first["l"], 99.0)
        self.assertEqual(first["c"], 100.5)
        self.assertEqual(first["v"], 1000.0)
        self.assertTrue(first["is_closed"])
        self.assertEqual(chart["overlay"], {})

    def test_chart_overlay_includes_active_signal_levels(self):
        self.seed_active_signal()
        data = self.get_json("/api/chart?interval=1h")
        overlay = data["chart"]["overlay"]
        self.assertTrue(overlay["has_signal"])
        self.assertEqual(overlay["direction"], "LONG")
        self.assertEqual(overlay["status"], "PENDING_ENTRY")
        self.assertEqual(overlay["entry"], "100.00")
        self.assertEqual(overlay["stop_loss"], "99.00")
        self.assertEqual(overlay["take_profit"], "101.00")
        self.assertFalse(overlay["position_open"])

    def test_chart_requires_known_interval(self):
        status, raw = _request(self.base + "/api/chart?interval=30m")
        self.assertEqual(status, 400)
        body = json.loads(raw)
        self.assertFalse(body["ok"])
        self.assertIn("interval", body["error"])
        self.assertEqual(self.market.calls, [])

    def test_chart_symbol_param_scopes_data_and_overlay(self):
        self.seed_active_signal()
        data = self.get_json("/api/chart?interval=1h&limit=30&symbol=ETHUSDT")
        self.assertEqual(data["chart"]["overlay"], {})
        self.assertIn(("ETHUSDT", "1h", 30), self.market.calls)
        data = self.get_json("/api/chart?interval=1h&limit=30&symbol=BTCUSDT")
        overlay = data["chart"]["overlay"]
        self.assertTrue(overlay["has_signal"])
        self.assertEqual(overlay["direction"], "LONG")

    def test_chart_rejects_unknown_symbol(self):
        status, raw = _request(self.base + "/api/chart?symbol=DOGE")
        self.assertEqual(status, 400)
        body = json.loads(raw)
        self.assertFalse(body["ok"])
        self.assertIn("symbol", body["error"].lower())

    def test_dashboard_includes_symbols_combo(self):
        data = self.get_json("/api/dashboard")
        self.assertEqual(data["symbols"], ["BTCUSDT"])

    def test_chart_maps_binance_failure_to_502(self):
        failing = FakeMarketData(error=BinanceConnectionError("socket timeout"))
        app = WebApplication(
            self.db,
            market_data=failing,
            config_service=ConfigService(
                self.db,
                secrets_path=os.path.join(self._tmp.name, "secrets.json"),
                env={},
            ),
        )
        server = WebServer(app, host="127.0.0.1", port=0)
        server.start()
        try:
            status, raw = _request(f"http://127.0.0.1:{server.bound_port}/api/chart")
        finally:
            server.stop()
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(raw), {"ok": False, "error": "market data unavailable"})

    def test_chart_caches_successful_snapshots_within_ttl(self):
        self.get_json("/api/chart?interval=1h")
        self.get_json("/api/chart?interval=1h")
        self.assertEqual(len(self.market.calls), 1)

    def test_chart_clamps_limit_to_minimum_ten(self):
        data = self.get_json("/api/chart?interval=1h&limit=3")
        self.assertEqual(len(data["chart"]["candles"]), 10)
        self.assertEqual(self.market.calls[-1][2], 10)

    def test_chart_accepts_all_selector_intervals(self):
        for interval in ("1m", "5m", "15m", "1h", "4h"):
            with self.subTest(interval=interval):
                data = self.get_json("/api/chart?interval=" + interval)
                self.assertEqual(len(data["chart"]["candles"]), 30)
                self.assertEqual(self.market.calls[-1][1], interval)

    def test_static_app_js_contains_chart_renderer(self):
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        self.assertIn("drawPriceChart", raw)
        self.assertIn("loadChart", raw)
        # The live-price label is drawn even while a signal overlay is visible.
        self.assertIn('fillStyle = "#f5c542"', raw)
        # TP/SL/Entry chips anchor to the LEFT edge so the right-side live-price
        # chip never moves off its true price position.
        self.assertIn("fillRect(pad.left + 2, y - 8, tw, chipH)", raw)
        # Client-side zoom/pan keeps a view window over the fetched candles.
        self.assertIn("chartView", raw)
        self.assertIn('addEventListener("wheel"', raw)
        self.assertIn("&limit=500", raw)

    def test_static_index_contains_chart_card(self):
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn('id="priceChart"', raw)
        self.assertIn('id="chartInterval"', raw)
        for control in ("chartZoomIn", "chartZoomOut", "chartPanLeft", "chartPanRight", "chartReset", "chartRange"):
            self.assertIn('id="' + control + '"', raw)

    # -- live PnL (unrealized) -------------------------------------------------

    def _open_long_position(self) -> None:
        repo = SignalRepository(self.db)
        signal = repo.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        )
        repo.transition_signal(signal.id, STATUS_OPEN)
        DemoExecutor(self.db).open_position(repo.get_signal(signal.id))

    def test_dashboard_unrealized_when_position_open(self):
        self._open_long_position()
        data = self.get_json("/api/dashboard")
        unrealized = data["unrealized"]
        self.assertIsNotNone(unrealized)
        pos = data["position"]
        qty = Decimal(pos["quantity"])
        margin = Decimal(pos["margin"])
        # FakeMarketData returns one 1m candle closing at 100.5; LONG entry 100.
        expected_gross = (Decimal("100.5") - Decimal("100.00")) * qty
        self.assertEqual(Decimal(unrealized["gross_pnl"]), expected_gross)
        self.assertEqual(
            Decimal(unrealized["pnl_percent"]),
            expected_gross / margin * 100,
        )
        self.assertEqual(unrealized["result"], "WIN")

    def test_dashboard_unrealized_absent_when_idle(self):
        data = self.get_json("/api/dashboard")
        self.assertIsNone(data["unrealized"])
        self.assertEqual(self.market.calls, [])

    def test_dashboard_unrealized_hidden_when_price_unavailable(self):
        failing = FakeMarketData(error=BinanceConnectionError("boom"))
        app = WebApplication(
            self.db,
            market_data=failing,
            config_service=ConfigService(
                self.db,
                secrets_path=os.path.join(self._tmp.name, "secrets.json"),
                env={},
            ),
        )
        server = WebServer(app, host="127.0.0.1", port=0)
        server.start()
        try:
            status, raw = _request(f"http://127.0.0.1:{server.bound_port}/api/dashboard")
            self.assertEqual(status, 200)
            self.assertIsNone(json.loads(raw)["unrealized"])
        finally:
            server.stop()


# -- symbol switching ------------------------------------------------------

    def test_dashboard_reports_configured_symbol(self):
        data = self.get_json("/api/dashboard")
        self.assertEqual(data["symbol"], "BTCUSDT")

    def test_settings_returns_symbol_block(self):
        raw = self.get_json("/api/settings")
        self.assertEqual(raw["settings"]["symbol"]["symbol"], "BTCUSDT")
        self.assertEqual(
            raw["settings"]["symbol"]["supported"], ["BTCUSDT", "ETHUSDT", "XAUUSDT"]
        )

    def test_settings_put_switches_symbol(self):
        status, raw = _request(
            self.base + "/api/settings",
            method="PUT",
            body={"symbol": {"symbol": "ETHUSDT"}},
        )
        self.assertEqual(status, 200)
        settings = json.loads(raw)["settings"]
        self.assertEqual(settings["symbol"]["symbol"], "ETHUSDT")
        self.assertEqual(self.get_json("/api/dashboard")["symbol"], "ETHUSDT")

    def test_settings_put_rejects_unknown_symbol(self):
        status, raw = _request(
            self.base + "/api/settings",
            method="PUT",
            body={"symbol": {"symbol": "SOLUSDT"}},
        )
        self.assertEqual(status, 400)
        self.assertIn("symbol", json.loads(raw)["error"])

    def test_chart_serves_configured_symbol(self):
        status, raw = _request(
            self.base + "/api/settings",
            method="PUT",
            body={"symbol": {"symbol": "XAUUSDT"}},
        )
        self.assertEqual(status, 200)
        self.market.calls.clear()
        data = self.get_json("/api/chart?interval=1h&limit=30")
        self.assertEqual(len(data["chart"]["candles"]), 30)
        self.assertEqual(self.market.calls[-1][0], "XAUUSDT")

    def test_dashboard_unrealized_uses_position_symbol_not_configured(self):
        # Regression: opening PnL must use the OPEN position's own symbol price,
        # never the newly-changed Settings symbol (would show nonsense PnL).
        self._open_long_position()
        status, raw = _request(
            self.base + "/api/settings",
            method="PUT",
            body={"symbol": {"symbol": "ETHUSDT"}},
        )
        self.assertEqual(status, 200)
        self.market.calls.clear()
        data = self.get_json("/api/dashboard")
        self.assertIsNotNone(data["unrealized"])
        self.assertEqual(self.market.calls[-1][0], "BTCUSDT")
        self.assertEqual(self.market.calls[-1][1], "1m")

    def test_chart_overlay_for_mismatched_symbol_is_empty(self):
        # Regression: with a BTCUSDT signal active and the symbol switched to
        # ETHUSDT, the ETH chart must not stretch its price axis with unrelated
        # BTC levels (which rendered ETH candles invisible).
        self.seed_active_signal()
        status, raw = _request(
            self.base + "/api/settings",
            method="PUT",
            body={"symbol": {"symbol": "ETHUSDT"}},
        )
        self.assertEqual(status, 200)
        self.market.calls.clear()
        data = self.get_json("/api/chart?interval=1h&limit=30")
        self.assertEqual(data["chart"]["overlay"], {})
        self.assertIn(("ETHUSDT", "1h", 30), self.market.calls)

    # -- static symbol UI ------------------------------------------------------

    def test_static_contains_symbol_settings_ui(self):
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        for marker in ("symbolForm", "symbolSelect", "saveSymbol", "priceCardTitle"):
            self.assertIn('id="' + marker + '"', raw)
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in ("renderSymbolForm", "window._symbol", '"/api/settings"'):
            self.assertIn(marker, raw)

    def test_static_contains_analysis_mode_settings_ui(self):
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in ('id="analysisMode"', "analysis_mode", "Multi-agent debate"):
            self.assertIn(marker, raw)

    def test_static_contains_reset_demo_ui(self):
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            'id="resetDemo"',
            '"/api/demo/reset"',
            "Reset Demo Account",
            "fresh start",
        ):
            self.assertIn(marker, raw)

    def test_static_contains_clear_last_error_ui(self):
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        for marker in ('id="clearErrorBtn"', "Clear last error"):
            self.assertIn(marker, raw)
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            "clearErrorBtn",
            '"/api/runtime/clear-error"',
            "Clear the stored",
        ):
            self.assertIn(marker, raw)

    def test_static_contains_mobile_responsive_ui(self):
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        for marker in (
            'class="bottom-nav"',
            'data-panel="dashboard"',
            'class="nav-item nav-center"',
            'id="settingsGear"',
            'name="theme-color"',
        ):
            self.assertIn(marker, raw)
        status, raw = _request(self.base + "/static/style.css")
        self.assertEqual(status, 200)
        for marker in (
            "max-width: 900px",
            "max-width: 720px",
            ".nav-center",
            "attr(data-label)",
            "safe-area-inset-bottom",
        ):
            self.assertIn(marker, raw)
        self.assertGreaterEqual(
            raw.count("padding-bottom: calc(100px + env(safe-area-inset-bottom))"),
            2,
            "bottom padding clearance must survive the 720px override",
        )
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in ("data-label=", "settingsGear", 'showPanel("settings")'):
            self.assertIn(marker, raw)

    def test_static_contains_loading_overlay(self):
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        for marker in ('id="loading"', 'class="loading hidden"', "loading-logo", "/static/ai.png"):
            self.assertIn(marker, raw)
        status, raw = _request(self.base + "/static/style.css")
        self.assertEqual(status, 200)
        for marker in (".loading {", ".loading-ring", ".loading-ring::before", ".loading-logo", "112px", "@keyframes load-spin"):
            self.assertIn(marker, raw)
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in ("showLoader", "hideLoader", "Promise.allSettled(jobs)", "loadSeq"):
            self.assertIn(marker, raw)

    def test_static_daemon_pager(self):
        import shutil
        import subprocess

        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn('id="daemonPager"', raw)
        self.assertIn('id="signalsPager"', raw)
        status, js = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            "DD_PAGE_SIZE",
            "renderDaemonPager",
            'data-pg="prev"',
            'data-pg="next"',
            "pager-label",
            "ddPage = 0",
        ):
            self.assertIn(marker, js)
        status, css = _request(self.base + "/static/style.css")
        self.assertEqual(status, 200)
        self.assertIn(".pager", css)
        if shutil.which("node") is None:
            self.skipTest("node is not installed")

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            "let ddPage = 0;\n"
            "const DD_PAGE_SIZE = 50;\n"
            "let _el = null;\n"
            "const $ = () => _el;\n"
            "const t = (k, vs) => { let s = {'pager.prev': '‹ Prev', 'pager.next': 'Next ›', 'pager.label': 'Page {n} · rows {from}'}[k] || k; "
            "if (vs) for (const [kk, vv] of Object.entries(vs)) s = s.split('{' + kk + '}').join(vv); return s; };\n"
            + _extract_fn("pagerHtml") + "\n" + _extract_fn("renderDaemonPager") + "\n"
            "const out = [];\n"
            "const snap = () => ({ hidden: _el.hidden, html: _el.innerHTML });\n"
            "_el = { hidden: false, innerHTML: '' };\n"
            "ddPage = 0; renderDaemonPager(false); out.push(['first-no-more', snap()]);\n"
            "ddPage = 0; renderDaemonPager(true); out.push(['first-more', snap()]);\n"
            "ddPage = 2; renderDaemonPager(true); out.push(['mid-more', snap()]);\n"
            "ddPage = 2; renderDaemonPager(false); out.push(['mid-last', snap()]);\n"
            "console.log(JSON.stringify(out));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        states = dict(json.loads(proc.stdout.strip().splitlines()[-1]))
        # Single short page: pager hides itself.
        self.assertTrue(states["first-no-more"]["hidden"])
        # First page with more: prev disabled, next enabled, label page 1.
        first = states["first-more"]
        self.assertFalse(first["hidden"])
        self.assertIn("Page 1", first["html"])
        self.assertIn('data-pg="prev" disabled', first["html"])
        self.assertNotIn('data-pg="next" disabled', first["html"])
        # Middle page: both enabled.
        mid = states["mid-more"]
        self.assertIn("Page 3", mid["html"])
        self.assertNotIn("disabled", mid["html"])
        # Last page: next disabled, prev enabled.
        last = states["mid-last"]
        self.assertIn('data-pg="next" disabled', last["html"])
        self.assertNotIn('data-pg="prev" disabled', last["html"])

    def test_position_cards_render(self):
        import shutil
        import subprocess

        status, js = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            "renderPositionCards",
            "pos-card",
            "pos-grid",
            "pos.waiting",
            "dash.statusMulti",
            "baseAsset",
        ):
            self.assertIn(marker, js)
        if shutil.which("node") is None:
            self.skipTest("node is not installed")

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        helpers = "\n".join(
            _extract_fn(n)
            for n in (
                "emptyBox",
                "badge",
                "escapeHtml",
                "baseAsset",
                "renderPositionCards",
            )
        )
        stubs = (
            "const fmtNum = (v, d) => Number(v).toFixed(d);\n"
            "const fmtVn = (s) => s;\n"
        )
        program = (
            "const window = { _symbol: 'BTCUSDT' };\n"
            "const t = (k) => k;\n"
            + stubs + helpers + "\n"
            "const OPEN = {signal: {symbol: 'BTCUSDT', direction: 'SHORT', confidence: 75, entry: 80310, stop_loss: 81003, take_profit: 78067, model_name: 'm', analysis_timestamp: null},"
            " position: {side: 'SHORT', quantity: 0.000955, margin: 10, position_size: 81.24, leverage: 50, symbol: 'BTCUSDT'},"
            " position_outcomes: {take_profit: {net_pnl: 2.37}, stop_loss: {net_pnl: -1.07}},"
            " unrealized: {gross_pnl: 0.92, pnl_percent: 9.2}};\n"
            "const PENDING = {signal: {symbol: 'ETHUSDT', direction: 'LONG', confidence: 66, entry: 3500, stop_loss: 3450, take_profit: 3600, model_name: 'm', analysis_timestamp: null},"
            " position: null, position_outcomes: null, unrealized: null};\n"
            "const html = renderPositionCards([OPEN, PENDING]);\n"
            "const empty = renderPositionCards([]);\n"
            "console.log(JSON.stringify([html, empty]));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        html, empty = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertIn("pos-card short", html)
        self.assertIn("BTCUSDT", html)
        self.assertIn("+$0.92", html)
        self.assertIn("pos.waiting", html)
        self.assertIn("ETHUSDT", html)
        self.assertIn("dash.noSignal", empty)
        self.assertIn("pos-side", html)
        self.assertIn("lev-chip", html)
        self.assertIn(">50x<", html)
        self.assertIn("pos-line entry", html)
        self.assertIn("data-close-sid", html)
        self.assertNotIn("pos-hero", html)
        self.assertNotIn("pos-actions", html)

    def test_portfolio_select_render(self):
        import shutil
        import subprocess

        status, js = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in ("chartSymbol", "renderPortfolioBar", "id=\"chartSymbol\""):
            self.assertIn(marker, js + _request(self.base + "/")[1])
        self.assertIn("lastPortfolio", js)
        status, css = _request(self.base + "/static/style.css")
        self.assertEqual(status, 200)
        self.assertIn(".portfolio-dot.ready", css)
        self.assertIn(".pos-grid { display: grid; grid-template-columns: 1fr;", css)
        if shutil.which("node") is None:
            self.skipTest("node is not installed")

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            "let chartSymbol = null;\n"
            "let _el = null;\n"
            "const $ = () => _el;\n"
            "const t = (k) => k;\n"
            "const escapeHtml = (s) => s;\n"
            + _extract_fn("renderPortfolioBar") + "\n"
            "_el = { hidden: true, innerHTML: '' };\n"
            "renderPortfolioBar({symbols: [{symbol: 'BTCUSDT', state: 'OPEN'}, {symbol: 'ETHUSDT', state: 'SEEKING'}], full: false}, ['BTCUSDT', 'ETHUSDT']);\n"
            "const mixed = _el.innerHTML;\n"
            "const sel = chartSymbol;\n"
            "chartSymbol = 'ETHUSDT';\n"
            "renderPortfolioBar({symbols: [{symbol: 'BTCUSDT', state: 'OPEN'}, {symbol: 'ETHUSDT', state: 'OPEN'}], full: true}, ['BTCUSDT', 'ETHUSDT']);\n"
            "const full = _el.innerHTML;\n"
            "chartSymbol = 'XAUUSDT';\n"
            "renderPortfolioBar({symbols: [{symbol: 'XAUUSDT', state: 'SEEKING'}], full: false}, ['XAUUSDT']);\n"
            "const seeking = _el.innerHTML;\n"
            "console.log(JSON.stringify([mixed, sel, full, seeking]));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        mixed, sel, full, seeking = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertIn('<option value="BTCUSDT" selected>', mixed)
        self.assertIn('<option value="ETHUSDT">', mixed)
        self.assertEqual(sel, "BTCUSDT")
        # Chip follows the dropdown selection (BTC holding -> amber, no ETH chip).
        self.assertIn("BTCUSDT", mixed)
        self.assertIn("portfolio-dot holding", mixed)
        self.assertNotIn("ETHUSDT</strong>", mixed)
        self.assertIn("port.full", full)
        # Seeking chip follows selection with the green ready dot, exactly once.
        self.assertIn("XAUUSDT", seeking)
        self.assertIn("portfolio-dot ready", seeking)
        self.assertEqual(seeking.count("XAUUSDT</strong>"), 1)

    def test_static_signals_pager(self):
        import shutil
        import subprocess

        status, js = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            "SIG_PAGE_SIZE",
            "renderSignalsPager",
            "sigPage = 0",
            "signalsPager",
            "dd-check",
            "ddSelAll",
            "ddDeleteSel",
            "updateDdDeleteCount",
            "daemon.selectAll",
            "daemon.deleteSel",
            "daemon.confirmDelSel",
        ):
            self.assertIn(marker, js)
        if shutil.which("node") is None:
            self.skipTest("node is not installed")

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            "let sigPage = 0;\n"
            "const SIG_PAGE_SIZE = 25;\n"
            "let _el = null;\n"
            "const $ = () => _el;\n"
            "const t = (k, vs) => { let s = {'pager.prev': '‹ Prev', 'pager.next': 'Next ›', 'pager.label': 'Page {n} · rows {from}'}[k] || k; "
            "if (vs) for (const [kk, vv] of Object.entries(vs)) s = s.split('{' + kk + '}').join(vv); return s; };\n"
            + _extract_fn("pagerHtml") + "\n" + _extract_fn("renderSignalsPager") + "\n"
            "_el = { hidden: false, innerHTML: '' };\n"
            "sigPage = 0; renderSignalsPager(false);\n"
            "const hidden = _el.hidden;\n"
            "sigPage = 1; renderSignalsPager(true);\n"
            "const mid = _el.innerHTML;\n"
            "console.log(JSON.stringify([hidden, mid]));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        hidden, mid = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(hidden)
        self.assertIn("Page 2", mid)

    def test_dd_delete_count_logic(self):
        import shutil
        import subprocess

        status, js = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        if shutil.which("node") is None:
            self.skipTest("node is not installed")

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            "let CHECKED = 0;\n"
            "const BTN = { disabled: true, textContent: '' };\n"
            "const document = { querySelectorAll: (sel) => Array.from({length: CHECKED}, () => ({})) };\n"
            "const $ = (sel) => BTN;\n"
            "const t = (k, vs) => k === 'daemon.deleteSel' ? `Delete selected (${vs.n})` : k;\n"
            + _extract_fn("ddSelectedCount") + "\n" + _extract_fn("updateDdDeleteCount") + "\n"
            "CHECKED = 0; updateDdDeleteCount();\n"
            "const dis0 = BTN.disabled;\n"
            "CHECKED = 3; updateDdDeleteCount();\n"
            "console.log(JSON.stringify([dis0, BTN.disabled, BTN.textContent]));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        dis0, dis3, label = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(dis0)
        self.assertFalse(dis3)
        self.assertEqual(label, "Delete selected (3)")

    def test_static_daemon_detail_sections(self):
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            "dd-sec-title",
            "dd-levels",
            "dd-level sl",
            "dd-level tp",
            "dd-stats",
            "dd-stat",
            "dd-ind-grid",
            "dd-ind-group",
            "Tokens in/out/total",
            "(~)",
            "tokens_estimated",
            "Nhận định AI",
        ):
            self.assertIn(marker, raw)
        status, css = _request(self.base + "/static/style.css")
        self.assertEqual(status, 200)
        for marker in (
            ".dd-sec-title",
            ".dd-levels",
            ".dd-level.sl strong",
            ".dd-level.tp strong",
            ".dd-stats",
            ".dd-ind-grid",
        ):
            self.assertIn(marker, css)

    def test_dd_risk_reward_logic(self):
        import shutil
        import subprocess

        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        start = raw.index("function ddRiskReward(")
        depth = 0
        for pos in range(start, len(raw)):
            if raw[pos] == "{":
                depth += 1
            elif raw[pos] == "}":
                depth -= 1
                if depth == 0:
                    fn = raw[start : pos + 1]
                    break
        program = (
            "const fmtNum = (v, d) => Number(v).toFixed(d);\n" + fn + "\nconsole.log(JSON.stringify(["
            "ddRiskReward({decision:'LONG',entry:80310,stop_loss:81003,take_profit:78067}),"
            "ddRiskReward({decision:'SHORT',entry:80310,stop_loss:81003,take_profit:78067}),"
            "ddRiskReward({decision:'WAIT'}),"
            "ddRiskReward({decision:'LONG',entry:1,stop_loss:1,take_profit:1}),"
            "]));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        long_inverted, short_valid, wait_rr, flat_rr = json.loads(
            proc.stdout.strip().splitlines()[-1]
        )
        # LONG 80310/81003/78067 is inverted -> no RR; SHORT gives ~3.24.
        self.assertEqual(long_inverted, "")
        self.assertTrue(short_valid.startswith("RR 3.2"))
        self.assertEqual(wait_rr, "")
        self.assertEqual(flat_rr, "")

    def test_static_daemon_table_shows_quota(self):
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in (
            't("daemon.thCalls")',
            'data-label="${t("daemon.thCalls")}"',
            'colspan="11"',
            "daemon.kTokens",
            "daemon.kTokensEst",
            "tokens_estimated",
            "ddLevelsSection",
            "ddQuotaSection",
            "ddIndicatorsSection",
            "ddRiskReward",
            "daemon.thSymbol",
            "sig.thSymbol",
            "trades.thSymbol",
        ):
            self.assertIn(marker, raw)


class FakeChatResponder:
    """Fake LLM chat: yields configured text chunks, recording the call."""

    def __init__(self, chunks=("Hello", " there"), fail_at=None) -> None:
        self.chunks = chunks
        self.fail_at = fail_at
        self.calls = 0
        self.last_messages = None

    def __call__(self, messages, llm_config):
        self.calls += 1
        self.last_messages = messages
        for i, chunk in enumerate(self.chunks):
            if self.fail_at is not None and i >= self.fail_at:
                raise RuntimeError("provider boom")
            yield chunk


class SignalChatApiTestCase(unittest.TestCase):
    def make_app(self, responder=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self._tmp.name, "chat.db"))
        self.db.initialize()
        config_service = ConfigService(
            self.db,
            secrets_path=os.path.join(self._tmp.name, "secrets.json"),
            env={},
        )
        self.responder = responder
        self.app = WebApplication(
            self.db, config_service=config_service, chat_responder=responder
        )
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.bound_port}"

    def tearDown(self) -> None:
        self.server.stop()
        self._tmp.cleanup()

    def seed_signal(self) -> int:
        return SignalRepository(self.db).create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        ).id

    def post_chat(self, body: dict):
        return _request(self.base + "/api/signal-chat", method="POST", body=body)

    # -- happy path -----------------------------------------------------------

    def test_streams_answer_chunks_with_signal_context(self):
        self.make_app(responder=FakeChatResponder(chunks=("Hel", "lo", "!")))
        self.seed_signal()
        status, raw = self.post_chat({"question": "Why this signal?"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in raw.strip().splitlines()]
        text = "".join(e.get("delta", "") for e in events)
        self.assertEqual(text, "Hello!")
        self.assertEqual(self.responder.calls, 1)
        user = self.responder.last_messages[1]["content"]
        self.assertIn("Why this signal?", user)
        self.assertIn('"direction": "LONG"', user)
        self.assertIn('"statistics"', user)
        self.assertIn('"account_risk"', user)

    def test_chat_is_read_only_no_db_writes(self):
        self.make_app(responder=FakeChatResponder(chunks=("ok",)))
        self.seed_signal()

        def counts():
            conn = self.db.connect()
            try:
                return (
                    conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM demo_positions").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM demo_trades").fetchone()[0],
                )
            finally:
                conn.close()

        before = counts()
        self.post_chat({"question": "risk?"})
        self.assertEqual(before, counts())

    def test_idle_returns_canned_without_calling_llm(self):
        self.make_app(responder=FakeChatResponder(chunks=("nope",)))
        status, raw = self.post_chat({"question": "what signal?"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in raw.strip().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertIn("active signal", events[0]["text"])
        self.assertEqual(self.responder.calls, 0)

    # -- validation -----------------------------------------------------------

    def test_empty_question_is_rejected(self):
        self.make_app(responder=FakeChatResponder())
        self.seed_signal()
        status, raw = self.post_chat({"question": "   "})
        self.assertEqual(status, 400)
        self.assertIn("question", json.loads(raw)["error"])
        self.assertEqual(self.responder.calls, 0)

    def test_overlong_question_is_rejected(self):
        self.make_app(responder=FakeChatResponder())
        self.seed_signal()
        status, raw = self.post_chat({"question": "x" * 1001})
        self.assertEqual(status, 400)
        self.assertIn("too long", json.loads(raw)["error"])
        self.assertEqual(self.responder.calls, 0)

    # -- failure handling -----------------------------------------------------

    def test_mid_stream_failure_emits_error_event(self):
        self.make_app(responder=FakeChatResponder(chunks=("Hi", "?"), fail_at=1))
        self.seed_signal()
        status, raw = self.post_chat({"question": "hi"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in raw.strip().splitlines()]
        self.assertEqual(events[0]["delta"], "Hi")
        self.assertIn("error", events[1])

    def test_unconfigured_provider_returns_502(self):
        from unittest import mock

        from signal_engine.llm import LLMConfigError

        self.make_app(responder=None)
        self.seed_signal()
        with mock.patch(
            "web.server.build_llm_client",
            side_effect=LLMConfigError("missing API key"),
        ):
            status, raw = self.post_chat({"question": "risk?"})
        self.assertEqual(status, 502)
        self.assertIn("AI provider", json.loads(raw)["error"])

    # -- static UI ------------------------------------------------------------

    def test_static_contains_signal_chat_ui(self):
        self.make_app(responder=FakeChatResponder(chunks=("hi",)))
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        for marker in ("chatOpenBtn", "chatDrawer", "chatBackdrop", "chatInput", "chatSend", "chatMessages"):
            self.assertIn('id="' + marker + '"', raw)
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        for marker in ("openSignalChat", "sendSignalChat", "/api/signal-chat", "chat-chip"):
            self.assertIn(marker, raw)

    def test_settings_audit_is_own_card_with_table(self):
        self.make_app(responder=FakeChatResponder(chunks=("hi",)))
        status, raw = _request(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("Settings Audit", raw)
        self.assertIn('id="auditBox"', raw)
        self.assertNotIn('id="runtimeBox" class="form"></div>\n            <div id="auditBox"', raw)
        status, raw = _request(self.base + "/static/app.js")
        self.assertEqual(status, 200)
        self.assertIn("renderAudit", raw)


class ManualCloseApiTestCase(unittest.TestCase):
    """POST /api/positions/:id/close ("Chốt"): OPEN only, market price exit,
    TP_HIT/SL_HIT by PnL sign with close_reason=MANUAL (no schema change)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self._tmp.name, "web.db"))
        self.db.initialize()
        self.market = FakeMarketData()
        self.app = WebApplication(
            self.db,
            market_data=self.market,
            config_service=ConfigService(
                self.db,
                secrets_path=os.path.join(self._tmp.name, "secrets.json"),
                env={},
                transport_get=lambda url, timeout: FakeResponse(
                    200, {"models": [{"name": "qwen3:4b"}]}
                ),
            ),
        )
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.bound_port}"
        self.signals = SignalRepository(self.db)

    def tearDown(self) -> None:
        self.server.stop()
        self._tmp.cleanup()

    def seed_open(self, direction: str = "LONG") -> int:
        stop, take = (Decimal("99.00"), Decimal("101.00")) if direction == "LONG" else (Decimal("101.00"), Decimal("99.00"))
        sig = self.signals.create_signal(
            "BTCUSDT", "1h", direction, Decimal("100.00"), stop, take
        )
        opened = self.signals.transition_signal(sig.id, STATUS_OPEN)
        position = DemoExecutor(self.db).open_position(opened)
        self.assertIsNotNone(position)
        return sig.id

    def test_close_profitable_long_lands_tp_hit_manual(self):
        sid = self.seed_open("LONG")  # market 100.5 >= entry 100
        before = DemoRepository(self.db).get_account("demo")
        payload = self.app.close_position(sid)
        self.assertEqual(payload["status"], STATUS_TP_HIT)
        self.assertEqual(payload["close_reason"], "MANUAL")
        self.assertEqual(payload["result"], "WIN")
        self.assertAlmostEqual(payload["close_price"], 100.5)
        self.assertGreater(payload["trade"]["net_pnl"], 0)
        after = DemoRepository(self.db).get_account("demo")
        self.assertGreater(after["balance"], before["balance"])
        row = self.signals.get_signal(sid)
        self.assertEqual(row.status, STATUS_TP_HIT)
        trade = DemoRepository(self.db).get_trade_for_signal(sid)
        self.assertIsNotNone(trade)
        self.assertEqual(trade["result"], "WIN")

    def test_close_losing_short_lands_sl_hit_manual(self):
        sid = self.seed_open("SHORT")  # market 100.5 > entry 100: SHORT loses
        payload = self.app.close_position(sid)
        self.assertEqual(payload["status"], STATUS_SL_HIT)
        self.assertEqual(payload["close_reason"], "MANUAL")
        self.assertEqual(payload["result"], "LOSS")
        self.assertLess(payload["trade"]["net_pnl"], 0)

    def test_close_rejects_non_open(self):
        sig = self.signals.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        )
        with self.assertRaises(SettingsValidationError):
            self.app.close_position(sig.id)

    def test_close_rejects_unknown_id(self):
        with self.assertRaises(SettingsValidationError):
            self.app.close_position(9999)
        with self.assertRaises(SettingsValidationError):
            self.app.close_position("abc")

    def test_close_rejects_open_without_position(self):
        sig = self.signals.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        )
        self.signals.transition_signal(sig.id, STATUS_OPEN)
        with self.assertRaises(SettingsValidationError):
            self.app.close_position(sig.id)

    def test_close_fails_when_price_unavailable_and_stays_open(self):
        sid = self.seed_open("LONG")
        self.market._error = BinanceConnectionError("down")
        self.app._price_cache = None
        with self.assertRaises(SettingsValidationError):
            self.app.close_position(sid)
        self.assertEqual(self.signals.get_signal(sid).status, STATUS_OPEN)

    def test_close_twice_reports_already_closed(self):
        sid = self.seed_open("LONG")
        self.app.close_position(sid)
        with self.assertRaisesRegex(SettingsValidationError, "only OPEN"):
            self.app.close_position(sid)

    def test_close_http_route(self):
        sid = self.seed_open("LONG")
        status, raw = _request(f"{self.base}/api/positions/{sid}/close", method="POST")
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(body["close"]["status"], STATUS_TP_HIT)
        self.assertEqual(body["close"]["close_reason"], "MANUAL")
        status, raw = _request(f"{self.base}/api/positions/9999/close", method="POST")
        self.assertEqual(status, 400)
        self.assertFalse(json.loads(raw)["ok"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
