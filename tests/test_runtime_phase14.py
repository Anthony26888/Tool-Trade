"""Phase 14 unit tests for the production runtime + CLI (signal_engine/runtime.py,
signal_engine/__main__.py).

Covers: config from env, daemon startup recovery, the scheduler+monitor wiring
into the demo ledger and Telegram notifications (OPEN/TP/SL once, with PnL +
balance), the AI lock, `--once`, graceful shutdown, health heartbeats, the CLI
query commands, and JSON render path.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from binance.market_data import Candle
from database.database import (
    CandleLogRepository,
    Database,
    DemoRepository,
    RuntimeStateRepository,
    SignalRepository,
)
from database.models import STATUS_OPEN, STATUS_PENDING_ENTRY, STATUS_TP_HIT
from signal_engine import OneHourScheduler, SignalMonitor
from signal_engine.__main__ import main
from signal_engine.config import ConfigService
from signal_engine.runtime import (
    RUNTIME_KEY_LAST_ERROR,
    RUNTIME_KEY_LAST_ERROR_AT,
    RUNTIME_KEY_MONITOR_LAST_POLL,
    RUNTIME_KEY_PID,
    RUNTIME_KEY_SCHEDULER_LAST_TICK,
    RUNTIME_KEY_STATE,
    RUNTIME_STATE_RUNNING,
    RUNTIME_STATE_STOPPED,
    Runtime,
    RuntimeConfig,
    health_from_snapshot,
    render_active,
    render_demo,
    render_status,
    runtime_config_from_env,
)
from signal_engine.scheduler import DEFAULT_WINDOW_CANDLES
from tests.signal_engine_test_helpers import make_analysis, make_candles

ENV_PREFIX = "BTCUSDT_"


def _strip_env() -> None:
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            os.environ.pop(key, None)


class ScriptedMarketData:
    """Deterministic fake for BOTH the scheduler and the monitor fetchers."""

    def __init__(self, candles_1h=None, monitor_script=None) -> None:
        self.candles_1h = list(candles_1h or [])
        self.monitor_script = list(monitor_script or [])
        self.monitor_index = 0
        self.closed_calls = 0
        self.klines_calls = 0

    def fetch_closed_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
        self.closed_calls += 1
        return [c for c in self.candles_1h if c.is_closed]

    def fetch_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
        self.klines_calls += 1
        if not self.monitor_script:
            return []
        idx = min(self.monitor_index, len(self.monitor_script) - 1)
        self.monitor_index += 1
        entry = self.monitor_script[idx]
        if isinstance(entry, Candle):
            return [entry]
        return entry


class TracingNotifier:
    """Records every method call so tests can assert ordering + dedup."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def _record(self, method: str, **kwargs) -> object:
        self.events.append((method, kwargs))
        return None

    def notify_signal_created(self, signal, *, candle_ts=None):
        return self._record("created", signal=signal, candle_ts=candle_ts)

    def notify_signal_opened(self, signal):
        return self._record("opened", signal=signal)

    def notify_signal_tp(self, signal, *, pnl=None, balance=None):
        return self._record("tp", signal=signal, pnl=pnl, balance=balance)

    def notify_signal_sl(self, signal, *, pnl=None, balance=None):
        return self._record("sl", signal=signal, pnl=pnl, balance=balance)

    def notify_ambiguous(self, signal, reason="", *, candle_ts=None):
        return self._record("ambiguous", signal=signal, reason=reason, candle_ts=candle_ts)


def _make_analyzer(decision="WAIT"):
    calls = []

    def analyzer(candles, indicators):
        calls.append(1)
        return make_analysis(decision)

    return analyzer, calls


def _entry_candle(high=61500.0, low=60200.0) -> Candle:
    mid = (high + low) / 2.0
    return Candle(
        timestamp=1_000_000,
        open=mid,
        high=high,
        low=low,
        close=mid,
        volume=1.0,
        close_time=1_059_999,
        is_closed=True,
    )


class TempDb:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "runtime.db")
        self.db = Database(self.path)
        self.db.initialize()

    def close(self) -> None:
        self._tmp.cleanup()


RUN_CONFIG = RuntimeConfig(scheduler_poll=60.0, monitor_poll=30.0)


@pytest.mark.unit
class TestRuntimeConfig(unittest.TestCase):
    def tearDown(self) -> None:
        _strip_env()

    def test_defaults_match_phase14_spec(self):
        config = RuntimeConfig()
        self.assertEqual(config.symbol, "BTCUSDT")
        self.assertEqual(config.timeframe, "1h")
        self.assertEqual(config.scheduler_poll, 30.0)
        self.assertEqual(config.monitor_poll, 15.0)

    def test_env_config(self):
        os.environ["BTCUSDT_SYMBOL"] = "btcusdt"
        os.environ["BTCUSDT_TIMEFRAME"] = "15m"
        os.environ["BTCUSDT_RUNTIME_SCHEDULER_POLL"] = "5"
        os.environ["BTCUSDT_RUNTIME_MONITOR_POLL"] = "2.5"
        os.environ["BTCUSDT_DEMO_INITIAL_BALANCE"] = "100"
        os.environ["BTCUSDT_DEMO_MARGIN_PER_TRADE"] = "7"
        os.environ["BTCUSDT_DEMO_LEVERAGE"] = "5"
        os.environ["BTCUSDT_DEMO_RISK_PERCENT"] = "2"
        os.environ["BTCUSDT_DEMO_FEE_RATE"] = "0.001"
        config = runtime_config_from_env()
        self.assertEqual(config.symbol, "BTCUSDT")
        self.assertEqual(config.timeframe, "15m")
        self.assertEqual(config.scheduler_poll, 5.0)
        self.assertEqual(config.monitor_poll, 2.5)
        self.assertEqual(config.demo.initial_balance, Decimal("100"))
        self.assertEqual(config.demo.margin_per_trade, Decimal("7"))
        self.assertEqual(config.demo.leverage, 5)
        self.assertEqual(config.demo.risk_percent, Decimal("2"))
        self.assertEqual(config.demo.fee_rate, Decimal("0.001"))


@pytest.mark.unit
class TestHealth(unittest.TestCase):
    def test_fresh_heartbeat_reports_running(self):
        now = datetime.now(timezone.utc)
        snapshot = {
            RUNTIME_KEY_STATE: RUNTIME_STATE_RUNNING,
            RUNTIME_KEY_PID: "1234",
            RUNTIME_KEY_SCHEDULER_LAST_TICK: (now - timedelta(seconds=5)).isoformat(),
            RUNTIME_KEY_MONITOR_LAST_POLL: (now - timedelta(seconds=5)).isoformat(),
        }
        health = health_from_snapshot(snapshot, now=now, scheduler_poll=30, monitor_poll=15)
        self.assertEqual(health["scheduler"], "RUNNING")
        self.assertEqual(health["monitor"], "RUNNING")
        self.assertEqual(health["state"], RUNTIME_STATE_RUNNING)
        self.assertEqual(health["pid"], "1234")

    def test_stale_heartbeat_reports_stopped(self):
        now = datetime.now(timezone.utc)
        snapshot = {
            RUNTIME_KEY_SCHEDULER_LAST_TICK: (now - timedelta(seconds=120)).isoformat(),
            RUNTIME_KEY_MONITOR_LAST_POLL: (now - timedelta(seconds=120)).isoformat(),
        }
        health = health_from_snapshot(snapshot, now=now, scheduler_poll=30, monitor_poll=15)
        self.assertEqual(health["scheduler"], "STOPPED")
        self.assertEqual(health["monitor"], "STOPPED")

    def test_no_heartbeat_reports_stopped(self):
        now = datetime.now(timezone.utc)
        health = health_from_snapshot({}, now=now, scheduler_poll=30, monitor_poll=15)
        self.assertEqual(health["scheduler"], "STOPPED")
        self.assertEqual(health["monitor"], "STOPPED")
        self.assertIsNone(health["last_error"])
        self.assertIsNone(health["last_error_at"])


def _runtime(tmp: TempDb, *, md, notifier, analyzer_decision="LONG", config=None):
    config = config if config is not None else RUN_CONFIG
    repo = SignalRepository(tmp.db)
    analyzer, calls = _make_analyzer(analyzer_decision)
    scheduler = OneHourScheduler(
        repo,
        market_data=md,
        analyzer=analyzer,
        notifier=notifier,
    )
    monitor = SignalMonitor(repo, market_data=md)
    runtime = Runtime(
        config,
        database=tmp.db,
        scheduler=scheduler,
        monitor=monitor,
        notifier=notifier,
    )
    return runtime, calls, notifier


@pytest.mark.unit
class TestRuntimeDaemon(unittest.TestCase):
    def setUp(self):
        self._tmp = TempDb()
        self.addCleanup(self._tmp.close)

    def test_single_lifecycle_long_then_tp(self):
        candles = make_candles(DEFAULT_WINDOW_CANDLES)
        md = ScriptedMarketData(candles_1h=candles, monitor_script=[_entry_candle(61500, 60200), _entry_candle(65000, 60500)])
        notifier = TracingNotifier()
        runtime, calls, notifier = _runtime(self._tmp, md=md, notifier=notifier)

        first = runtime.poll()
        self.assertEqual(first["scheduler"], "CREATED")
        self.assertEqual(first["monitor"], "ENTRY_HIT")
        # One AI analysis, one "created", one "opened" (exactly once each).
        self.assertEqual(len(calls), 1)
        self.assertEqual([m for m, _ in notifier.events], ["created", "opened"])
        active = SignalRepository(self._tmp.db).get_active_signal()
        self.assertEqual(active.status, STATUS_OPEN)
        positions = DemoRepository(self._tmp.db).list_positions(status="OPEN")
        self.assertEqual(len(positions), 1)

        second = runtime.poll()
        # The marker for this candle was already processed; the AI never runs
        # again for the same candle while a position is open.
        self.assertEqual(second["scheduler"], "ALREADY_PROCESSED")
        self.assertEqual(second["monitor"], "TP_HIT")
        self.assertEqual(len(calls), 1)
        methods = [m for m, _ in notifier.events]
        self.assertEqual(methods, ["created", "opened", "tp"])
        tp_event = [kwargs for m, kwargs in notifier.events if m == "tp"][0]
        self.assertIsNotNone(tp_event["pnl"])
        self.assertGreater(tp_event["pnl"], 0)
        self.assertEqual(tp_event["balance"], runtime.executor.account().balance)

        # Demo ledger is consistent after the TP.
        executor = runtime.executor
        stats = executor.statistics()
        self.assertEqual(stats.total_trades, 1)
        self.assertEqual(stats.wins, 1)
        trades = DemoRepository(self._tmp.db).list_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(stats.final_balance, executor.account().balance)

    def test_single_lifecycle_short_then_sl(self):
        candles = make_candles(DEFAULT_WINDOW_CANDLES)
        md = ScriptedMarketData(candles_1h=candles, monitor_script=[])
        notifier = TracingNotifier()
        # Use a sad 1h window -> the only monitor poll opens + SL-closes.
        def short_analysis(candles_list, indicators):
            return make_analysis(
                "SHORT",
                entry_price=59000.0,
                stop_loss=60000.0,
                take_profit=57700.0,  # RR 1.3 >= Phase B minimum 1.2
            )

        class ShortAnalyzer:
            calls = 0

            def __call__(self, candles_list, indicators):
                ShortAnalyzer.calls += 1
                return short_analysis(candles_list, indicators)

        # First monitor poll gives the entry touch for SHORT (low <= 59000),
        # second poll gives the SL touch (high >= 60000).
        md.monitor_script = [
            _entry_candle(58800, 58500),
            _entry_candle(60100, 59500),
        ]
        repo = SignalRepository(self._tmp.db)
        scheduler = OneHourScheduler(repo, market_data=md, analyzer=ShortAnalyzer(), notifier=notifier)
        monitor = SignalMonitor(repo, market_data=md)
        runtime = Runtime(RUN_CONFIG, database=self._tmp.db, scheduler=scheduler, monitor=monitor, notifier=notifier)

        first = runtime.poll()
        self.assertEqual(first["monitor"], "ENTRY_HIT")
        second = runtime.poll()
        self.assertEqual(second["monitor"], "SL_HIT")
        methods = [m for m, _ in notifier.events]
        self.assertEqual(methods, ["created", "opened", "sl"])
        sl_event = [kwargs for m, kwargs in notifier.events if m == "sl"][0]
        self.assertLess(sl_event["pnl"], 0)
        self.assertEqual(sl_event["balance"], runtime.executor.account().balance)
        created_event = [kwargs for m, kwargs in notifier.events if m == "created"][0]
        signal_id = created_event["signal"].id
        trade = DemoRepository(self._tmp.db).get_trade_for_signal(signal_id)
        self.assertIsNotNone(trade)
        self.assertLess(Decimal(trade["net_pnl"]), 0)

    def test_ai_lock_prevents_analysis_while_pending(self):
        candles = make_candles(DEFAULT_WINDOW_CANDLES)
        md = ScriptedMarketData(candles_1h=candles, monitor_script=[_entry_candle(61200, 60500)])
        notifier = TracingNotifier()
        runtime, calls, _ = _runtime(self._tmp, md=md, notifier=notifier)
        runtime.poll()  # creates PENDING_ENTRY ONLY (monitor 1m never confirms entry)
        md.monitor_script = []
        runtime.poll()
        self.assertEqual(len(calls), 1)  # analyzer not called again

    def test_run_once_summary_and_stop_state(self):
        candles = make_candles(DEFAULT_WINDOW_CANDLES)
        md = ScriptedMarketData(candles_1h=candles, monitor_script=[])
        notifier = TracingNotifier()
        runtime, _, _ = _runtime(self._tmp, md=md, notifier=notifier)
        summary = runtime.run_once()
        self.assertEqual(summary["recovery"], "NO_OPEN_SIGNAL")
        self.assertIn("scheduler", summary)
        self.assertIn("monitor", summary)
        self.assertIn("reconcile_positions", summary)
        store = RuntimeStateRepository(self._tmp.db)
        self.assertEqual(store.get(RUNTIME_KEY_STATE), RUNTIME_STATE_STOPPED)
        self.assertIn(RUNTIME_KEY_PID, store.snapshot())

    def test_restart_recovery_never_notifies_and_recovers_position(self):
        # Build an OPEN signal + demo position directly.
        signal = SignalRepository(self._tmp.db)
        engine = None
        from signal_engine import SignalEngine, SignalState

        engine = SignalEngine(signal, SignalState(signal))
        created = engine.process(make_analysis("LONG")).signal
        opened = signal.transition_signal(created.id, STATUS_OPEN)
        from demo.account import DemoConfig
        from demo.executor import DemoExecutor

        executor = DemoExecutor(self._tmp.db, config=DemoConfig(margin_per_trade=20, leverage=5))
        executor.open_position(opened)

        # Restart: a fresh runtime must recover + reconcile silently.
        notifier = TracingNotifier()
        md = ScriptedMarketData(candles_1h=[], monitor_script=[])
        analyzer, calls = _make_analyzer("WAIT")
        repo = SignalRepository(self._tmp.db)
        scheduler = OneHourScheduler(repo, market_data=md, analyzer=analyzer, notifier=notifier)
        runtime = Runtime(
            RUN_CONFIG,
            database=self._tmp.db,
            scheduler=scheduler,
            monitor=SignalMonitor(repo, market_data=md),
            notifier=notifier,
        )
        recovery = runtime.start()
        self.assertEqual(recovery.outcome.value, "RECOVERED_OPEN")
        self.assertEqual(notifier.events, [])  # recovery never notifies
        self.assertEqual(len(calls), 0)  # AI must not run
        positions = DemoRepository(self._tmp.db).list_positions(status="OPEN")
        self.assertEqual(len(positions), 1)  # position preserved, not duplicated

        store = RuntimeStateRepository(self._tmp.db)
        self.assertEqual(store.get(RUNTIME_KEY_STATE), RUNTIME_STATE_RUNNING)

    def test_run_forever_graceful_stop(self):
        md = ScriptedMarketData(candles_1h=[], monitor_script=[])
        notifier = TracingNotifier()
        config = RuntimeConfig(scheduler_poll=0.02, monitor_poll=0.02)
        repo = SignalRepository(self._tmp.db)
        analyzer, _ = _make_analyzer("WAIT")
        scheduler = OneHourScheduler(repo, market_data=md, analyzer=analyzer, notifier=notifier)
        runtime = Runtime(
            config,
            database=self._tmp.db,
            scheduler=scheduler,
            monitor=SignalMonitor(repo, market_data=md),
            notifier=notifier,
        )
        thread = threading.Thread(target=runtime.run_forever, daemon=True)
        thread.start()
        runtime.stop()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        store = RuntimeStateRepository(self._tmp.db)
        self.assertEqual(store.get(RUNTIME_KEY_STATE), RUNTIME_STATE_STOPPED)

    def test_default_notifier_is_refreshable_and_uses_config_service(self):
        from signal_engine.telegram import RefreshableTelegramNotifier

        secrets_path = os.path.join(self._tmp._tmp.name, "secrets.json")
        config_service = ConfigService(self._tmp.db, secrets_path=secrets_path, env={})
        built = TracingNotifier()
        config_service.resolve_telegram_notifier = lambda: built
        repo = SignalRepository(self._tmp.db)
        runtime = Runtime(
            RUN_CONFIG,
            database=self._tmp.db,
            scheduler=OneHourScheduler(
                repo,
                market_data=ScriptedMarketData(),
                analyzer=_make_analyzer("WAIT")[0],
                notifier=None,
            ),
            monitor=SignalMonitor(repo, market_data=ScriptedMarketData()),
            config_service=config_service,
        )
        self.assertIsInstance(runtime.notifier, RefreshableTelegramNotifier)
        signal = repo.create_signal(
            "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
        )
        runtime._notify("notify_signal_opened", signal)
        self.assertEqual([m for m, _ in built.events], ["opened"])


@pytest.mark.unit
class TestRuntimeCandleLog(unittest.TestCase):
    """Phase 16: the Runtime's default scheduler carries the candle log, so the
    production daemon records one diagnostic row per analysed candle."""

    def setUp(self):
        self._tmp = TempDb()
        self.addCleanup(self._tmp.close)

    def _config_service(self):
        return ConfigService(
            self._tmp.db,
            secrets_path=os.path.join(self._tmp._tmp.name, "secrets.json"),
            env={},
        )

    def _runtime(self, md):
        return Runtime(
            RUN_CONFIG,
            database=self._tmp.db,
            market_data=md,
            config_service=self._config_service(),
        )

    def test_default_scheduler_is_wired_with_candle_log(self):
        md = ScriptedMarketData(
            candles_1h=make_candles(DEFAULT_WINDOW_CANDLES), monitor_script=[]
        )
        runtime = self._runtime(md)
        self.assertIsInstance(runtime.scheduler.candle_log, CandleLogRepository)

    def test_poll_records_wait_row_in_candle_log(self):
        md = ScriptedMarketData(
            candles_1h=make_candles(DEFAULT_WINDOW_CANDLES), monitor_script=[]
        )
        runtime = self._runtime(md)
        with patch("signal_engine.runtime.analyze_signal", return_value=make_analysis("WAIT")):
            result = runtime.poll()
        self.assertEqual(result["scheduler"], "WAIT")
        rows = CandleLogRepository(self._tmp.db).list(limit=100)
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertEqual(entry.outcome, "WAIT")
        self.assertEqual(entry.decision, "WAIT")
        self.assertEqual(entry.symbol, "BTCUSDT")
        self.assertIsNotNone(entry.recorded_at)


@pytest.mark.unit
class TestLastErrorClear(unittest.TestCase):
    """Stale ``runtime.last_error`` is cleared once the system recovers."""

    def setUp(self):
        self._tmp = TempDb()
        self.addCleanup(self._tmp.close)

    def _runtime_with(self, analyzer, md=None):
        md = md or ScriptedMarketData(
            candles_1h=make_candles(DEFAULT_WINDOW_CANDLES), monitor_script=[]
        )
        notifier = TracingNotifier()
        repo = SignalRepository(self._tmp.db)
        scheduler = OneHourScheduler(
            repo, market_data=md, analyzer=analyzer, notifier=notifier
        )
        runtime = Runtime(
            RUN_CONFIG,
            database=self._tmp.db,
            scheduler=scheduler,
            monitor=SignalMonitor(repo, market_data=md),
            notifier=notifier,
        )
        return runtime, repo

    def test_successful_tick_clears_stale_error(self):
        analyzer, _ = _make_analyzer("WAIT")
        runtime, _ = self._runtime_with(analyzer)
        runtime._record_error("stale gemma 429 boom")
        store = RuntimeStateRepository(self._tmp.db)
        self.assertNotEqual(store.get(RUNTIME_KEY_LAST_ERROR), "")
        self.assertNotEqual(store.get(RUNTIME_KEY_LAST_ERROR_AT), "")
        summary = runtime.poll()
        self.assertNotEqual(summary["scheduler"], "ERROR")
        self.assertEqual(store.get(RUNTIME_KEY_LAST_ERROR), "")
        self.assertEqual(store.get(RUNTIME_KEY_LAST_ERROR_AT), "")

    def test_entry_hit_clears_stale_error_on_open(self):
        analyzer, _ = _make_analyzer("LONG")
        md = ScriptedMarketData(
            candles_1h=make_candles(DEFAULT_WINDOW_CANDLES),
            monitor_script=[_entry_candle(61500, 60200)],
        )
        runtime, repo = self._runtime_with(analyzer, md=md)
        runtime._record_error("old error before open")
        summary = runtime.poll()
        self.assertEqual(summary["monitor"], "ENTRY_HIT")
        active = repo.get_active_signal()
        self.assertEqual(active.status, STATUS_OPEN)
        store = RuntimeStateRepository(self._tmp.db)
        self.assertEqual(store.get(RUNTIME_KEY_LAST_ERROR), "")

    def test_error_tick_records_then_success_clears(self):
        inner_analyzer, _ = _make_analyzer("WAIT")

        class OnceBoom:
            def __init__(self, inner):
                self.inner = inner
                self.calls = 0

            def __call__(self, candles, indicators):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("kaboom")
                return self.inner(candles, indicators)

        runtime, _ = self._runtime_with(OnceBoom(inner_analyzer))
        store = RuntimeStateRepository(self._tmp.db)
        failed = runtime.poll()
        self.assertEqual(failed["scheduler"], "ERROR")
        self.assertIn("kaboom", store.get(RUNTIME_KEY_LAST_ERROR))
        self.assertNotEqual(store.get(RUNTIME_KEY_LAST_ERROR_AT), "")
        ok = runtime.poll()
        self.assertNotEqual(ok["scheduler"], "ERROR")
        self.assertEqual(store.get(RUNTIME_KEY_LAST_ERROR), "")
        self.assertEqual(store.get(RUNTIME_KEY_LAST_ERROR_AT), "")


@pytest.mark.unit
class TestRuntimeRecoveryReconcile(unittest.TestCase):
    def setUp(self):
        self._tmp = TempDb()
        self.addCleanup(self._tmp.close)

    def test_start_reconciles_closed_signal_missing_trade(self):
        from signal_engine import SignalEngine, SignalState

        repo = SignalRepository(self._tmp.db)
        engine = SignalEngine(repo, SignalState(repo))
        opened = repo.transition_signal(engine.process(make_analysis("LONG")).signal.id, STATUS_OPEN)
        from demo.account import DemoConfig
        from demo.executor import DemoExecutor

        executor = DemoExecutor(self._tmp.db, config=DemoConfig(margin_per_trade=20, leverage=5))
        position = executor.open_position(opened)
        self.assertIsNotNone(position)
        repo.transition_signal(
            opened.id,
            STATUS_TP_HIT,
            close_price=opened.take_profit,
            close_reason="TP",
        )

        # A fresh runtime's start() repairs the trade ledger.
        notifier = TracingNotifier()
        md = ScriptedMarketData(candles_1h=[], monitor_script=[])
        runtime = Runtime(
            RUN_CONFIG,
            database=self._tmp.db,
            scheduler=OneHourScheduler(repo, market_data=md, analyzer=lambda c, i: make_analysis("WAIT"), notifier=notifier),
            monitor=SignalMonitor(repo, market_data=md),
            notifier=notifier,
        )
        recovery = runtime.start()
        self.assertEqual(recovery.outcome.value, "NO_OPEN_SIGNAL")
        trades = DemoRepository(self._tmp.db).list_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(notifier.events, [])


@pytest.mark.unit
class TestRenderers(unittest.TestCase):
    def test_render_active_none(self):
        self.assertIn("No active signal", render_active(None))

    def test_render_active_signal(self):
        signal = _fake_signal()
        text = render_active(signal)
        self.assertIn("LONG", text)
        self.assertIn("PENDING_ENTRY", text)
        self.assertIn("61,000.00", text)

    def test_render_status(self):
        health = {
            "state": RUNTIME_STATE_RUNNING,
            "pid": "1",
            "started_at": "2026-09-10T00:00:00.000Z",
            "scheduler": "RUNNING",
            "scheduler_last_tick": "2026-09-10T00:00:00.000Z",
            "monitor": "STOPPED",
            "monitor_last_poll": None,
            "last_error": None,
        }
        text = render_status(health, active=None, db_path="/tmp/x.db")
        self.assertIn("Scheduler: RUNNING", text)
        self.assertIn("Monitor:   STOPPED", text)
        self.assertIn("Active signal: none", text)

    def test_render_demo_empty(self):
        from demo.position import DemoAccountRecord

        account = DemoAccountRecord(
            id=1, name="demo", initial_balance=Decimal("20"), balance=Decimal("20"),
            equity=Decimal("20"), margin_per_trade=Decimal("2"), leverage=10,
            risk_percent=Decimal("1"), fee_rate=Decimal("0.0004"),
            created_at="2026-09-10T00:00:00.000Z", updated_at=None, peak_equity=Decimal("20"),
        )
        text = render_demo(account, [], None)
        self.assertIn("Balance: 20.00 USDT", text)
        self.assertIn("Equity:  20.00 USDT", text)
        self.assertIn("Peak equity: 20.00 USDT", text)
        self.assertNotIn("No active position", text)


def _fake_signal():
    from database.models import Signal

    return Signal(
        id=42,
        symbol="BTCUSDT",
        timeframe="1h",
        direction="LONG",
        status=STATUS_PENDING_ENTRY,
        entry=Decimal("61000"),
        stop_loss=Decimal("60000"),
        take_profit=Decimal("64000"),
        created_at="2026-09-10T00:00:00.000Z",
    )


@pytest.mark.unit
class TestCliQueries(unittest.TestCase):
    def setUp(self):
        _strip_env()
        self._tmp = TempDb()
        self.addCleanup(self._tmp.close)

    def _capture(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(argv)
        return code, stream.getvalue()

    def test_demo_no_account(self):
        code, out = self._capture(["demo", "--db", self._tmp.path])
        self.assertEqual(code, 0)
        self.assertIn("No demo account yet", out)

    def test_active_none(self):
        code, out = self._capture(["active", "--db", self._tmp.path])
        self.assertEqual(code, 0)
        self.assertIn("No active signal", out)

    def test_signals_empty(self):
        code, out = self._capture(["signals", "--db", self._tmp.path])
        self.assertEqual(code, 0)
        self.assertIn("No signals recorded", out)

    def test_status_reports_runtime_state(self):
        code, out = self._capture(["status", "--db", self._tmp.path])
        self.assertEqual(code, 0)
        self.assertIn("Runtime state:", out)
        self.assertIn("Scheduler:", out)

    def test_health_exit_codes(self):
        code, _ = self._capture(["health", "--db", self._tmp.path])
        self.assertEqual(code, 1)  # daemon not running

        # Write fresh heartbeats -> healthy daemon.
        store = RuntimeStateRepository(self._tmp.db)
        store.set(RUNTIME_KEY_STATE, RUNTIME_STATE_RUNNING)
        store.set(RUNTIME_KEY_SCHEDULER_LAST_TICK, datetime.now(timezone.utc).isoformat())
        store.set(RUNTIME_KEY_MONITOR_LAST_POLL, datetime.now(timezone.utc).isoformat())
        code, out = self._capture(["health", "--db", self._tmp.path])
        self.assertEqual(code, 0)
        self.assertIn("Scheduler: RUNNING", out)

    def test_health_json(self):
        store = RuntimeStateRepository(self._tmp.db)
        store.set(RUNTIME_KEY_SCHEDULER_LAST_TICK, datetime.now(timezone.utc).isoformat())
        store.set(RUNTIME_KEY_MONITOR_LAST_POLL, datetime.now(timezone.utc).isoformat())
        code, out = self._capture(["health", "--db", self._tmp.path, "--json"])
        self.assertEqual(code, 0)
        self.assertIn('"running": true', out)

    def test_once_patches_runtime(self):
        import signal_engine.__main__ as cli

        summary = {"recovery": "NO_OPEN_SIGNAL", "scheduler": "CREATED", "monitor": "NO_ACTIVE_SIGNAL"}

        class FakeRuntime:
            def __init__(self, *a, **k):
                pass

            def run_once(self):
                return summary

        original = cli.Runtime
        cli.Runtime = FakeRuntime
        try:
            code, out = self._capture(["--once", "--db", self._tmp.path])
        finally:
            cli.Runtime = original
        self.assertEqual(code, 0)
        for key in summary:
            self.assertIn(f"{key}: {summary[key]}", out)


@pytest.mark.unit
class TestRuntimeResolvingSymbol(unittest.TestCase):
    """Runtime symbol source: DB setting > env > default (Phase 16)."""

    def _runtime_with_service(self, env: dict) -> tuple[Runtime, TempDb]:
        tmp = TempDb()
        service = ConfigService(
            tmp.db,
            secrets_path=os.path.join(tempfile.mkdtemp(), "secrets.json"),
            env=env,
        )
        runtime = Runtime(
            RUN_CONFIG,
            database=tmp.db,
            config_service=service,
            scheduler=None,
            monitor=SignalMonitor(SignalRepository(tmp.db), market_data=ScriptedMarketData()),
            notifier=TracingNotifier(),
        )
        return runtime, tmp

    def test_runtime_resolves_db_symbol_over_env(self):
        runtime, tmp = self._runtime_with_service({"BTCUSDT_SYMBOL": "ETHUSDT"})
        self.addCleanup(tmp.close)
        self.assertEqual(runtime._resolving_symbol(), "ETHUSDT")
        runtime.config_service.update_symbol({"symbol": "XAUUSDT"})
        self.assertEqual(runtime._resolving_symbol(), "XAUUSDT")

    def test_runtime_uses_default_when_env_not_supported(self):
        runtime, tmp = self._runtime_with_service({"BTCUSDT_SYMBOL": "SOLUSDT"})
        self.addCleanup(tmp.close)
        self.assertEqual(runtime._resolving_symbol(), "BTCUSDT")

    def test_runtime_symbol_change_requires_no_restart(self):
        runtime, tmp = self._runtime_with_service({})
        self.addCleanup(tmp.close)
        self.assertEqual(runtime._resolving_symbol(), "BTCUSDT")
        runtime.config_service.update_symbol({"symbol": "ETHUSDT"})
        self.assertEqual(runtime._resolving_symbol(), "ETHUSDT")


@pytest.mark.unit
class TestSymbolSwitch(unittest.TestCase):
    """Settings tick list is the single switch: unticked daemons skip analysis
    (zero LLM calls) but keep monitoring; every skip is visible in Daemon Log."""

    def setUp(self):
        self._tmp = TempDb()
        self.addCleanup(self._tmp.close)

    def _runtime_for(self, env_symbol, combo=None, analyzer_decision="LONG"):
        md = ScriptedMarketData(
            candles_1h=make_candles(DEFAULT_WINDOW_CANDLES), monitor_script=[]
        )
        analyzer, calls = _make_analyzer(analyzer_decision)
        notifier = TracingNotifier()
        svc = ConfigService(self._tmp.db, env={"BTCUSDT_SYMBOL": env_symbol})
        if combo is not None:
            svc.update_symbols({"symbols": combo})
        repo = SignalRepository(self._tmp.db)
        scheduler = OneHourScheduler(
            repo, market_data=md, analyzer=analyzer, notifier=notifier
        )
        monitor = SignalMonitor(repo, market_data=md)
        runtime = Runtime(
            RUN_CONFIG,
            database=self._tmp.db,
            scheduler=scheduler,
            monitor=monitor,
            notifier=notifier,
            config_service=svc,
        )
        return runtime, calls, md

    def test_disabled_symbol_skips_without_llm(self):
        runtime, calls, md = self._runtime_for("ETHUSDT", ["BTCUSDT", "XAUUSDT"])
        self.assertEqual(runtime.tick_scheduler(), "SYMBOL_DISABLED")
        self.assertEqual(calls, [])
        self.assertEqual(md.klines_calls, 0)
        heartbeat = runtime.state_store.get(RUNTIME_KEY_SCHEDULER_LAST_TICK)
        self.assertIsNotNone(heartbeat)

    def test_disabled_writes_one_log_row_per_candle(self):
        runtime, calls, _ = self._runtime_for("ETHUSDT", ["BTCUSDT"])
        runtime.tick_scheduler()
        runtime.tick_scheduler()
        rows = CandleLogRepository(self._tmp.db).list(limit=100)
        disabled = [r for r in rows if r.outcome == "SYMBOL_DISABLED"]
        self.assertEqual(len(disabled), 1)
        self.assertEqual(disabled[0].symbol, "ETHUSDT")
        self.assertEqual(disabled[0].decision, "NONE")
        self.assertEqual(disabled[0].llm_calls, 0)

    def test_enabled_symbol_runs_normally(self):
        runtime, calls, _ = self._runtime_for(
            "XAUUSDT", ["BTCUSDT", "XAUUSDT"]
        )
        self.assertEqual(runtime.tick_scheduler(), "CREATED")
        self.assertEqual(len(calls), 1)

    def test_legacy_single_symbol_runs_without_combo(self):
        runtime, calls, _ = self._runtime_for("BTCUSDT", None)
        self.assertNotEqual(runtime.tick_scheduler(), "SYMBOL_DISABLED")
        self.assertEqual(len(calls), 1)

    def test_disabled_symbol_monitor_still_closes_open_position(self):
        from demo.executor import DemoExecutor

        runtime, calls, md = self._runtime_for("ETHUSDT", ["BTCUSDT"])
        repo = SignalRepository(self._tmp.db)
        sig = repo.create_signal(
            "ETHUSDT",
            "1h",
            "LONG",
            Decimal("60000"),
            Decimal("59000"),
            Decimal("61000"),
        )
        opened = repo.transition_signal(sig.id, STATUS_OPEN)
        self.assertIsNotNone(DemoExecutor(self._tmp.db).open_position(opened))
        md.monitor_script = [_entry_candle(high=61500.0, low=60500.0)]
        self.assertEqual(runtime.tick_scheduler(), "SYMBOL_DISABLED")
        self.assertEqual(runtime.poll_monitor(), "TP_HIT")
        self.assertEqual(repo.get_signal(sig.id).status, STATUS_TP_HIT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
