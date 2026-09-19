"""Phase P1 tests: futures positioning (funding/OI/long-short).

Covers the public-feed fetch/parse/fail-soft layer, the prompt note, the
crowded-side funding guardrail, scheduler threading (fetch only when an
analysis will run), the backtest history wiring, and the web endpoint + UI
markers. No test touches the real network.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backtest.benchmark.runner import build_positioning_wiring
from binance.market_data import Candle
from binance.positioning import (
    PositioningHistory,
    PositioningSnapshot,
    fetch_positioning,
    render_note,
)
from database.database import CandleLogRepository, Database
from signal_engine import OneHourScheduler, SchedulerOutcome, SignalAnalyzer
from signal_engine.analysis import SignalDecisionModel
from signal_engine.context import build_analysis_context
from signal_engine.llm import LLMConfig
from signal_engine.validator import (
    GuardrailConfig,
    SignalValidationError,
    default_guardrails,
    validate_analysis,
    validate_funding_crowd,
)
from tests.signal_engine_test_helpers import (
    INTERVAL_MS,
    TempSignalDb,
    indicators_for,
    make_analysis,
    make_candles,
)
from web.server import WebApplication, WebServer

CONFIG = LLMConfig(provider="ollama", model="qwen3:8b")


class FakeBinanceClient:
    """Scripted public-GET responses (records calls, can raise)."""

    def __init__(self, routes: dict, error=None) -> None:
        self.routes = routes
        self.error = error
        self.calls: list = []

    def get(self, path: str, params=None):
        self.calls.append((path, params))
        if self.error is not None:
            raise self.error
        return self.routes[path]


def live_client() -> FakeBinanceClient:
    return FakeBinanceClient(
        {
            "/fapi/v1/fundingRate": [
                {"fundingRate": "0.00042", "fundingTime": 1720000000000}
            ],
            "/fapi/v1/openInterest": {"openInterest": "12345.6"},
            "/futures/data/globalLongShortAccountRatio": [
                {"longShortRatio": "1.4", "timestamp": 1720000000000}
            ],
        }
    )


# ── fetch / parse / fail-soft ────────────────────────────────────────────


@pytest.mark.unit
class TestFetchPositioning(unittest.TestCase):
    def test_full_snapshot(self):
        snap = fetch_positioning(live_client(), "BTCUSDT")
        self.assertAlmostEqual(snap.funding_rate, 0.00042)
        self.assertEqual(snap.funding_time_ms, 1720000000000)
        self.assertAlmostEqual(snap.open_interest, 12345.6)
        self.assertAlmostEqual(snap.long_pct, 1.4 / 2.4 * 100.0)
        self.assertAlmostEqual(snap.short_pct, 100.0 - 1.4 / 2.4 * 100.0)

    def test_transport_failure_is_empty_snapshot(self):
        snap = fetch_positioning(
            FakeBinanceClient({}, error=ConnectionError("down")), "BTCUSDT"
        )
        self.assertEqual(snap, PositioningSnapshot())

    def test_partial_and_corrupt_rows(self):
        client = FakeBinanceClient(
            {
                "/fapi/v1/fundingRate": [{"fundingRate": "9.99", "fundingTime": 1}],
                "/fapi/v1/openInterest": {"openInterest": "-3"},
                "/futures/data/globalLongShortAccountRatio": [
                    {"longShortRatio": "0"}
                ],
            }
        )
        snap = fetch_positioning(client, "BTCUSDT")
        self.assertIsNone(snap.funding_rate)  # absurd 999% ignored
        self.assertIsNone(snap.open_interest)  # negative ignored
        self.assertIsNone(snap.long_pct)  # non-positive ratio ignored

    def test_non_list_shapes(self):
        client = FakeBinanceClient(
            {
                "/fapi/v1/fundingRate": {"fundingRate": "0.0001"},
                "/fapi/v1/openInterest": [{"openInterest": "5"}],
                "/futures/data/globalLongShortAccountRatio": {},
            }
        )
        snap = fetch_positioning(client, "BTCUSDT")
        self.assertEqual(snap, PositioningSnapshot())


# ── note rendering ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestRenderNote(unittest.TestCase):
    def test_full_note(self):
        note = render_note(0.00042, 58.3, 41.7, 12345.6)
        self.assertIn("funding=+0.0420%/8h", note)
        self.assertIn("longs 58.3% / shorts 41.7%", note)
        self.assertIn("open interest 12,345.6 BTC", note)

    def test_extreme_funding_adds_warning(self):
        note = render_note(0.0008, 60.0, 40.0, None)
        self.assertIn("extremely crowded long", note)
        note = render_note(-0.0008, 40.0, 60.0, None)
        self.assertIn("extremely crowded short", note)
        calm = render_note(0.0001, 51.0, 49.0, None)
        self.assertNotIn("crowded", calm)

    def test_empty_is_none(self):
        self.assertIsNone(render_note(None, None, None, None))
        self.assertIsNotNone(render_note(None, 55.0, 45.0, None))

    def test_context_section(self):
        candles = make_candles(220)
        indicators = indicators_for(make_candles(220))
        plain = build_analysis_context(candles, indicators)
        self.assertNotIn("Futures positioning", plain.rendered)
        noted = build_analysis_context(
            candles, indicators, positioning="Futures positioning: test"
        )
        self.assertIn("## Futures positioning", noted.rendered)
        self.assertIn("Futures positioning: test", noted.rendered)


# ── guardrail ────────────────────────────────────────────────────────────


def _with_funding(analysis, funding_rate):
    from dataclasses import replace

    return replace(analysis, funding_rate=funding_rate)


@pytest.mark.unit
class TestFundingGuardrail(unittest.TestCase):
    def test_rejects_crowded_side(self):
        guards = GuardrailConfig(max_funding_rate=Decimal("0.0005"))
        with self.assertRaises(SignalValidationError):
            validate_funding_crowd("LONG", 0.0008, guards)
        with self.assertRaises(SignalValidationError):
            validate_funding_crowd("SHORT", -0.0008, guards)
        # With-trend side is untouched.
        validate_funding_crowd("SHORT", 0.0008, guards)
        validate_funding_crowd("LONG", -0.0008, guards)

    def test_zero_cap_disables_and_none_skips(self):
        off = GuardrailConfig(max_funding_rate=Decimal("0"))
        validate_funding_crowd("LONG", 0.01, off)
        validate_funding_crowd("SHORT", -0.01, off)
        guards = GuardrailConfig(max_funding_rate=Decimal("0.0005"))
        validate_funding_crowd("LONG", None, guards)
        validate_funding_crowd("LONG", float("nan"), guards)

    def test_end_to_end_reject_and_pass(self):
        # make_analysis LONG: entry 61000 / SL 60000 / TP 64000, conf 80.
        crowded = _with_funding(make_analysis("LONG"), 0.0008)
        with self.assertRaises(SignalValidationError):
            validate_analysis(
                crowded, guardrails=GuardrailConfig(max_funding_rate=Decimal("0.0005"))
            )
        calm = _with_funding(make_analysis("LONG"), 0.0001)
        candidate = validate_analysis(
            calm, guardrails=GuardrailConfig(max_funding_rate=Decimal("0.0005"))
        )
        self.assertEqual(candidate.decision, "LONG")

    def test_env_knob(self):
        guards = default_guardrails(env={"BTCUSDT_MAX_FUNDING_RATE": "0.001"})
        self.assertEqual(guards.max_funding_rate, Decimal("0.001"))
        self.assertEqual(
            default_guardrails(env={}).max_funding_rate, Decimal("0.0005")
        )


# ── analyzer threading ───────────────────────────────────────────────────


def _structured_llm(model):
    llm = MagicMock()
    structured = MagicMock()
    llm.with_structured_output.return_value = structured
    structured.invoke.return_value = model
    return llm


@pytest.mark.unit
class TestAnalyzerThreading(unittest.TestCase):
    def test_single_threads_note_and_rate(self):
        candles = make_candles(240)
        indicators = indicators_for(make_candles(240))
        model = SignalDecisionModel(
            decision="WAIT", confidence=50.0, reasoning="mixed"
        )
        llm = _structured_llm(model)
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(
            candles,
            indicators,
            positioning="Futures positioning: test",
            funding_rate=0.0001,
        )
        self.assertEqual(result.funding_rate, 0.0001)
        prompt = llm.with_structured_output.return_value.invoke.call_args[0][0]
        bodies = [
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in (prompt if isinstance(prompt, list) else [prompt])
        ]
        self.assertIn("Futures positioning: test", "\n".join(bodies))

    def test_multi_accepts_kwargs(self):
        from signal_engine.multiagent import MultiAgentSignalAnalyzer

        analyzer = MultiAgentSignalAnalyzer.__new__(MultiAgentSignalAnalyzer)
        # Signature check only: multi path forwards like single (no LLM here).
        import inspect

        params = inspect.signature(analyzer.analyze).parameters
        self.assertIn("positioning", params)
        self.assertIn("funding_rate", params)


# ── scheduler ────────────────────────────────────────────────────────────


class FakeSchedulerData:
    def __init__(self, *candles, client=None) -> None:
        self.candles = list(candles)
        self.client = client

    def fetch_closed_klines(
        self, symbol, interval, limit, *, end_time_ms=None, now_ms=None
    ):
        return [
            c
            for c in self.candles
            if c.is_closed and (now_ms is None or c.close_time <= now_ms)
        ]


@pytest.mark.unit
class TestSchedulerPositioning(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
        # 222 hourly candles: the 12:00 tick sees the fresh 11:00 candle, the
        # 13:00 tick a fresh 12:00 candle (detection is newest-closed-first).
        self.candles = make_candles(222, start_ms=int(self.now.timestamp() * 1000) - 220 * INTERVAL_MS)

    def test_fetch_only_when_analysis_runs(self):
        client = live_client()
        md = FakeSchedulerData(*self.candles, client=client)
        calls: list = []

        def analyzer(candles, indicators):
            calls.append(True)
            return make_analysis("WAIT")

        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=analyzer,
            candle_log=CandleLogRepository(self.harness.db),
        )
        result = sched.tick(now=self.now)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(client.calls), 3)
        self.assertAlmostEqual(sched._tick_funding_rate, 0.00042)
        self.assertIn("funding=", sched._tick_positioning_note or "")
        # Blocked tick (active signal): no positioning fetch at all.
        client.calls.clear()
        repo = self.harness.repository
        repo.create_signal("BTCUSDT", "1h", "LONG", "100", "99", "101")
        blocked = sched.tick(
            now=datetime(2026, 9, 10, 13, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(blocked.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(client.calls, [])

    def test_default_analyzer_forwards_note_and_rate(self):
        client = live_client()
        md = FakeSchedulerData(*self.candles, client=client)
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            candle_log=CandleLogRepository(self.harness.db),
        )
        sched._tick_positioning_note = "Futures positioning: test"
        sched._tick_funding_rate = 0.0002
        seen: dict = {}

        def fake_analyze_signal(config, candles, indicators, **kwargs):
            seen.update(kwargs)
            return make_analysis("WAIT")

        with patch("signal_engine.scheduler.analyze_signal", fake_analyze_signal):
            sched._default_analyzer(self.candles, object())
        self.assertEqual(seen.get("positioning"), "Futures positioning: test")
        self.assertEqual(seen.get("funding_rate"), 0.0002)

    def test_market_data_without_client_skips(self):
        md = FakeSchedulerData(*self.candles)  # client=None
        calls: list = []

        def analyzer(candles, indicators):
            calls.append(True)
            return make_analysis("WAIT")

        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            analyzer=analyzer,
        )
        result = sched.tick(now=self.now)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(calls), 1)


# ── history + backtest wiring ────────────────────────────────────────────


def hour_candle_ts(ts: int) -> Candle:
    return Candle(
        timestamp=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0,
        close_time=ts + 3599_999, is_closed=True,
    )


def history_client() -> FakeBinanceClient:
    base = 1720000000000
    funding_rows = [
        {"fundingRate": "0.0001", "fundingTime": base + i * 8 * 3600_000}
        for i in range(3)
    ]
    ls_rows = [
        {"longShortRatio": "1.2", "timestamp": base + i * 3600_000}
        for i in range(5)
    ]
    client = FakeBinanceClient(
        {"/fapi/v1/fundingRate": funding_rows,
         "/futures/data/globalLongShortAccountRatio": ls_rows}
    )
    return client


@pytest.mark.unit
class TestPositioningHistory(unittest.TestCase):
    def test_fetch_freeze_roundtrip(self):
        history = PositioningHistory.fetch(
            history_client(), "BTCUSDT", 1720000000000, 1720100000000
        )
        frozen = history.frozen()
        self.assertTrue(len(frozen) >= 5)
        clone = PositioningHistory.from_frozen(json.loads(json.dumps(frozen)))
        self.assertEqual(clone.funding_at(1720000000000), 0.0001)
        self.assertAlmostEqual(clone.long_pct_at(1720000000000), 1.2 / 2.2 * 100.0)

    def test_value_at_latest_at_or_before(self):
        history = PositioningHistory(
            funding=[(100, 0.0001), (200, 0.0003)], long_short=[(150, 60.0)]
        )
        self.assertEqual(history.funding_at(150), 0.0001)
        self.assertEqual(history.funding_at(99), None)
        self.assertEqual(history.long_pct_at(200), 60.0)

    def test_fetch_failure_is_empty(self):
        history = PositioningHistory.fetch(
            FakeBinanceClient({}, error=ConnectionError("down")),
            "BTCUSDT",
            1,
            2,
        )
        self.assertEqual(history.frozen(), [])
        self.assertIsNone(history.note_at(10))

    def test_note_at(self):
        history = PositioningHistory(
            funding=[(100, 0.0008)], long_short=[(100, 62.0)]
        )
        note = history.note_at(200)
        self.assertIn("funding=+0.0800%/8h", note)
        self.assertIn("longs 62.0%", note)

    def test_build_positioning_wiring(self):
        history = PositioningHistory(
            funding=[(100, 0.0008)], long_short=[(100, 62.0)]
        )
        seen: dict = {}

        def base(candles, indicators, *, symbol, timeframe, max_candles, event_note=None, positioning=None, funding_rate=None, htf_note=None, regime=None):
            seen.update(
                event_note=event_note, positioning=positioning, funding_rate=funding_rate,
                htf_note=htf_note, regime=regime,
            )
            return make_analysis("WAIT")

        candles = [hour_candle_ts(50 + i * 3600_000) for i in range(3)]
        analyze_fn, frozen = build_positioning_wiring(
            history=history, period_ms=3600_000, analyze_base=base
        )
        analyze_fn(candles, None, symbol="BTCUSDT", timeframe="1h", max_candles=3, event_note="ev")
        self.assertEqual(seen["event_note"], "ev")
        self.assertIn("funding=", seen["positioning"])
        self.assertEqual(seen["funding_rate"], 0.0008)
        self.assertEqual(len(frozen), 1)


# ── web ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebPositioning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "pos.db"))
        self.db.initialize()
        market_data = SimpleNamespace(client=live_client())
        self.app = WebApplication(self.db, market_data=market_data)
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.bound_port}"

    def _get_json(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _get_raw(self, path: str) -> str:
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return resp.read().decode("utf-8")

    def test_positioning_endpoint(self):
        status, payload = self._get_json("/api/positioning")
        self.assertEqual(status, 200)
        self.assertTrue(payload["available"])
        self.assertAlmostEqual(payload["funding_rate"], 0.00042)
        self.assertAlmostEqual(payload["long_pct"], 1.4 / 2.4 * 100.0)
        self.assertAlmostEqual(payload["short_pct"], 100.0 - 1.4 / 2.4 * 100.0)
        self.assertAlmostEqual(payload["open_interest"], 12345.6)

    def test_endpoint_cached_60s(self):
        client = live_client()
        app = WebApplication(
            self.db, market_data=SimpleNamespace(client=client)
        )
        first = app.positioning()
        second = app.positioning()
        self.assertTrue(first["available"])
        self.assertEqual(first, second)
        self.assertEqual(len(client.calls), 3)  # one batch, then cache

    def test_positioning_unavailable_degrades(self):
        market_data = SimpleNamespace(client=FakeBinanceClient({}, error=ConnectionError("x")))
        app = WebApplication(self.db, market_data=market_data)
        payload = app.positioning()
        self.assertFalse(payload["available"])

    def test_static_markers(self):
        html = self._get_raw("/")
        self.assertIn('id="lsBar"', html)
        js = self._get_raw("/static/app.js")
        for marker in ("loadPositioning", "/api/positioning", "ls-track", "ls-long-t"):
            self.assertIn(marker, js)
        css = self._get_raw("/static/style.css")
        for marker in (".ls-bar", ".ls-track", ".ls-long", ".ls-short", ".ls-label"):
            self.assertIn(marker, css)


if __name__ == "__main__":
    unittest.main()
