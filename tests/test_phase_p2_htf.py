"""Phase P2 tests: 4H trend bias (soft version).

Covers regime classification (EMA stack), swing-structure detection,
rollup parity, the prompt note, the counter-regime veto guardrail, analyzer
threading, scheduler fetch discipline, and the backtest wiring + CLI flag.
"""

from __future__ import annotations

import inspect
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from backtest.benchmark.runner import build_htf_wiring
from binance.htf import (
    H4_MS,
    REGIME_DOWN,
    REGIME_NONE,
    REGIME_UP,
    classify_htf,
    detect_structure,
    render_note,
    rollup_1h_to_4h,
)
from binance.market_data import Candle
from database.database import CandleLogRepository
from signal_engine import OneHourScheduler, SchedulerOutcome, SignalAnalyzer
from signal_engine.analysis import SignalDecisionModel
from signal_engine.context import build_analysis_context
from signal_engine.llm import LLMConfig
from signal_engine.validator import (
    GuardrailConfig,
    SignalValidationError,
    default_guardrails,
    validate_analysis,
    validate_htf_bias,
)
from tests.signal_engine_test_helpers import (
    INTERVAL_MS,
    TempSignalDb,
    indicators_for,
    make_analysis,
    make_candles,
)

CONFIG = LLMConfig(provider="ollama", model="qwen3:8b")
HOUR_MS = 3_600_000


def h4_candles(highs: list[float], lows: list[float], *, step_ms: int = H4_MS) -> list[Candle]:
    out = []
    for i, (high, low) in enumerate(zip(highs, lows, strict=True)):
        out.append(
            Candle(
                timestamp=i * step_ms,
                open=low,
                high=high,
                low=low,
                close=(high + low) / 2.0,
                volume=10.0,
                close_time=i * step_ms + step_ms - 1,
                is_closed=True,
            )
        )
    return out


UP_HIGHS = [10, 12, 15, 13, 16, 14, 19, 15, 22]
UP_LOWS = [8, 9, 12, 10, 13, 11, 15, 12, 17]
DOWN_HIGHS = [22, 20, 17, 19, 16, 18, 13, 17, 10]
DOWN_LOWS = [17, 15, 12, 14, 11, 13, 8, 12, 5]


# ── structure + regime ───────────────────────────────────────────────────


@pytest.mark.unit
class TestHtfClassify(unittest.TestCase):
    def test_structure_up_down_flat(self):
        self.assertEqual(
            detect_structure(h4_candles(UP_HIGHS, UP_LOWS)), "HH_HL"
        )
        self.assertEqual(
            detect_structure(h4_candles(DOWN_HIGHS, DOWN_LOWS)), "LH_LL"
        )
        self.assertEqual(
            detect_structure(h4_candles([10] * 9, [9] * 9)), "NONE"
        )
        self.assertEqual(
            detect_structure(h4_candles(UP_HIGHS[:5], UP_LOWS[:5])), "NONE"
        )
        self.assertEqual(detect_structure([]), "NONE")
        self.assertEqual(detect_structure(None), "NONE")

    def test_regime_from_ema_stack(self):
        up = classify_htf(make_candles(60, trend=2.0))
        self.assertEqual(up.regime, REGIME_UP)
        self.assertIsNotNone(up.note)
        self.assertIn("UPTREND", up.note)
        down = classify_htf(make_candles(60, trend=-2.0))
        self.assertEqual(down.regime, REGIME_DOWN)
        self.assertIn("DOWNTREND", down.note)
        flat = classify_htf(make_candles(60, trend=0.0))
        self.assertEqual(flat.regime, REGIME_NONE)

    def test_short_input_is_none_without_raise(self):
        bias = classify_htf(make_candles(10))
        self.assertEqual(bias.regime, REGIME_NONE)
        self.assertIsNone(bias.note)
        self.assertEqual(classify_htf([]).regime, REGIME_NONE)
        self.assertEqual(classify_htf(None).regime, REGIME_NONE)

    def test_render_note_variants(self):
        self.assertIn("chop", render_note(REGIME_NONE, "NONE", 14.1))
        self.assertIn("UNCLEAR", render_note(REGIME_NONE, "HH_HL", 25.0))
        self.assertIn("ADX n/a", render_note(REGIME_UP, "NONE", None))


# ── rollup parity ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRollup(unittest.TestCase):
    def test_groups_of_four(self):
        hours = make_candles(8, start_ms=0)
        rolled = rollup_1h_to_4h(hours)
        self.assertEqual(len(rolled), 2)
        self.assertEqual(rolled[0].timestamp, 0)
        self.assertEqual(rolled[1].timestamp, H4_MS)
        self.assertAlmostEqual(rolled[0].open, hours[0].open)
        self.assertAlmostEqual(rolled[0].close, hours[3].close)
        self.assertAlmostEqual(
            rolled[0].high, max(h.high for h in hours[:4])
        )
        self.assertAlmostEqual(rolled[0].low, min(h.low for h in hours[:4]))
        self.assertTrue(all(c.is_closed for c in rolled))

    def test_incomplete_trailing_group_dropped(self):
        rolled = rollup_1h_to_4h(make_candles(10, start_ms=0))
        self.assertEqual(len(rolled), 2)

    def test_unclosed_excluded(self):
        hours = make_candles(4, start_ms=0, forming=True)
        rolled = rollup_1h_to_4h(hours)
        # 4 closed roll into one 4H candle; the lone forming candle's
        # incomplete bucket is dropped.
        self.assertEqual(len(rolled), 1)
        self.assertEqual(rolled[0].timestamp, 0)
        only_forming = [c for c in hours if not c.is_closed]
        self.assertEqual(len(only_forming), 1)
        self.assertEqual(rollup_1h_to_4h(only_forming), [])


# ── prompt note ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestHtfNote(unittest.TestCase):
    def test_section_rendered(self):
        candles = make_candles(220)
        indicators = indicators_for(make_candles(220))
        plain = build_analysis_context(candles, indicators)
        self.assertNotIn("4H bias", plain.rendered)
        noted = build_analysis_context(candles, indicators, htf_note="4H bias: test")
        self.assertIn("## 4H bias", noted.rendered)
        self.assertIn("4H bias: test", noted.rendered)


# ── guardrail ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestHtfGuardrail(unittest.TestCase):
    def test_vetoes_counter_regime(self):
        guards = GuardrailConfig(htf_bias=1)
        with self.assertRaises(SignalValidationError):
            validate_htf_bias("LONG", "DOWN", guards)
        with self.assertRaises(SignalValidationError):
            validate_htf_bias("SHORT", "UP", guards)
        validate_htf_bias("LONG", "UP", guards)
        validate_htf_bias("SHORT", "DOWN", guards)

    def test_unclear_never_vetoes_and_zero_disables(self):
        guards = GuardrailConfig(htf_bias=1)
        for regime in (None, "NONE", "RANGE", ""):
            validate_htf_bias("LONG", regime, guards)
            validate_htf_bias("SHORT", regime, guards)
        off = GuardrailConfig(htf_bias=0)
        validate_htf_bias("LONG", "DOWN", off)
        validate_htf_bias("SHORT", "UP", off)

    def test_end_to_end(self):
        long_vs_down = make_analysis("LONG")
        from dataclasses import replace

        long_vs_down = replace(long_vs_down, regime="DOWN")
        with self.assertRaises(SignalValidationError):
            validate_analysis(long_vs_down, guardrails=GuardrailConfig(htf_bias=1))
        long_with_up = replace(make_analysis("LONG"), regime="UP")
        candidate = validate_analysis(
            long_with_up, guardrails=GuardrailConfig(htf_bias=1)
        )
        self.assertEqual(candidate.decision, "LONG")

    def test_env_knob(self):
        self.assertEqual(default_guardrails(env={"BTCUSDT_HTF_BIAS": "0"}).htf_bias, 0)
        self.assertEqual(default_guardrails(env={}).htf_bias, 1)


# ── threading ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestHtfThreading(unittest.TestCase):
    def test_single_threads_note_and_regime(self):
        candles = make_candles(240)
        indicators = indicators_for(make_candles(240))
        model = SignalDecisionModel(decision="WAIT", confidence=50.0, reasoning="x")
        llm = MagicMock()
        structured = MagicMock()
        llm.with_structured_output.return_value = structured
        structured.invoke.return_value = model
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(
            candles, indicators, htf_note="4H bias: test", regime="UP"
        )
        self.assertEqual(result.regime, "UP")
        prompt = structured.invoke.call_args[0][0]
        bodies = [
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in (prompt if isinstance(prompt, list) else [prompt])
        ]
        self.assertIn("4H bias: test", "\n".join(bodies))

    def test_multi_signature(self):
        from signal_engine.multiagent import MultiAgentSignalAnalyzer

        params = inspect.signature(MultiAgentSignalAnalyzer.analyze).parameters
        self.assertIn("htf_note", params)
        self.assertIn("regime", params)


# ── scheduler ────────────────────────────────────────────────────────────


class FakeSchedulerData:
    """4H-aware fake: serves scripted 4H candles, 1H passthrough otherwise."""

    def __init__(self, candles_1h, candles_4h=None) -> None:
        self.candles_1h = list(candles_1h)
        self.candles_4h = list(candles_4h) if candles_4h is not None else []
        self.calls: list = []

    def fetch_closed_klines(
        self, symbol, interval, limit, *, end_time_ms=None, now_ms=None
    ):
        self.calls.append((symbol, interval, limit))
        if interval == "4h":
            return list(self.candles_4h)
        return [
            c
            for c in self.candles_1h
            if c.is_closed and (now_ms is None or c.close_time <= now_ms)
        ]


@pytest.mark.unit
class TestSchedulerHtf(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
        self.candles = make_candles(
            222, start_ms=int(self.now.timestamp() * 1000) - 220 * INTERVAL_MS
        )
        # 60 rising 4H candles -> clear UP regime.
        self.candles_4h = make_candles(60, trend=5.0)

    def test_fetch_only_on_analysis_path(self):
        md = FakeSchedulerData(self.candles, self.candles_4h)
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
        self.assertIn(("BTCUSDT", "4h", 60), md.calls)
        self.assertEqual(sched._tick_regime, "UP")
        self.assertIn("UPTREND", sched._tick_htf_note or "")
        # Blocked tick: detection (1h) still runs, but no 4H fetch and no
        # analyzer call happen past the active-signal gate.
        md.calls.clear()
        calls.clear()
        self.harness.repository.create_signal("BTCUSDT", "1h", "LONG", "100", "99", "101")
        blocked = sched.tick(now=datetime(2026, 9, 10, 13, 0, tzinfo=timezone.utc))
        self.assertEqual(blocked.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertNotIn(("BTCUSDT", "4h", 60), md.calls)
        self.assertEqual(calls, [])

    def test_default_analyzer_forwards(self):
        md = FakeSchedulerData(self.candles, self.candles_4h)
        sched = OneHourScheduler(
            self.harness.repository,
            market_data=md,
            engine=self.harness.engine,
            state=self.harness.state,
            candle_log=CandleLogRepository(self.harness.db),
        )
        sched._tick_htf_note = "4H bias: test"
        sched._tick_regime = "DOWN"
        seen: dict = {}

        def fake_analyze_signal(config, candles, indicators, **kwargs):
            seen.update(kwargs)
            return make_analysis("WAIT")

        with patch("signal_engine.scheduler.analyze_signal", fake_analyze_signal):
            sched._default_analyzer(self.candles, object())
        self.assertEqual(seen.get("htf_note"), "4H bias: test")
        self.assertEqual(seen.get("regime"), "DOWN")

    def test_empty_4h_degrades(self):
        md = FakeSchedulerData(self.candles, [])
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
        self.assertIsNone(sched._tick_regime)
        self.assertIsNone(sched._tick_htf_note)


# ── backtest wiring + CLI ────────────────────────────────────────────────


@pytest.mark.unit
class TestBacktestHtf(unittest.TestCase):
    def test_wiring_threads_and_forwards(self):
        seen: dict = {}

        def base(candles, indicators, *, symbol, timeframe, max_candles,
                 event_note=None, positioning=None, funding_rate=None,
                 htf_note=None, regime=None):
            seen.update(
                event_note=event_note, positioning=positioning,
                funding_rate=funding_rate, htf_note=htf_note, regime=regime,
            )
            return make_analysis("WAIT")

        analyze_fn = build_htf_wiring(analyze_base=base)
        hours = make_candles(220, start_ms=0)
        analyze_fn(
            hours, None, symbol="BTCUSDT", timeframe="1h", max_candles=20,
            event_note="ev", positioning="pos", funding_rate=0.0001,
        )
        # 220 rising 1H candles roll up to a clear UP 4H regime.
        self.assertEqual(seen["regime"], "UP")
        self.assertIn("UPTREND", seen["htf_note"])
        self.assertEqual(seen["event_note"], "ev")
        self.assertEqual(seen["positioning"], "pos")
        self.assertEqual(seen["funding_rate"], 0.0001)

    def test_wiring_short_history_degrades(self):
        seen: dict = {}

        def base(candles, indicators, **kwargs):
            seen.update(kwargs)
            return make_analysis("WAIT")

        analyze_fn = build_htf_wiring(analyze_base=base)
        analyze_fn(make_candles(10, start_ms=0), None)
        self.assertIsNone(seen.get("regime"))
        self.assertIsNone(seen.get("htf_note"))

    def test_cli_flag(self):
        from backtest.__main__ import _build_benchmark_parser

        args = _build_benchmark_parser().parse_args(
            ["--data", "d.json", "--model", "m", "--htf"]
        )
        self.assertTrue(args.htf)
        args2 = _build_benchmark_parser().parse_args(
            ["--data", "d.json", "--model", "m"]
        )
        self.assertFalse(args2.htf)


if __name__ == "__main__":
    unittest.main()
