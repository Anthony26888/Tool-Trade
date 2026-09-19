"""Phase 5 multi-agent debate tests (mocked LLM, no network).

``signal_engine/multiagent.py`` runs the five-role debate
(analyst > bull > bear > trader > risk manager) against the SAME closed-candle
context and the SAME provider/model configuration, plus the opt-in mode
resolution (``BTCUSDT_ANALYSIS_MODE``). These tests prove the event order,
prompt wiring, strictness (WAIT carries no prices, SL/Entry/TP ordering is
enforced by the Risk Manager), fail-safe behaviour, mode dispatch, and an
end-to-end scheduler run that persists a PENDING_ENTRY signal.
"""

from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from database.models import STATUS_PENDING_ENTRY
from signal_engine import (
    LLMConfig,
    MultiAgentSignalAnalyzer,
    OneHourScheduler,
    RateLimitedError,
    SchedulerOutcome,
    SignalAnalysis,
    SignalAnalysisError,
    analyze_signal,
    resolve_analysis_mode,
)
from signal_engine.scheduler import DEFAULT_WINDOW_CANDLES
from tests.signal_engine_test_helpers import (
    TempSignalDb,
    indicators_for,
    make_candles,
)

CONFIG = LLMConfig(provider="ollama", model="qwen3:8b")

_ROLE_ORDER = ["analyst", "bull", "bear", "trader", "risk"]

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)

_MISSING = object()


def _analyst_payload(bias: str = "LONG", summary: str = "ema stack bullish"):
    return {"bias": bias, "summary": summary}


def _case_payload(thesis: str, *evidence: str):
    return {"thesis": thesis, "key_evidence": list(evidence)}


def _decision_payload(**overrides):
    values = {
        "decision": "LONG",
        "confidence": 80.0,
        "reasoning": "uptrend continuation",
        "entry_price": 61000.0,
        "stop_loss": 60000.0,
        "take_profit": 64000.0,
    }
    values.update(overrides)
    return values


def _risk_payload(**overrides):
    values = _decision_payload()
    values["risk_note"] = "risk acceptable; stop distance within ATR"
    values.update(overrides)
    return values


def _long_results(**risk_overrides):
    return {
        "analyst": _analyst_payload(),
        "bull": _case_payload("uptrend continuation", "close above ema20"),
        "bear": _case_payload("overextension risk"),
        "trader": _decision_payload(),
        "risk": _risk_payload(**risk_overrides),
    }


def _make_llm(results: dict):
    """Return ``(llm, structured_by_role)`` with canned dict outputs per role."""

    llm = MagicMock()
    by_role: dict[str, MagicMock] = {}

    def _bind(role: str) -> MagicMock:
        structured = MagicMock()

        def _invoke(prompt):
            payload = results.get(role, _MISSING)
            if payload is _MISSING:
                raise AssertionError(f"no canned result registered for role {role!r}")
            if isinstance(payload, Exception):
                raise payload
            return payload

        structured.invoke.side_effect = _invoke
        by_role[role] = structured
        return structured

    def _with_structured_output(schema):
        del schema
        return _bind(_ROLE_ORDER[len(by_role)])

    llm.with_structured_output.side_effect = _with_structured_output
    return llm, by_role


class _StructuredUnsupportedLLM:
    """A provider that only accepts plain ``invoke(content=...)`` JSON strings."""

    def __init__(self, payloads: list[str]):
        self._payloads = iter(payloads)
        self.prompts: list = []

    def with_structured_output(self, schema):
        del schema
        raise NotImplementedError("provider does not support structured output")

    def invoke(self, prompt):
        self.prompts.append(prompt)
        try:
            content = next(self._payloads)
        except StopIteration as exc:  # pragma: no cover - safety net
            raise AssertionError("more invokes than canned payloads") from exc
        return SimpleNamespace(content=content)


class FakeSchedulerData:
    """Deterministic replacement for ``fetch_closed_klines`` (no network)."""

    def __init__(self, *candles) -> None:
        self.candles = list(candles)

    def fetch_closed_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        end_time_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list:
        return [c for c in self.candles if c.is_closed]


@pytest.mark.unit
class TestMultiAgentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles(240)
        self.indicators = indicators_for(self.candles)

    # ── happy paths ──────────────────────────────────────────────────────────

    def test_full_debate_produces_valid_long(self):
        llm, by_role = _make_llm(_long_results())
        analyzer = MultiAgentSignalAnalyzer(CONFIG, llm=llm)
        analysis = analyzer.analyze(
            self.candles, self.indicators, symbol="BTCUSDT", timeframe="1h"
        )
        self.assertIsInstance(analysis, SignalAnalysis)
        self.assertEqual(analysis.decision, "LONG")
        self.assertEqual(analysis.entry_price, 61000.0)
        self.assertEqual(analysis.stop_loss, 60000.0)
        self.assertEqual(analysis.take_profit, 64000.0)
        self.assertEqual(analysis.provider, "ollama")
        self.assertEqual(analysis.model, "qwen3:8b")
        self.assertEqual(analysis.symbol, "BTCUSDT")
        self.assertEqual(analysis.timeframe, "1h")
        self.assertIn("[Risk note]", analysis.reasoning)
        for role in _ROLE_ORDER:
            by_role[role].invoke.assert_called_once()

    def test_prompt_wiring_carries_context_and_prior_steps(self):
        llm, by_role = _make_llm(_long_results())
        analyzer = MultiAgentSignalAnalyzer(CONFIG, llm=llm)
        analyzer.analyze(self.candles, self.indicators)

        analyst_prompt = by_role["analyst"].invoke.call_args.args[0]
        self.assertEqual(analyst_prompt[0]["role"], "system")
        self.assertEqual(analyst_prompt[1]["role"], "user")

        bull_prompt = by_role["bull"].invoke.call_args.args[0]
        self.assertIn("### Technical Analyst", bull_prompt[1]["content"])
        self.assertIn("ema stack bullish", bull_prompt[1]["content"])

        bear_prompt = by_role["bear"].invoke.call_args.args[0]
        self.assertIn("### Technical Analyst", bear_prompt[1]["content"])

        trader_prompt = by_role["trader"].invoke.call_args.args[0]
        self.assertIn("### Technical Analyst", trader_prompt[1]["content"])
        self.assertIn("### Bull Researcher", trader_prompt[1]["content"])
        self.assertIn("### Bear Researcher", trader_prompt[1]["content"])
        self.assertIn("uptrend continuation", trader_prompt[1]["content"])

        risk_prompt = by_role["risk"].invoke.call_args.args[0]
        self.assertIn("### Trader Proposal", risk_prompt[1]["content"])
        self.assertIn("uptrend continuation", risk_prompt[1]["content"])

    def test_wait_final_carries_no_prices(self):
        llm, _ = _make_llm(
            {
                "analyst": _analyst_payload(bias="NEUTRAL", summary="mixed signals"),
                "bull": _case_payload("mild upside"),
                "bear": _case_payload("mild downside"),
                "trader": _decision_payload(decision="WAIT", confidence=50.0),
                "risk": _risk_payload(
                    decision="WAIT",
                    confidence=45.0,
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                ),
            }
        )
        analysis = MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
            self.candles, self.indicators
        )
        self.assertEqual(analysis.decision, "WAIT")
        self.assertIsNone(analysis.entry_price)
        self.assertIsNone(analysis.stop_loss)
        self.assertIsNone(analysis.take_profit)

    def test_valid_short_passes(self):
        results = _long_results()
        results["trader"] = _decision_payload(
            decision="SHORT",
            entry_price=59000.0,
            stop_loss=60000.0,
            take_profit=58000.0,
        )
        results["risk"] = _risk_payload(
            decision="SHORT",
            entry_price=59000.0,
            stop_loss=60000.0,
            take_profit=58000.0,
        )
        llm, _ = _make_llm(results)
        analysis = MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
            self.candles, self.indicators
        )
        self.assertEqual(analysis.decision, "SHORT")
        self.assertEqual(analysis.entry_price, 59000.0)

    # ── strictness ───────────────────────────────────────────────────────────

    def test_risk_gate_rejects_long_without_ordering(self):
        results = _long_results(
            decision="LONG", entry_price=61000.0, stop_loss=62000.0, take_profit=64000.0
        )
        llm, _ = _make_llm(results)
        with self.assertRaisesRegex(SignalAnalysisError, "ordering"):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )

    def test_risk_gate_rejects_short_without_ordering(self):
        results = _long_results(
            decision="SHORT",
            entry_price=59000.0,
            stop_loss=58000.0,
            take_profit=60000.0,
        )
        llm, _ = _make_llm(results)
        with self.assertRaisesRegex(SignalAnalysisError, "ordering"):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )

    def test_risk_gate_requires_all_prices_for_directional(self):
        results = _long_results(decision="LONG", take_profit=None)
        llm, _ = _make_llm(results)
        with self.assertRaisesRegex(
            SignalAnalysisError, "requires entry, stop-loss, and take-profit"
        ):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )

    def test_wait_with_prices_is_rejected(self):
        results = _long_results(decision="WAIT", entry_price=61000.0)
        llm, _ = _make_llm(results)
        with self.assertRaisesRegex(SignalAnalysisError, "WAIT must not propose"):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )

    # ── fail-safe ────────────────────────────────────────────────────────────

    def test_any_agent_failure_aborts_without_partial_signal(self):
        results = _long_results()
        results["bear"] = RuntimeError("bear model timeout")
        llm, by_role = _make_llm(results)
        with self.assertRaisesRegex(SignalAnalysisError, "Bear Researcher"):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )
        by_role["trader"].invoke.assert_not_called()
        by_role["risk"].invoke.assert_not_called()

    def test_agent_rate_limit_is_classified(self):
        results = _long_results()
        results["bull"] = RuntimeError("Error code: 429, rate limit exceeded")
        llm, _ = _make_llm(results)
        with self.assertRaises(RateLimitedError):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )

    def test_full_debate_counts_five_calls(self):
        llm, _ = _make_llm(_long_results())
        analysis = MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
            self.candles, self.indicators
        )
        self.assertEqual(analysis.llm_calls, 5)
        # Structured path drops the envelope: token counts stay NULL.
        self.assertIsNone(analysis.prompt_tokens)
        self.assertIsNone(analysis.total_tokens)

    def test_structured_fallback_parses_strict_json(self):
        payloads = [
            json.dumps(_analyst_payload()),
            json.dumps(_case_payload("uptrend continuation", "close above ema20")),
            json.dumps(_case_payload("overextension risk")),
            json.dumps(_decision_payload()),
            json.dumps(_risk_payload()),
        ]
        llm = _StructuredUnsupportedLLM(payloads)
        analysis = MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
            self.candles, self.indicators
        )
        self.assertEqual(analysis.decision, "LONG")
        self.assertEqual(len(llm.prompts), len(_ROLE_ORDER))

    def test_structured_fallback_rejects_invalid_json(self):
        payloads = [
            json.dumps(_analyst_payload()),
            json.dumps(_case_payload("uptrend continuation")),
            json.dumps(_case_payload("overextension risk")),
            json.dumps(_decision_payload()),
            "not-json-at-all",
        ]
        llm = _StructuredUnsupportedLLM(payloads)
        with self.assertRaisesRegex(SignalAnalysisError, "Risk Manager"):
            MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                self.candles, self.indicators
            )


@pytest.mark.unit
class TestAnalyzeSignalMode(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles(240)
        self.indicators = indicators_for(self.candles)

    def test_mode_multi_runs_five_agents(self):
        llm, by_role = _make_llm(_long_results())
        analysis = analyze_signal(
            CONFIG,
            self.candles,
            self.indicators,
            llm=llm,
            mode="multi",
            symbol="BTCUSDT",
            timeframe="1h",
        )
        self.assertEqual(analysis.decision, "LONG")
        self.assertEqual(len(by_role), len(_ROLE_ORDER))

    def test_mode_single_keeps_single_invoke(self):
        llm = MagicMock()
        structured = MagicMock()
        llm.with_structured_output.return_value = structured
        structured.invoke.return_value = _decision_payload()
        analysis = analyze_signal(
            CONFIG, self.candles, self.indicators, llm=llm, mode="single"
        )
        self.assertEqual(analysis.decision, "LONG")
        llm.with_structured_output.assert_called_once()
        structured.invoke.assert_called_once()

    def test_unknown_mode_falls_back_to_single(self):
        llm = MagicMock()
        structured = MagicMock()
        llm.with_structured_output.return_value = structured
        structured.invoke.return_value = _decision_payload()
        analysis = analyze_signal(
            CONFIG, self.candles, self.indicators, llm=llm, mode="committee"
        )
        self.assertEqual(analysis.decision, "LONG")


@pytest.mark.unit
class TestResolveAnalysisMode(unittest.TestCase):
    def test_env_lookup(self):
        cases = [
            ({}, False),
            ({"BTCUSDT_ANALYSIS_MODE": "multi"}, True),
            ({"BTCUSDT_ANALYSIS_MODE": "MULTI"}, True),
            ({"BTCUSDT_ANALYSIS_MODE": "  multi "}, True),
            ({"BTCUSDT_ANALYSIS_MODE": "single"}, False),
            ({"BTCUSDT_ANALYSIS_MODE": "1"}, False),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(resolve_analysis_mode(env), expected)

    def test_env_none_reads_os_environ(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(resolve_analysis_mode())
        with patch.dict(
            os.environ, {"BTCUSDT_ANALYSIS_MODE": "multi"}, clear=True
        ):
            self.assertTrue(resolve_analysis_mode())


@pytest.mark.unit
class TestMultiAgentThroughScheduler(unittest.TestCase):
    def test_multi_agent_persists_pending_entry(self):
        # ATR snapshot neutralized on purpose: the canned levels are
        # real-market scale while the synthetic candles have tiny volatility;
        # ATR-bound logic gets dedicated validator unit tests.
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        candles = make_candles(DEFAULT_WINDOW_CANDLES)
        md = FakeSchedulerData(*candles)
        llm, _ = _make_llm(_long_results())

        def analyzer(c, i):
            with patch(
                "signal_engine.multiagent._snapshot_atr", return_value=None
            ):
                return MultiAgentSignalAnalyzer(CONFIG, llm=llm).analyze(
                    c, i, symbol="BTCUSDT", timeframe="1h"
                )

        sched = OneHourScheduler(
            harness.repository,
            market_data=md,
            engine=harness.engine,
            state=harness.state,
            analyzer=analyzer,
        )
        result = sched.tick(now=NOW)
        self.assertEqual(result.outcome, SchedulerOutcome.CREATED)
        signal = harness.repository.get_active_signal()
        self.assertIsNotNone(signal)
        self.assertEqual(signal.status, STATUS_PENDING_ENTRY)
        self.assertEqual(signal.direction, "LONG")
        self.assertEqual(signal.entry, Decimal("61000"))
        self.assertEqual(signal.stop_loss, Decimal("60000"))
        self.assertEqual(signal.take_profit, Decimal("64000"))
