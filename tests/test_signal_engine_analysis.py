"""Phase 5 unit tests: strict LONG/SHORT/WAIT LLM analysis.

``signal_engine/analysis.py`` turns the closed-candle context into a structured
decision. These tests mock the LLM (no network), and prove strict parsing,
fail-safe error handling, prompt constraints, and that the path never touches
vendor dataflows, the database, or Binance order execution.
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from signal_engine import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DECISIONS,
    LLMConfig,
    RateLimitedError,
    SignalAnalysisError,
    SignalAnalyzer,
    SignalDecision,
    SignalDecisionModel,
    analyze_signal,
    build_prompt,
)
from signal_engine.analysis import _extract_usage, _is_rate_limited, _reset_hint
from signal_engine.context import build_analysis_context
from tests.signal_engine_test_helpers import indicators_for, make_candles

CONFIG = LLMConfig(provider="ollama", model="qwen3:8b")


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _context_input():
    return make_candles(240), indicators_for(make_candles(240))


def _long_payload(**overrides):
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


def _short_payload(**overrides):
    values = {
        "decision": "SHORT",
        "confidence": 75.0,
        "reasoning": "distribution forming",
        "entry_price": 59000.0,
        "stop_loss": 60000.0,
        "take_profit": 58000.0,
    }
    values.update(overrides)
    return values


def _wait_payload(**overrides):
    values = {"decision": "WAIT", "confidence": 50.0, "reasoning": "mixed signals"}
    values.update(overrides)
    return values


def _structured_llm(model=None, side_effect=None, raw_result=None):
    llm = MagicMock()
    structured = MagicMock()

    def _bind(schema, **kwargs):
        if kwargs.get("include_raw"):
            if raw_result is None:
                # Simulate a provider without include_raw support.
                raise TypeError("with_structured_output() got an unexpected keyword 'include_raw'")
            raw_structured = MagicMock()
            raw_structured.invoke.return_value = raw_result
            return raw_structured
        return structured

    llm.with_structured_output.side_effect = _bind
    if side_effect is not None:
        structured.invoke.side_effect = side_effect
    elif model is not None:
        structured.invoke.return_value = model
    return llm, structured


@pytest.mark.unit
class TestSignalAnalyzer(unittest.TestCase):
    def test_long_structured_analysis(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.decision, "LONG")
        self.assertEqual(result.confidence, 80.0)
        self.assertEqual(result.entry_price, 61000.0)
        self.assertEqual(result.stop_loss, 60000.0)
        self.assertEqual(result.take_profit, 64000.0)
        structured.invoke.assert_called_once()

    def test_short_structured_analysis(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_short_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.decision, "SHORT")
        self.assertEqual(result.confidence, 75.0)

    def test_wait_structured_analysis(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_wait_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.decision, "WAIT")
        self.assertIsNone(result.entry_price)
        self.assertIsNone(result.stop_loss)
        self.assertIsNone(result.take_profit)

    def test_wait_with_price_levels_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            SignalDecisionModel(
                **_wait_payload(
                    entry_price=60000.0, stop_loss=59000.0, take_profit=62000.0
                )
            )
        )
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_wait_with_placeholder_prices_is_valid(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            _wait_payload(entry_price="n/a", stop_loss="None", take_profit="null")
        )
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.decision, "WAIT")
        self.assertIsNone(result.entry_price)

    def test_invalid_decision_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(_long_payload(decision="BUY"))
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_confidence_above_max_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            _long_payload(confidence=CONFIDENCE_MAX + 1)
        )
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_confidence_below_min_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(_long_payload(confidence=-0.5))
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_llm_failure_produces_no_signal(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            side_effect=RuntimeError("provider timeout")
        )
        with self.assertRaises(SignalAnalysisError) as ctx:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertIn("LLM analysis failed", str(ctx.exception))

    def test_rate_limit_failure_is_classified(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            side_effect=RuntimeError(
                "Error code: 429 - rate limit exceeded, retry after 20"
            )
        )
        with self.assertRaises(RateLimitedError) as ctx:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertIn("retry after 20s", str(ctx.exception))

    def test_non_rate_limit_failure_stays_generic(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            side_effect=RuntimeError("provider timeout")
        )
        with self.assertRaises(SignalAnalysisError) as ctx:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertNotIsInstance(ctx.exception, RateLimitedError)

    def test_rate_limit_hint_matching(self):
        for text in (
            "Error code: 429",
            "Rate limit reached for free models",
            "RATE_LIMIT exceeded",
            "Too Many Requests",
            "quota exceeded for the day",
            "upstream provider at capacity",
            "server overloaded, try again",
        ):
            self.assertTrue(_is_rate_limited(text), text)
        for text in (
            "provider timeout",
            "connection reset by peer",
            "500 internal server error",
            "API key is not set",
            "malformed JSON response",
        ):
            self.assertFalse(_is_rate_limited(text), text)

    def test_reset_hint_extraction(self):
        self.assertEqual(
            _reset_hint("rate limit, retry after 45"), " (provider asks to retry after 45s)"
        )
        self.assertEqual(
            _reset_hint("429: Retry-After: 120"), " (provider asks to retry after 120s)"
        )
        self.assertEqual(_reset_hint("provider timeout"), "")

    def test_structured_none_result_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(model=None)
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_structured_json_string_is_parsed(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(model=json.dumps(_short_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.decision, "SHORT")

    def test_malformed_json_string_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(model="{{not json}")
        with self.assertRaises(SignalAnalysisError) as ctx:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertIn("JSON", str(ctx.exception))

    def test_non_object_json_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(model=json.dumps([1, 2, 3]))
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_unsupported_raw_type_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(model=42)
        with self.assertRaises(SignalAnalysisError) as ctx:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertIn("unsupported output type", str(ctx.exception))

    def test_fallback_strict_json_when_structured_unsupported(self):
        candles, indicators = _context_input()
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("unsupported")
        llm.invoke.return_value = SimpleNamespace(content=json.dumps(_long_payload()))
        analyzer = SignalAnalyzer(CONFIG, llm=llm)
        self.assertIsNone(analyzer.structured_llm)
        result = analyzer.analyze(candles, indicators)
        self.assertEqual(result.decision, "LONG")

    def test_quota_single_call_and_usage_from_metadata(self):
        candles, indicators = _context_input()
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("unsupported")
        llm.invoke.return_value = SimpleNamespace(
            content=json.dumps(_long_payload()),
            response_metadata={"token_usage": {"prompt_tokens": 1500, "completion_tokens": 180, "total_tokens": 1680}},
        )
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.prompt_tokens, 1500)
        self.assertEqual(result.completion_tokens, 180)
        self.assertEqual(result.total_tokens, 1680)

    def test_quota_tokens_estimated_without_metadata(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.llm_calls, 1)
        # No usage metadata anywhere: counts are ~4-chars/token estimates,
        # always flagged so the UI can prefix "~".
        self.assertTrue(result.tokens_estimated)
        self.assertGreater(result.prompt_tokens, 0)
        self.assertGreater(result.completion_tokens, 0)
        self.assertEqual(
            result.total_tokens, result.prompt_tokens + result.completion_tokens
        )

    def test_quota_tokens_metered_via_include_raw(self):
        from types import SimpleNamespace as _NS

        candles, indicators = _context_input()
        model = SignalDecisionModel(**_long_payload())
        raw_message = _NS(
            usage_metadata={
                "input_tokens": 1200,
                "output_tokens": 300,
                "total_tokens": 1500,
            }
        )
        llm, structured = _structured_llm(
            model, raw_result={"parsed": model, "raw": raw_message, "parsing_error": None}
        )
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertFalse(result.tokens_estimated)
        self.assertEqual(result.prompt_tokens, 1200)
        self.assertEqual(result.completion_tokens, 300)
        self.assertEqual(result.total_tokens, 1500)
        structured.invoke.assert_not_called()

    def test_include_raw_parsing_error_rejected(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(
            SignalDecisionModel(**_long_payload()),
            raw_result={"parsed": None, "raw": None, "parsing_error": "bad json"},
        )
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_call_structured_shapes(self):
        from signal_engine.analysis import _call_structured

        model = SignalDecisionModel(**_long_payload())
        # Metered envelope.
        raw_msg = SimpleNamespace(
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        )
        raw_variant = MagicMock()
        raw_variant.invoke.return_value = {
            "parsed": model,
            "raw": raw_msg,
            "parsing_error": None,
        }
        content, usage = _call_structured(MagicMock(), raw_variant, "p", "agent")
        self.assertIs(content, model)
        self.assertEqual(usage["prompt_tokens"], 10)
        # Non-dict raw result degrades to the parsed-only binding.
        plain = MagicMock()
        plain.invoke.return_value = model
        odd_raw = MagicMock()
        odd_raw.invoke.return_value = model
        content, usage = _call_structured(plain, odd_raw, "p", "agent")[:2]
        self.assertIs(content, model)
        self.assertIsNone(usage)
        plain.invoke.assert_called_once()
        # Parsing errors and empty results raise like the old path.
        bad = MagicMock()
        bad.invoke.return_value = {"parsed": None, "raw": None, "parsing_error": "x"}
        with self.assertRaises(SignalAnalysisError):
            _call_structured(plain, bad, "p", "agent")
        with self.assertRaises(SignalAnalysisError):
            _call_structured(None, None, "p", "agent")

    def test_prompt_text_and_estimate(self):
        from signal_engine.analysis import _estimate_usage, _prompt_text

        self.assertEqual(_prompt_text("abc"), "abc")
        self.assertIn("hi", _prompt_text([{"role": "user", "content": "hi"}]))
        self.assertIn("yo", _prompt_text([SimpleNamespace(content="yo")]))
        est = _estimate_usage("x" * 400, SimpleNamespace())
        self.assertEqual(est["prompt_tokens"], 100)
        self.assertEqual(
            est["total_tokens"], est["prompt_tokens"] + est["completion_tokens"]
        )

    def test_extract_usage_variants(self):
        self.assertEqual(
            _extract_usage(SimpleNamespace(response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}})),
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual(
            _extract_usage(SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})),
            {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
        self.assertIsNone(_extract_usage(SimpleNamespace(content="{}")))
        self.assertIsNone(_extract_usage(SimpleNamespace(response_metadata={"token_usage": {"prompt_tokens": -1}})))
        self.assertIsNone(_extract_usage(object()))

    def test_fallback_plain_string_response(self):
        candles, indicators = _context_input()
        llm = MagicMock()
        llm.with_structured_output.side_effect = AttributeError("not supported")
        llm.invoke.return_value = json.dumps(_wait_payload())
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.decision, "WAIT")

    def test_fallback_malformed_rejected(self):
        candles, indicators = _context_input()
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("unsupported")
        llm.invoke.return_value = SimpleNamespace(content="not json at all")
        with self.assertRaises(SignalAnalysisError):
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)

    def test_result_metadata(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "qwen3:8b")
        self.assertEqual(result.symbol, "BTCUSDT")
        self.assertEqual(result.timeframe, "1h")
        self.assertRegex(
            result.analysis_timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        )

    def test_analysis_tied_to_latest_closed_candle(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        latest = candles[-1]
        self.assertEqual(result.market_timestamp, _ms_to_iso(latest.timestamp))
        self.assertEqual(result.closed_at, _ms_to_iso(latest.close_time))
        self.assertEqual(result.candle_close_price, latest.close)

    def test_to_dict_round_trip(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        result = SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        data = result.to_dict()
        self.assertEqual(data["decision"], "LONG")
        self.assertEqual(data["confidence"], 80.0)
        self.assertEqual(data["provider"], "ollama")
        self.assertEqual(data["model"], "qwen3:8b")
        self.assertEqual(data["timeframe"], "1h")

    def test_analyze_signal_convenience(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        result = analyze_signal(CONFIG, candles, indicators, llm=llm)
        self.assertEqual(result.decision, "LONG")
        structured.invoke.assert_called_once()

    def test_prompt_constraints_present(self):
        candles, indicators = _context_input()
        context = build_analysis_context(candles, indicators)
        prompt = build_prompt(context)
        self.assertEqual([m["role"] for m in prompt], ["system", "user"])
        combined = "\n".join(m["content"] for m in prompt)
        for needle in (
            "BTCUSDT",
            "Binance USDT-M Futures",
            "1H",
            "closed",
            "LONG",
            "SHORT",
            "WAIT",
            "0 to 100",
            "Do not call external tools",
            "news",
            "invent",
        ):
            self.assertIn(needle, combined)

    def test_analysis_places_no_binance_orders(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        with patch("binance.client.BinanceFuturesClient") as client_cls:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
            client_cls.assert_not_called()

    def test_analysis_never_touches_database(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        with patch("database.database.SignalRepository") as repo_cls:
            SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
            repo_cls.assert_not_called()

    def test_analysis_never_invokes_vendor_dataflows(self):
        candles, indicators = _context_input()
        llm, structured = _structured_llm(SignalDecisionModel(**_long_payload()))
        vendor_targets = (
            "yfinance.download",
            "tradingagents.agents.utils.core_stock_tools.get_stock_data",
            "tradingagents.agents.utils.news_data_tools.get_news",
            "tradingagents.dataflows.yfinance_news.get_news_yfinance",
        )
        mocks = []
        for target in vendor_targets:
            patcher = patch(target)
            mocks.append((target, patcher.start()))
            self.addCleanup(patcher.stop)
        SignalAnalyzer(CONFIG, llm=llm).analyze(candles, indicators)
        for target, mock in mocks:
            with self.subTest(target=target):
                mock.assert_not_called()

    def test_import_introduces_no_vendor_dataflow_modules(self):
        import importlib

        target_names = (
            "signal_engine",
            "signal_engine.context",
            "signal_engine.analysis",
            "signal_engine.llm",
            "signal_engine.engine",
            "signal_engine.validator",
            "signal_engine.state",
        )
        forbidden = (
            "yfinance",
            "tradingagents.dataflows",
            "tradingagents.agents",
        )
        saved = {name: sys.modules.pop(name, None) for name in target_names + forbidden}
        try:
            fresh = importlib.import_module("signal_engine.analysis")
            self.assertTrue(callable(fresh.analyze_signal))
            for name in forbidden:
                self.assertNotIn(name, sys.modules, f"{name} must not be imported")
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module


@pytest.mark.unit
class TestValidations(unittest.TestCase):
    def test_valid_long_payload_parses(self):
        model = SignalDecisionModel(**_long_payload())
        self.assertEqual(model.decision, SignalDecision.LONG)
        self.assertEqual(model.confidence, 80.0)
        self.assertEqual(model.entry_price, 61000.0)

    def test_invalid_decision_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            SignalDecisionModel.model_validate(
                {"decision": "BUY", "confidence": 50.0, "reasoning": "x"}
            )

    def test_confidence_out_of_range_raises_validation_error(self):
        for bad_confidence in (CONFIDENCE_MAX + 1, CONFIDENCE_MIN - 0.5, -100):
            with self.assertRaises(ValidationError):
                SignalDecisionModel.model_validate(
                    {
                        "decision": "LONG",
                        "confidence": bad_confidence,
                        "reasoning": "x",
                    }
                )

    def test_non_positive_price_rejected(self):
        with self.assertRaises(ValidationError):
            SignalDecisionModel.model_validate(_long_payload(entry_price=0.0))
        with self.assertRaises(ValidationError):
            SignalDecisionModel.model_validate(_long_payload(stop_loss=-1.0))

    def test_placeholder_prices_coerced_to_none(self):
        model = SignalDecisionModel(
            decision="LONG",
            confidence=60.0,
            reasoning="x",
            entry_price="n/a",
            stop_loss="None",
            take_profit="null",
        )
        self.assertIsNone(model.entry_price)
        self.assertIsNone(model.stop_loss)
        self.assertIsNone(model.take_profit)

    def test_empty_price_coerced_to_none(self):
        model = SignalDecisionModel(**_long_payload(entry_price="", take_profit=""))
        self.assertIsNone(model.entry_price)
        self.assertIsNone(model.take_profit)

    def test_declared_decisions(self):
        self.assertEqual(DECISIONS, ("LONG", "SHORT", "WAIT"))
        self.assertEqual(
            [d.value for d in SignalDecision], ["LONG", "SHORT", "WAIT"]
        )

    def test_default_price_fields_are_none(self):
        model = SignalDecisionModel(decision="WAIT", confidence=50.0, reasoning="x")
        self.assertIsNone(model.entry_price)
        self.assertIsNone(model.stop_loss)
        self.assertIsNone(model.take_profit)
