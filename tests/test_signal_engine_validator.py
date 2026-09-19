"""Phase 6 unit tests: validation boundary (signal_engine/validator.py).

These tests prove the validator is the second gate after the AI: Decimals
only, no float comparisons, strict LONG/SHORT ordering, no rounding of prices,
and the Phase 5 confidence contract.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import pytest

from database.models import SignalValidationError
from signal_engine import (
    GuardrailConfig,
    to_price,
    validate_analysis,
    validate_confidence,
    validate_decision,
)
from tests.signal_engine_test_helpers import make_analysis

HALF_OPEN, OPEN, TP = Decimal("99.999999999999"), Decimal("100"), Decimal("100.000000000001")


@pytest.mark.unit
class TestValidateDecision(unittest.TestCase):
    def test_accepts_all_valid_decisions(self):
        for decision in ("LONG", "SHORT", "WAIT"):
            with self.subTest(decision=decision):
                self.assertEqual(validate_decision(decision), decision)

    def test_rejects_unknown_decisions(self):
        for bad in ("BUY", "SELL", "HOLD", "buy", "", 1, None):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                validate_decision(bad)


@pytest.mark.unit
class TestValidateConfidence(unittest.TestCase):
    def test_bounds_accepted(self):
        self.assertEqual(validate_confidence(0), 0)
        self.assertEqual(validate_confidence(50), 50)
        self.assertEqual(validate_confidence(100), 100)

    def test_deterministic_half_up_normalization(self):
        self.assertEqual(validate_confidence(83.5), 84)
        self.assertEqual(validate_confidence(83.4), 83)
        self.assertEqual(validate_confidence(84.5), 85)

    def test_out_of_range_rejected(self):
        for bad in (-1, 101, 100.5, -0.1):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                validate_confidence(bad)

    def test_non_finite_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                validate_confidence(bad)

    def test_non_numeric_rejected(self):
        for bad in (None, True, "high", "abc"):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                validate_confidence(bad)


@pytest.mark.unit
class TestToPrice(unittest.TestCase):
    def test_decimal_input_is_kept_exact(self):
        value = Decimal("61000.123456789012345678901234")
        self.assertEqual(to_price(value, "entry"), value)

    def test_float_input_converts_via_exact_decimal_string(self):
        self.assertEqual(to_price(61000.1, "entry"), Decimal("61000.1"))
        self.assertEqual(to_price(61000.0, "entry"), Decimal("61000.0"))

    def test_numeric_string_input(self):
        self.assertEqual(to_price("61000.500", "entry"), Decimal("61000.500"))

    def test_zero_and_negative_rejected(self):
        for bad in (0, 0.0, Decimal("0"), "-1", -0.5, Decimal("-0.5")):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                to_price(bad, "entry")

    def test_nan_and_infinity_rejected(self):
        for bad in (
            float("nan"),
            float("inf"),
            float("-inf"),
            Decimal("NaN"),
            Decimal("Infinity"),
            Decimal("-Infinity"),
        ):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                to_price(bad, "entry")

    def test_invalid_decimal_rejected(self):
        for bad in ("abc", "", "1,000", None, object()):
            with self.subTest(bad=bad), self.assertRaises(SignalValidationError):
                to_price(bad, "entry")


@pytest.mark.unit
class TestValidateAnalysis(unittest.TestCase):
    def test_valid_long_becomes_exact_candidate(self):
        candidate = validate_analysis(make_analysis())
        self.assertEqual(candidate.decision, "LONG")
        self.assertEqual(candidate.entry, Decimal("61000.0"))
        self.assertEqual(candidate.stop_loss, Decimal("60000.0"))
        self.assertEqual(candidate.take_profit, Decimal("64000.0"))
        self.assertEqual(candidate.confidence, 80)
        self.assertEqual(candidate.provider, "ollama")
        self.assertEqual(candidate.model_name, "qwen3:8b")
        self.assertEqual(candidate.symbol, "BTCUSDT")
        self.assertEqual(candidate.timeframe, "1h")
        self.assertEqual(candidate.analysis_timestamp, "2026-09-10T01:00:00.000Z")
        self.assertEqual(candidate.market_timestamp, "2026-09-10T00:00:00.000Z")
        self.assertEqual(candidate.candle_close_price, Decimal("61050.5"))

    def test_valid_short_becomes_exact_candidate(self):
        candidate = validate_analysis(
            make_analysis("SHORT", entry_price=59000.0, stop_loss=60000.0, take_profit=57700.0)
        )
        self.assertEqual(candidate.decision, "SHORT")
        self.assertEqual(candidate.entry, Decimal("59000.0"))

    def test_wait_rejected_by_validator(self):
        with self.assertRaises(SignalValidationError):
            validate_analysis(make_analysis("WAIT"))

    def test_non_analysis_rejected(self):
        with self.assertRaises(SignalValidationError):
            validate_analysis({"decision": "LONG"})

    def test_missing_price_levels_rejected(self):
        for field in ("entry_price", "stop_loss", "take_profit"):
            with self.subTest(field=field), self.assertRaises(SignalValidationError) as ctx:
                validate_analysis(make_analysis(**{field: None}))
            self.assertIn(field, str(ctx.exception))

    def test_invalid_long_ordering_rejected(self):
        cases = (
            {"entry_price": 61000.0, "stop_loss": 61000.0, "take_profit": 64000.0},  # SL == entry
            {"entry_price": 61000.0, "stop_loss": 62000.0, "take_profit": 64000.0},  # SL > entry
            {"entry_price": 61000.0, "stop_loss": 60000.0, "take_profit": 61000.0},  # TP == entry
            {"entry_price": 61000.0, "stop_loss": 60000.0, "take_profit": 60000.0},  # SL >= TP
        )
        for overrides in cases:
            with self.subTest(**overrides), self.assertRaises(SignalValidationError):
                validate_analysis(make_analysis(**overrides))

    def test_invalid_short_ordering_rejected(self):
        cases = (
            {"entry_price": 59000.0, "stop_loss": 60000.0, "take_profit": 59000.0},  # TP == entry
            {"entry_price": 59000.0, "stop_loss": 60000.0, "take_profit": 61000.0},  # TP > entry
            {"entry_price": 60000.0, "stop_loss": 60000.0, "take_profit": 58000.0},  # entry == SL
            {"entry_price": 61000.0, "stop_loss": 60000.0, "take_profit": 58000.0},  # entry > SL
        )
        for overrides in cases:
            with self.subTest(**overrides), self.assertRaises(SignalValidationError):
                validate_analysis(make_analysis("SHORT", **overrides))

    def test_decimal_precision_preserved_no_rounding(self):
        analysis = make_analysis(
            entry_price=Decimal("61000.123456789012345678901234"),
            stop_loss=Decimal("60888.000000000000000001"),
            take_profit=Decimal("61234.9876543210987654321"),
        )
        candidate = validate_analysis(analysis)
        self.assertEqual(candidate.entry, Decimal("61000.123456789012345678901234"))
        self.assertEqual(candidate.stop_loss, Decimal("60888.000000000000000001"))
        self.assertEqual(candidate.take_profit, Decimal("61234.9876543210987654321"))

    def test_near_equal_levels_use_decimal_ordering(self):
        # Guardrails relaxed on purpose: this test proves Decimal ordering
        # precision, not trade structure (dust distances fail fee/RR rules).
        relaxed = GuardrailConfig(
            min_confidence=0, min_risk_reward=Decimal("0"), fee_rate=Decimal("0")
        )
        candidate = validate_analysis(
            make_analysis(
                entry_price=OPEN, stop_loss=HALF_OPEN, take_profit=TP
            ),
            guardrails=relaxed,
        )
        self.assertEqual(candidate.entry, Decimal("100"))
        # Decimal comparison: 99.999... < 100 < 100.000...
        self.assertTrue(candidate.stop_loss < candidate.entry < candidate.take_profit)

    def test_nan_and_infinity_prices_rejected(self):
        for field, value in (
            ("entry_price", float("nan")),
            ("stop_loss", float("inf")),
            ("take_profit", float("-inf")),
        ):
            with self.subTest(field=field), self.assertRaises(SignalValidationError):
                validate_analysis(make_analysis(**{field: value}))

    def test_invalid_decimal_price_rejected(self):
        with self.assertRaises(SignalValidationError):
            validate_analysis(make_analysis(entry_price="one hundred"))

    def test_zero_prices_rejected(self):
        with self.assertRaises(SignalValidationError):
            validate_analysis(make_analysis(entry_price=0.0))

    def test_candle_close_price_optional(self):
        candidate = validate_analysis(make_analysis(candle_close_price=None))
        self.assertIsNone(candidate.candle_close_price)


@pytest.mark.unit
class TestGuardrails(unittest.TestCase):
    """Phase B trade-structure guardrails (policy, not contract)."""

    def test_confidence_below_minimum_rejected(self):
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(make_analysis(confidence=59.0))
        self.assertIn("below minimum 60", str(ctx.exception))

    def test_confidence_at_minimum_accepted(self):
        candidate = validate_analysis(make_analysis(confidence=60.0))
        self.assertEqual(candidate.confidence, 60)

    def test_min_confidence_from_env(self):
        guards = GuardrailConfig.from_env({"BTCUSDT_MIN_CONFIDENCE": "70"})
        self.assertEqual(guards.min_confidence, 70)
        with self.assertRaises(SignalValidationError):
            validate_analysis(make_analysis(confidence=69.0), guardrails=guards)
        candidate = validate_analysis(make_analysis(confidence=70.0), guardrails=guards)
        self.assertEqual(candidate.confidence, 70)

    def test_min_confidence_zero_disables_the_check(self):
        guards = GuardrailConfig.from_env({"BTCUSDT_MIN_CONFIDENCE": "0"})
        candidate = validate_analysis(make_analysis(confidence=5.0), guardrails=guards)
        self.assertEqual(candidate.confidence, 5)

    def test_invalid_env_falls_back_to_default(self):
        guards = GuardrailConfig.from_env({"BTCUSDT_MIN_CONFIDENCE": "high"})
        self.assertEqual(guards.min_confidence, 60)
        guards = GuardrailConfig.from_env({"BTCUSDT_MIN_RISK_REWARD": "nan"})
        self.assertEqual(guards.min_risk_reward, Decimal("1.2"))

    def test_low_risk_reward_rejected(self):
        # LONG 61000/60000/61500: RR = 500/1000 = 0.5 < 1.2.
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(make_analysis(take_profit=61500.0))
        self.assertIn("risk-reward", str(ctx.exception))

    def test_risk_reward_disabled_at_zero(self):
        guards = GuardrailConfig(
            min_confidence=0,
            min_risk_reward=Decimal("0"),
            fee_rate=Decimal("0"),
        )
        candidate = validate_analysis(make_analysis(take_profit=61500.0), guardrails=guards)
        self.assertEqual(candidate.take_profit, Decimal("61500.0"))

    def test_take_profit_below_fee_cover_rejected(self):
        # LONG 61000/60965/61042: RR = 42/35 = 1.2 but TP distance 42
        # < 61000*0.0004*2 = 48.8 fee cover.
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(
                make_analysis(stop_loss=60965.0, take_profit=61042.0)
            )
        self.assertIn("fees", str(ctx.exception))

    def test_fee_check_disabled_at_zero(self):
        guards = GuardrailConfig(
            min_confidence=0,
            min_risk_reward=Decimal("0"),
            fee_rate=Decimal("0"),
        )
        candidate = validate_analysis(
            make_analysis(stop_loss=60965.0, take_profit=61042.0), guardrails=guards
        )
        self.assertEqual(candidate.take_profit, Decimal("61042.0"))

    def test_atr_bounds_accept_sane_distances(self):
        import dataclasses

        # ATR 500: SL 1000 = 2xATR, TP 3000 = 6xATR, entry 50.5 from close.
        analysis = dataclasses.replace(make_analysis(), atr=500.0)
        candidate = validate_analysis(analysis)
        self.assertEqual(candidate.entry, Decimal("61000.0"))

    def test_stop_too_tight_rejected(self):
        import dataclasses

        # SL distance 200 < 0.5*500 = 250.
        analysis = dataclasses.replace(
            make_analysis(stop_loss=60800.0), atr=500.0
        )
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(analysis)
        self.assertIn("stop-loss distance", str(ctx.exception))

    def test_stop_too_wide_rejected(self):
        import dataclasses

        # SL distance 3000 > 5*500 = 2500 (TP keeps RR 5000/3000 valid).
        analysis = dataclasses.replace(
            make_analysis(stop_loss=58000.0, take_profit=66000.0), atr=500.0
        )
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(analysis)
        self.assertIn("stop-loss distance", str(ctx.exception))

    def test_take_profit_too_far_rejected(self):
        import dataclasses

        # TP distance 5000 > 8*500 = 4000.
        analysis = dataclasses.replace(
            make_analysis(take_profit=66000.0), atr=500.0
        )
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(analysis)
        self.assertIn("take-profit distance", str(ctx.exception))

    def test_stale_entry_rejected(self):
        import dataclasses

        # |61000 - 60000| = 1000 > 1.5*500 = 750.
        analysis = dataclasses.replace(
            make_analysis(candle_close_price=60000.0), atr=500.0
        )
        with self.assertRaises(SignalValidationError) as ctx:
            validate_analysis(analysis)
        self.assertIn("stale", str(ctx.exception))

    def test_missing_atr_skips_atr_checks(self):
        # make_analysis carries no ATR: identical levels must still pass.
        candidate = validate_analysis(make_analysis())
        self.assertEqual(candidate.entry, Decimal("61000.0"))
