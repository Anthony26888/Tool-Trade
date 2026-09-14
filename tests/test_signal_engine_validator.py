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
            make_analysis("SHORT", entry_price=59000.0, stop_loss=60000.0, take_profit=58000.0)
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
        candidate = validate_analysis(
            make_analysis(
                entry_price=OPEN, stop_loss=HALF_OPEN, take_profit=TP
            )
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
