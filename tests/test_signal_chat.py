"""Unit tests for the read-only Signal Chat prompt/context (web/signal_chat.py).

The chat must only ever expose facts that already exist in the ledger: signal
fields, position risk, account risk settings and account statistics. It must
never inject future data nor ask the model to invent numbers.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import pytest

from web.signal_chat import (
    CANNED_NO_SIGNAL,
    SYSTEM_PROMPT,
    build_messages,
    iter_text_chunks,
)


class _Chunk:
    def __init__(self, content) -> None:
        self.content = content


@pytest.mark.unit
class TestBuildMessages(unittest.TestCase):
    CONTEXT = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "signal": {
            "direction": "LONG",
            "entry": "100.00",
            "stop_loss": "99.00",
            "take_profit": "101.00",
            "confidence": 70,
            "risk_reward": "1.00",
            "rationale": "Bullish momentum alignment of EMA20 over EMA200.",
        },
        "statistics": {"wins": 3, "losses": 1, "net_pnl": "1.5"},
    }

    def test_messages_pair_with_system_prompt(self):
        messages = build_messages("Why LONG?", self.CONTEXT)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn(SYSTEM_PROMPT, messages[0]["content"])

    def test_user_message_embeds_question_and_context(self):
        user = build_messages("Why LONG?", self.CONTEXT)[1]["content"]
        self.assertIn("QUESTION: Why LONG?", user)
        self.assertIn('"direction": "LONG"', user)
        self.assertIn("Bullish momentum alignment", user)
        self.assertIn('"statistics"', user)
        self.assertIn('"risk_reward": "1.00"', user)

    def test_context_serializes_decimals_as_text(self):
        context = {"signal": {"entry": Decimal("100.00")}}
        user = build_messages("entry?", context)[1]["content"]
        self.assertIn('"entry": "100.00"', user)

    def test_system_prompt_forbids_inventing_and_orders(self):
        self.assertIn("Never invent", SYSTEM_PROMPT)
        self.assertIn("DISPLAY-ONLY", SYSTEM_PROMPT.upper())

    def test_prompt_has_no_future_data_reference(self):
        joined = " ".join(m["content"] for m in build_messages("future?", self.CONTEXT))
        self.assertNotIn("future candle", joined.lower())


@pytest.mark.unit
class TestIterTextChunks(unittest.TestCase):
    def test_yields_only_nonempty_content(self):
        chunks = [_Chunk("Hel"), _Chunk(""), _Chunk(" lo"), _Chunk(None)]
        self.assertEqual(list(iter_text_chunks(iter(chunks))), ["Hel", " lo"])

    def test_handles_non_string_content(self):
        self.assertEqual(list(iter_text_chunks(iter([_Chunk(42)]))), ["42"])

    def test_canned_no_signal_is_display_only(self):
        self.assertIn("active signal", CANNED_NO_SIGNAL)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
