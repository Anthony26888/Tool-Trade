"""Phase 11 unit tests for Telegram notifications (signal_engine/telegram.py).

Covers configuration-from-env, the HTTP client (transient-vs-permanent retry,
sanitized errors, no real network), message formatting (Decimal prices, no
sensitive data), and the notifier adapter rules (disabled = zero I/O,
ambiguity dedup, truncation).
"""

from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

import pytest
import requests

from database.models import Signal
from signal_engine import (
    ENV_TELEGRAM_BOT_TOKEN,
    ENV_TELEGRAM_CHAT_ID,
    ENV_TELEGRAM_ENABLED,
    MAX_MESSAGE_CHARS,
    RefreshableTelegramNotifier,
    TelegramClient,
    TelegramConfig,
    TelegramConfigError,
    TelegramNotifier,
    TelegramSendResult,
    format_price,
    format_signal_ambiguous,
    format_signal_created,
    format_signal_opened,
    format_signal_sl,
    format_signal_tp,
    format_timestamp,
    telegram_config_from_env,
    validate_telegram_config,
)

OK_PAYLOAD = {"ok": True, "result": {"message_id": 42}}


def _signal(**overrides) -> Signal:
    defaults = {
        "id": 7,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "direction": "LONG",
        "status": "PENDING_ENTRY",
        "entry": Decimal("61000"),
        "stop_loss": Decimal("60000"),
        "take_profit": Decimal("64000"),
        "created_at": "2026-09-10T01:00:00.000Z",
        "opened_at": None,
        "closed_at": None,
        "close_price": None,
        "close_reason": None,
        "confidence": 80,
        "risk_reward": Decimal("3.00"),
        "rationale": None,
        "provider": "ollama",
        "model_name": "qwen3:8b",
        "strategy_name": None,
        "strategy_version": None,
        "result": None,
        "analysis_timestamp": "2026-09-10T00:59:00.000Z",
        "market_timestamp": "2026-09-10T00:00:00.000Z",
        "candle_close_price": Decimal("61050"),
    }
    defaults.update(overrides)
    return Signal(**defaults)


def _enabled_config(*, max_retries: int = 2, backoff: float = 0.0) -> TelegramConfig:
    return TelegramConfig(
        bot_token="123456789:TESTtoken",
        chat_id="-1001234567890",
        enabled=True,
        timeout=1.0,
        max_retries=max_retries,
        backoff=backoff,
    )


class _Response:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakePost:
    """Injects responses/exceptions into the client; never touches the network."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        if not self.responses:
            raise AssertionError("no stubbed response left for the transport")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ── A. Configuration ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTelegramConfig(unittest.TestCase):
    def test_env_names_follow_btcusdt_prefix(self):
        self.assertEqual(ENV_TELEGRAM_BOT_TOKEN, "BTCUSDT_TELEGRAM_BOT_TOKEN")
        self.assertEqual(ENV_TELEGRAM_CHAT_ID, "BTCUSDT_TELEGRAM_CHAT_ID")
        self.assertEqual(ENV_TELEGRAM_ENABLED, "BTCUSDT_TELEGRAM_ENABLED")

    def test_missing_env_disables_telegram(self):
        config = telegram_config_from_env({})
        self.assertFalse(config.enabled)
        validate_telegram_config(config)

    def test_disabled_accepts_empty_token_and_chat(self):
        config = validate_telegram_config(TelegramConfig(enabled=False))
        self.assertFalse(config.enabled)

    def test_enabled_true_requires_token(self):
        env = {ENV_TELEGRAM_ENABLED: "true", ENV_TELEGRAM_CHAT_ID: "-100123"}
        with self.assertRaisesRegex(TelegramConfigError, ENV_TELEGRAM_BOT_TOKEN):
            telegram_config_from_env(env)

    def test_enabled_true_requires_chat_id(self):
        env = {
            ENV_TELEGRAM_ENABLED: "true",
            ENV_TELEGRAM_BOT_TOKEN: "123:ABC",
        }
        with self.assertRaisesRegex(TelegramConfigError, ENV_TELEGRAM_CHAT_ID):
            telegram_config_from_env(env)

    def test_invalid_token_format_rejected(self):
        with self.assertRaises(TelegramConfigError):
            validate_telegram_config(
                TelegramConfig(
                    bot_token="no-colon-token",
                    chat_id="-100123",
                    enabled=True,
                )
            )

    def test_invalid_timeout_rejected(self):
        with self.assertRaises(TelegramConfigError):
            validate_telegram_config(
                TelegramConfig(bot_token="123:ABC", chat_id="c", enabled=True, timeout=0)
            )

    def test_non_boolean_enabled_rejected(self):
        env = {ENV_TELEGRAM_ENABLED: "maybe"}
        with self.assertRaises(TelegramConfigError):
            telegram_config_from_env(env)

    def test_config_reads_all_env_values(self):
        env = {
            ENV_TELEGRAM_ENABLED: "1",
            ENV_TELEGRAM_BOT_TOKEN: "123456789:AAAbbb",
            ENV_TELEGRAM_CHAT_ID: "-100123",
            "BTCUSDT_TELEGRAM_TIMEOUT": "5",
            "BTCUSDT_TELEGRAM_MAX_RETRIES": "3",
            "BTCUSDT_TELEGRAM_BACKOFF": "0.25",
        }
        config = telegram_config_from_env(env)
        self.assertTrue(config.enabled)
        self.assertEqual(config.bot_token, "123456789:AAAbbb")
        self.assertEqual(config.chat_id, "-100123")
        self.assertEqual(config.timeout, 5.0)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.backoff, 0.25)

    def test_config_error_never_leaks_token(self):
        with self.assertRaises(TelegramConfigError) as ctx:
            validate_telegram_config(
                TelegramConfig(
                    bot_token="SUPERSECRETBOTTOKEN",
                    chat_id="-100123",
                    enabled=True,
                )
            )
        self.assertNotIn("SUPERSECRETBOTTOKEN", str(ctx.exception))
        self.assertNotIn("SUPERSECRETBOTTOKEN", repr(ctx.exception))


# ── B. HTTP client ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTelegramClient(unittest.TestCase):
    def test_disabled_client_makes_no_request(self):
        post = _FakePost()
        client = TelegramClient(TelegramConfig(enabled=False), post=post)
        result = client.send_message("hi")
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.error)
        self.assertEqual(post.calls, [])

    def test_disabled_client_never_builds_token_url(self):
        client = TelegramClient(TelegramConfig(enabled=False))
        self.assertEqual(client.send_message("hi").ok, False)

    def test_empty_message_refused(self):
        post = _FakePost()
        client = TelegramClient(_enabled_config(), post=post)
        result = client.send_message("   ")
        self.assertFalse(result.ok)
        self.assertEqual(post.calls, [])

    def test_success_returns_message_id(self):
        post = _FakePost(_Response(200, OK_PAYLOAD))
        client = TelegramClient(_enabled_config(), post=post)
        result = client.send_message("hello")
        self.assertTrue(result.ok)
        self.assertEqual(result.message_id, 42)
        self.assertEqual(result.status_code, 200)
        _, payload, timeout = post.calls[0]
        self.assertEqual(payload["chat_id"], "-1001234567890")
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(timeout, 1.0)
        # The token is only in the URL, never in the payload.
        self.assertNotIn("TESTtoken", str(payload))

    def test_ok_false_body_is_api_error(self):
        post = _FakePost(_Response(200, {"ok": False, "description": "chat not found"}))
        client = TelegramClient(_enabled_config(), post=post)
        result = client.send_message("hello")
        self.assertFalse(result.ok)
        self.assertIn("chat not found", result.error)

    def test_http_400_fails_without_retry(self):
        post = _FakePost(_Response(400, {"ok": False, "description": "bad request"}))
        client = TelegramClient(_enabled_config(max_retries=2), post=post)
        result = client.send_message("hello")
        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(len(post.calls), 1)

    def test_http_401_fails_without_retry(self):
        post = _FakePost(_Response(401, {"ok": False}))
        client = TelegramClient(_enabled_config(max_retries=2), post=post)
        result = client.send_message("hello")
        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 401)
        self.assertEqual(len(post.calls), 1)

    def test_429_is_retried_then_succeeds(self):
        post = _FakePost(_Response(429, {"ok": False}), _Response(200, OK_PAYLOAD))
        client = TelegramClient(_enabled_config(max_retries=2), post=post)
        result = client.send_message("hello")
        self.assertTrue(result.ok)
        self.assertEqual(len(post.calls), 2)

    def test_500_retried_then_succeeds(self):
        post = _FakePost(_Response(500, {"ok": False}), _Response(200, OK_PAYLOAD))
        client = TelegramClient(_enabled_config(max_retries=1), post=post)
        result = client.send_message("hello")
        self.assertTrue(result.ok)
        self.assertEqual(len(post.calls), 2)

    def test_transient_exhausted_reports_error(self):
        post = _FakePost(
            _Response(500, {"ok": False}),
            _Response(503, {"ok": False}),
        )
        client = TelegramClient(_enabled_config(max_retries=1), post=post)
        result = client.send_message("hello")
        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(len(post.calls), 2)

    def test_timeout_is_retried(self):
        post = _FakePost(requests.Timeout("slow"), _Response(200, OK_PAYLOAD))
        client = TelegramClient(_enabled_config(max_retries=1), post=post)
        result = client.send_message("hello")
        self.assertTrue(result.ok)
        self.assertEqual(len(post.calls), 2)

    def test_connection_error_is_retried(self):
        post = _FakePost(
            requests.ConnectionError("refused"), _Response(200, OK_PAYLOAD)
        )
        client = TelegramClient(_enabled_config(max_retries=1), post=post)
        result = client.send_message("hello")
        self.assertTrue(result.ok)
        self.assertEqual(len(post.calls), 2)

    def test_malformed_json_fails_without_retry(self):
        post = _FakePost(_Response(200, ValueError("no json")))
        client = TelegramClient(_enabled_config(max_retries=2), post=post)
        result = client.send_message("hello")
        self.assertFalse(result.ok)
        self.assertIn("non-JSON", result.error)
        self.assertEqual(len(post.calls), 1)

    def test_unknown_transport_exception_fails_without_retry(self):
        post = _FakePost(RuntimeError("broken transport"))
        client = TelegramClient(_enabled_config(max_retries=2), post=post)
        result = client.send_message("hello")
        self.assertFalse(result.ok)
        self.assertEqual(len(post.calls), 1)

    def test_backoff_is_capped_and_applied_only_on_transient(self):
        import signal_engine.telegram as telegram_mod

        post = _FakePost(
            _Response(429, {"ok": False}), _Response(200, OK_PAYLOAD)
        )
        client = TelegramClient(
            _enabled_config(max_retries=1, backoff=1.0), post=post
        )
        with patch.object(telegram_mod.time, "sleep") as sleep:
            result = client.send_message("hello")
        self.assertTrue(result.ok)
        self.assertEqual(sleep.call_count, 1)
        delay = sleep.call_args[0][0]
        self.assertGreaterEqual(delay, 1.0)
        self.assertLessEqual(delay, 10.0)

    def test_error_message_never_contains_url_or_token(self):
        post = _FakePost(_Response(500, {"ok": False}))
        client = TelegramClient(_enabled_config(max_retries=0), post=post)
        result = client.send_message("hello")
        self.assertNotIn("TESTtoken", result.error)
        self.assertNotIn("api.telegram.org", result.error)


# ── C. Message formatting ────────────────────────────────────────────────────


@pytest.mark.unit
class TestMessageFormatting(unittest.TestCase):
    def test_format_price_uses_decimal(self):
        self.assertEqual(format_price(Decimal("61000")), "61,000.00")
        self.assertEqual(format_price(Decimal("64999.99999999999")), "65,000.00")
        self.assertEqual(format_price(None), "-")

    def test_format_timestamp_is_vietnam_time(self):
        self.assertEqual(
            format_timestamp("2026-09-10T00:00:00.000Z"), "10/09/2026 07:00 +07"
        )

    def test_format_signal_created(self):
        text = format_signal_created(_signal(), candle_ts=1_720_000_000_000)
        for expected in (
            "NEW SIGNAL",
            "PENDING_ENTRY",
            "BTCUSDT LONG",
            "Entry: 61,000.00",
            "SL: 60,000.00",
            "TP: 64,000.00",
            "Confidence: 80%",
            "Signal ID: 7",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("uptrend continuation", text)
        self.assertNotIn("qwen3:8b", text)

    def test_format_signal_created_short_uses_red(self):
        text = format_signal_created(_signal(direction="SHORT"))
        self.assertIn("🔴 BTCUSDT SHORT", text)

    def test_format_signal_opened(self):
        opened = _signal(status="OPEN", opened_at="2026-09-10T02:00:00.000Z")
        text = format_signal_opened(opened)
        self.assertIn("POSITION OPEN", text)
        self.assertIn("Opened at: 10/09/2026 09:00 +07", text)

    def test_format_signal_tp(self):
        tp = _signal(
            status="TP_HIT",
            opened_at="2026-09-10T02:00:00.000Z",
            closed_at="2026-09-10T03:00:00.000Z",
            close_price=Decimal("64000"),
            close_reason="TP",
        )
        text = format_signal_tp(tp)
        self.assertIn("TAKE PROFIT", text)
        self.assertIn("Close price: 64,000.00", text)
        self.assertIn("Closed at: 10/09/2026 10:00 +07", text)

    def test_format_signal_sl(self):
        sl = _signal(
            status="SL_HIT",
            opened_at="2026-09-10T02:00:00.000Z",
            closed_at="2026-09-10T03:00:00.000Z",
            close_price=Decimal("60000"),
            close_reason="SL",
        )
        text = format_signal_sl(sl)
        self.assertIn("STOP LOSS", text)
        self.assertIn("Close price: 60,000.00", text)

    def test_format_signal_ambiguous(self):
        opened = _signal(status="OPEN")
        text = format_signal_ambiguous(
            opened,
            candle_ts=1_720_000_000_000,
            reason="execution order unknown",
        )
        self.assertIn("AMBIGUOUS CANDLE", text)
        self.assertIn("Status: OPEN", text)
        self.assertIn("Reason: execution order unknown", text)
        self.assertIn("No automatic transition was performed.", text)

    def test_messages_never_contain_provider_secrets(self):
        for text in (
            format_signal_created(_signal()),
            format_signal_opened(_signal(status="OPEN")),
        ):
            self.assertNotIn("qwen3:8b", text)


# ── D. Notifier adapter ──────────────────────────────────────────────────────


class _FakeClient:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_message(self, text: str) -> TelegramSendResult:
        self.sent.append(text)
        return TelegramSendResult(ok=True, message_id=1)


@pytest.mark.unit
class TestTelegramNotifier(unittest.TestCase):
    def test_disabled_sends_nothing(self):
        client = _FakeClient()
        notifier = TelegramNotifier(TelegramConfig(enabled=False), client=client)
        result = notifier.notify_signal_created(_signal(), candle_ts=1)
        self.assertFalse(result.ok)
        self.assertEqual(client.sent, [])

    def test_notify_created_dispatches_formatted_message(self):
        client = _FakeClient()
        notifier = TelegramNotifier(_enabled_config(), client=client)
        result = notifier.notify_signal_created(_signal(), candle_ts=1_720_000_000_000)
        self.assertTrue(result.ok)
        self.assertEqual(len(client.sent), 1)
        self.assertIn("NEW SIGNAL", client.sent[0])

    def test_notify_opened_tp_sl_all_dispatch(self):
        client = _FakeClient()
        notifier = TelegramNotifier(_enabled_config(), client=client)
        opened = _signal(status="OPEN", opened_at="2026-09-10T02:00:00.000Z")
        tp = _signal(status="TP_HIT", close_price=Decimal("64000"), closed_at="2026-09-10T03:00:00.000Z")
        sl = _signal(status="SL_HIT", close_price=Decimal("60000"), closed_at="2026-09-10T04:00:00.000Z")
        notifier.notify_signal_opened(opened)
        notifier.notify_signal_tp(tp)
        notifier.notify_signal_sl(sl)
        self.assertEqual(len(client.sent), 3)
        self.assertIn("POSITION OPEN", client.sent[0])
        self.assertIn("TAKE PROFIT", client.sent[1])
        self.assertIn("STOP LOSS", client.sent[2])

    def test_ambiguous_same_candle_notified_once(self):
        client = _FakeClient()
        notifier = TelegramNotifier(_enabled_config(), client=client)
        opened = _signal(status="OPEN")
        first = notifier.notify_ambiguous(opened, "reason", candle_ts=1_720_000_000_000)
        second = notifier.notify_ambiguous(opened, "reason", candle_ts=1_720_000_000_000)
        self.assertTrue(first.ok)
        self.assertFalse(second.ok)
        self.assertIn("duplicate", second.error)
        self.assertEqual(len(client.sent), 1)

    def test_ambiguous_new_candle_notifies_again(self):
        client = _FakeClient()
        notifier = TelegramNotifier(_enabled_config(), client=client)
        opened = _signal(status="OPEN")
        notifier.notify_ambiguous(opened, "r", candle_ts=1_720_000_000_000)
        notifier.notify_ambiguous(opened, "r", candle_ts=1_720_000_000_100)
        self.assertEqual(len(client.sent), 2)

    def test_ambiguous_new_signal_id_can_notify_again(self):
        client = _FakeClient()
        notifier = TelegramNotifier(_enabled_config(), client=client)
        notifier.notify_ambiguous(_signal(id=1, status="OPEN"), "r", candle_ts=5)
        notifier.notify_ambiguous(_signal(id=2, status="OPEN"), "r", candle_ts=5)
        self.assertEqual(len(client.sent), 2)


class _FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, args: tuple, kwargs: dict) -> TelegramSendResult:
        self.calls.append((name, args, kwargs))
        return TelegramSendResult(ok=True, message_id=1)

    def notify_signal_created(self, signal, *, candle_ts=None):
        return self._record("created", (signal.id,), {"candle_ts": candle_ts})

    def notify_signal_opened(self, signal):
        return self._record("opened", (signal.id,), {})

    def notify_signal_tp(self, signal, *, pnl=None, balance=None):
        return self._record("tp", (signal.id,), {"pnl": pnl, "balance": balance})

    def notify_signal_sl(self, signal, *, pnl=None, balance=None):
        return self._record("sl", (signal.id,), {"pnl": pnl, "balance": balance})

    def notify_ambiguous(self, signal, reason="", *, candle_ts=None):
        return self._record("ambiguous", (signal.id, reason), {"candle_ts": candle_ts})


@pytest.mark.unit
class TestRefreshableTelegramNotifier(unittest.TestCase):
    def test_re_resolves_and_delegates_all_notify_methods(self):
        fake = _FakeNotifier()
        refreshable = RefreshableTelegramNotifier(lambda: fake)
        created = _signal()
        opened = _signal(status="OPEN")
        tp = _signal(status="TP_HIT")
        sl = _signal(status="SL_HIT")
        refreshable.notify_signal_created(created, candle_ts=5)
        refreshable.notify_signal_opened(opened)
        refreshable.notify_signal_tp(tp, pnl=Decimal("1.5"), balance=Decimal("1001.5"))
        refreshable.notify_signal_sl(sl, pnl=Decimal("-1"), balance=Decimal("999"))
        refreshable.notify_ambiguous(opened, "reason", candle_ts=7)
        self.assertEqual([name for name, _, _ in fake.calls],
                         ["created", "opened", "tp", "sl", "ambiguous"])
        self.assertEqual(fake.calls[0][1], (created.id,))
        self.assertEqual(fake.calls[0][2], {"candle_ts": 5})
        self.assertEqual(fake.calls[4][2], {"candle_ts": 7})

    def test_reflects_config_change_between_calls(self):
        # The resolver is re-invoked on every call: a disabled -> enabled
        # transition is picked up automatically (exactly the Settings-page
        # save that used to require a daemon restart).
        stages = iter([False, True])
        def resolve():
            return TelegramNotifier(TelegramConfig(
                bot_token="t", chat_id="c", enabled=next(stages)
            ), client=_FakeClient())
        refreshable = RefreshableTelegramNotifier(resolve)
        self.assertFalse(refreshable.notify_signal_created(_signal(), candle_ts=1).ok)
        result = refreshable.notify_signal_created(_signal(), candle_ts=2)
        self.assertTrue(result.ok)

    def test_resolver_none_treats_as_disabled(self):
        refreshable = RefreshableTelegramNotifier(None)
        result = refreshable.notify_signal_created(_signal(), candle_ts=1)
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.error or "")

    def test_resolver_raising_yields_failed_result_without_raising(self):
        def resolve():
            raise RuntimeError("boom")
        refreshable = RefreshableTelegramNotifier(resolve)
        result = refreshable.notify_signal_opened(_signal(status="OPEN"))
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.error or "")

    def test_message_truncated_to_telegram_limit(self):
        client = _FakeClient()
        notifier = TelegramNotifier(_enabled_config(), client=client)
        notifier.send_message("x" * (MAX_MESSAGE_CHARS + 500))
        self.assertLessEqual(len(client.sent[0]), MAX_MESSAGE_CHARS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
