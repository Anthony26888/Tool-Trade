"""Phase 11: Telegram notifications — a read-only output adapter.

Telegram is a notification-only, outbound channel. It never accepts commands,
never controls trading, never opens/closes signals, and is NEVER required for a
trading operation to succeed. Messages are emitted only AFTER the corresponding
successful SQLite state transition has been committed:

- ``PENDING_ENTRY`` created (``CREATED``)          -> ``notify_signal_created``
- ``PENDING_ENTRY -> OPEN`` (Entry touch)          -> ``notify_signal_opened``
- ``OPEN -> TP_HIT``                                -> ``notify_signal_tp``
- ``OPEN -> SL_HIT``                                -> ``notify_signal_sl``
- ambiguous candle (no transition)                  -> ``notify_ambiguous``

Because notifications fire only on the winning transition, a lost concurrent
poll (which observes the already-changed state) never re-emits. Recovery never
notifies: after a restart the system resumes monitoring silently. ``WAIT``
(no signal) produces no notification.

Safety
------
- Secrets live only in the environment (``BTCUSDT_TELEGRAM_BOT_TOKEN``,
  ``BTCUSDT_TELEGRAM_CHAT_ID``). They are never logged, never included in
  exceptions, and never echoed in messages.
- ``TelegramClient`` retries only TRANSIENT failures (timeout, connection,
  HTTP 429/5xx) with a bounded budget; auth/config errors (400/401/403/404)
  and malformed responses fail fast.
- A Telegram failure never fails the trading operation: domain code persists
  the transition first and logs the notification failure as a warning only.
- When disabled (``BTCUSDT_TELEGRAM_ENABLED=false``, the default) no network
  request is made and nothing crashes.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import requests

from database.models import DIRECTION_LONG, Signal

logger = logging.getLogger(__name__)

# https://core.telegram.org/bots/api#sendmessage
TELEGRAM_API_BASE_URL = "https://api.telegram.org"

#: Vietnam standard time (UTC+7); the country has no DST, so a fixed offset is safe.
_VIETNAM_TZ = timezone(timedelta(hours=7))

#: Telegram hard cap on a single message; ``send_message`` truncates safely.
MAX_MESSAGE_CHARS = 4096

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 0.5

#: Only transient failures are retried: timeout/connection errors and the
#: Telegram-side 429 (rate limit) / 5xx (server) statuses.
_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)
_BACKOFF_MULTIPLIER = 2.0
_MAX_BACKOFF_SECONDS = 10.0

# Environment variables. Values are documented in ``.env.example``; nothing is
# ever hard-coded into trading logic.
ENV_TELEGRAM_ENABLED = "BTCUSDT_TELEGRAM_ENABLED"
ENV_TELEGRAM_BOT_TOKEN = "BTCUSDT_TELEGRAM_BOT_TOKEN"
ENV_TELEGRAM_CHAT_ID = "BTCUSDT_TELEGRAM_CHAT_ID"
ENV_TELEGRAM_TIMEOUT = "BTCUSDT_TELEGRAM_TIMEOUT"
ENV_TELEGRAM_MAX_RETRIES = "BTCUSDT_TELEGRAM_MAX_RETRIES"
ENV_TELEGRAM_BACKOFF = "BTCUSDT_TELEGRAM_BACKOFF"


class TelegramError(Exception):
    """Base class for Telegram notification errors."""


class TelegramConfigError(ValueError, TelegramError):
    """Invalid Telegram configuration.

    Messages NEVER include the bot token so a misconfiguration report cannot
    leak credentials into logs or tests.
    """


@dataclass(frozen=True)
class TelegramConfig:
    """Immutable Telegram notification configuration (like ``LLMConfig``).

    ``bot_token`` / ``chat_id`` are read from the environment; ``enabled``
    defaults to ``False`` so the very first deployment silently does nothing.
    ``timeout`` bounds a single HTTP request; ``max_retries`` bounds the
    transient retry budget and ``backoff`` seeds its exponential delay.
    """

    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff: float = DEFAULT_BACKOFF_SECONDS


@dataclass(frozen=True)
class TelegramSendResult:
    """Outcome of one ``sendMessage`` attempt.

    ``ok`` is never raised/persisted as a trading failure: a failed send only
    produces a warning log (see :mod:`signal_engine.telegram`). ``error`` is a
    sanitized description that never contains the token, chat id, or API URL.
    """

    ok: bool
    message_id: int | None = None
    status_code: int | None = None
    error: str | None = None


class TelegramClient:
    """Low-level HTTP client for ``/bot<token>/sendMessage``.

    ``post`` is injectable so tests never touch the network; it must behave
    like ``requests.post(url, json=..., timeout=...)`` and return an object
    with ``status_code`` and ``json()``.
    """

    def __init__(
        self,
        config: TelegramConfig,
        *,
        post: Any | None = None,
    ) -> None:
        self.config = config
        self._post = requests.post if post is None else post

    # -- HTTP rendering -------------------------------------------------------

    def _endpoint(self) -> str:
        # The token travels in the URL only; every downstream log/message is
        # built from sanitized fragments so this string is never echoed.
        return f"{TELEGRAM_API_BASE_URL}/bot{self.config.bot_token}/sendMessage"

    def send_message(self, text: str | None) -> TelegramSendResult:
        """Send ``text``; return a sanitized result instead of raising.

        ``post`` failures are classified: timeout/connection and the Telegram
        transient HTTP statuses are retried within the ``max_retries`` budget;
        everything else fails fast. The URL, token, and response body are never
        part of the returned error.
        """
        if not self.config.enabled:
            return TelegramSendResult(ok=False, error="telegram disabled")
        if text is None or not str(text).strip():
            return TelegramSendResult(ok=False, error="refusing to send an empty message")

        url = self._endpoint()
        payload = {"chat_id": self.config.chat_id, "text": str(text)}
        attempts = self.config.max_retries + 1
        last: TelegramSendResult | None = None

        for attempt in range(attempts):
            try:
                response = self._post(url, json=payload, timeout=self.config.timeout)
            except (requests.Timeout, requests.ConnectionError):
                last = TelegramSendResult(
                    ok=False,
                    error="connection or timeout failure sending the Telegram message",
                )
                logger.warning(
                    "Telegram request failed (attempt %d/%d); retrying",
                    attempt + 1,
                    attempts,
                )
                self._backoff(attempt)
                continue
            except Exception:  # pragma: no cover - defensive transport boundary
                return TelegramSendResult(
                    ok=False, error="request to the Telegram API failed"
                )

            status = int(getattr(response, "status_code", 200))
            if status in _RETRYABLE_STATUS_CODES:
                last = TelegramSendResult(
                    ok=False,
                    status_code=status,
                    error=f"Telegram transient HTTP {status}",
                )
                logger.warning(
                    "Telegram retryable HTTP %s (attempt %d/%d); retrying",
                    status,
                    attempt + 1,
                    attempts,
                )
                self._backoff(attempt)
                continue
            if status != 200:
                return TelegramSendResult(
                    ok=False,
                    status_code=status,
                    error=f"Telegram API returned HTTP {status}",
                )

            try:
                body = response.json()
            except ValueError:
                return TelegramSendResult(
                    ok=False,
                    status_code=status,
                    error="Telegram returned a non-JSON response",
                )
            if not isinstance(body, dict) or body.get("ok") is not True:
                description = "unknown Telegram error"
                if isinstance(body, dict) and body.get("description"):
                    description = str(body["description"])[:100]
                return TelegramSendResult(
                    ok=False,
                    status_code=status,
                    error=f"Telegram API error: {description}",
                )

            message_id = None
            result = body.get("result")
            if isinstance(result, dict) and result.get("message_id") is not None:
                try:
                    message_id = int(result["message_id"])
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    message_id = None
            return TelegramSendResult(ok=True, message_id=message_id, status_code=status)

        assert last is not None  # pragma: no cover - loop always yields a result
        return last

    def _backoff(self, attempt: int) -> None:
        if self.config.backoff <= 0:
            return
        delay = min(self.config.backoff * (_BACKOFF_MULTIPLIER**attempt), _MAX_BACKOFF_SECONDS)
        time.sleep(delay)


def validate_telegram_config(config: TelegramConfig) -> TelegramConfig:
    """Validate a :class:`TelegramConfig` (raises ``TelegramConfigError``).

    Requirement-level checks: an enabled config requires a well-formed bot
    token and a chat id; ``timeout`` must be positive. Nothing about the token
    value is ever included in an error message.
    """
    if not isinstance(config, TelegramConfig):
        raise TelegramConfigError("telegram config must be a TelegramConfig")
    if not isinstance(config.enabled, bool):
        raise TelegramConfigError("telegram enabled must be a boolean")
    if not isinstance(config.timeout, (int, float)) or config.timeout <= 0:
        raise TelegramConfigError("telegram timeout must be a positive number of seconds")
    if not isinstance(config.max_retries, int) or config.max_retries < 0:
        raise TelegramConfigError("telegram max_retries must be a non-negative integer")
    if not isinstance(config.backoff, (int, float)) or config.backoff < 0:
        raise TelegramConfigError("telegram backoff must be a non-negative number")

    if not config.enabled:
        return config

    token = str(config.bot_token or "").strip()
    chat_id = str(config.chat_id or "").strip()
    if not token:
        raise TelegramConfigError(
            f"{ENV_TELEGRAM_BOT_TOKEN} is not set but Telegram is enabled"
        )
    if ":" not in token:
        raise TelegramConfigError(
            f"{ENV_TELEGRAM_BOT_TOKEN} is invalid (expected '<bot-id>:<auth-token>')"
        )
    if not chat_id:
        raise TelegramConfigError(
            f"{ENV_TELEGRAM_CHAT_ID} is not set but Telegram is enabled"
        )
    return config


def _env_bool(env: dict, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise TelegramConfigError(f"{name} must be a boolean (true/false/1/0)")


def _env_float(env: dict, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise TelegramConfigError(f"{name} must be a number") from exc


def _env_int(env: dict, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise TelegramConfigError(f"{name} must be an integer") from exc


def telegram_config_from_env(env: dict | None = None) -> TelegramConfig:
    """Build a validated :class:`TelegramConfig` from environment variables."""
    env = dict(os.environ) if env is None else env
    config = TelegramConfig(
        bot_token=str(env.get(ENV_TELEGRAM_BOT_TOKEN, "") or ""),
        chat_id=str(env.get(ENV_TELEGRAM_CHAT_ID, "") or "").strip(),
        enabled=_env_bool(env, ENV_TELEGRAM_ENABLED, default=False),
        timeout=_env_float(env, ENV_TELEGRAM_TIMEOUT, default=DEFAULT_TIMEOUT_SECONDS),
        max_retries=_env_int(env, ENV_TELEGRAM_MAX_RETRIES, default=DEFAULT_MAX_RETRIES),
        backoff=_env_float(env, ENV_TELEGRAM_BACKOFF, default=DEFAULT_BACKOFF_SECONDS),
    )
    return validate_telegram_config(config)


# -- Message formatting --------------------------------------------------------


def format_price(value: Decimal | None) -> str:
    """Render a ``Decimal`` price as ``65,000.00``.

    ``Decimal`` handling avoids binary-float artifacts (a float such as
    ``64999.99999999999`` formats as its true rounded value, never an invented
    precision).
    """
    if value is None:
        return "-"
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    return f"{dec:,.2f}"


def format_amount(value: Decimal | None) -> str:
    """Render a signed money/PnL amount as ``+12.34`` / ``-12.34`` / ``-``.

    Used for demo PnL in TP/SL notifications; a zero is rendered as ``0.00``
    (no invented plus sign).
    """
    if value is None:
        return "-"
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    if dec == 0:
        return "0.00"
    sign = "+" if dec > 0 else "-"
    return f"{sign}{abs(dec):,.2f}"


def format_timestamp(value: str | None) -> str:
    """Render an ISO-8601 UTC timestamp as ``DD/MM/YYYY HH:MM +07`` (Vietnam time)."""
    if value is None:
        return "-"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return dt.astimezone(_VIETNAM_TZ).strftime("%d/%m/%Y %H:%M +07")


def format_epoch_ms(value: int | None) -> str:
    """Render an epoch-millisecond candle timestamp as ``DD/MM/YYYY HH:MM +07``."""
    if value is None:
        return "-"
    dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    return dt.astimezone(_VIETNAM_TZ).strftime("%d/%m/%Y %H:%M +07")


def _truncate(text: str) -> str:
    text = str(text or "")
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[: MAX_MESSAGE_CHARS - 1] + "…"


def _tag(signal: Signal) -> str:
    emoji = "🟢" if signal.direction == DIRECTION_LONG else "🔴"
    return f"{emoji} {signal.symbol} {signal.direction}"


def _levels(signal: Signal) -> list[str]:
    return [
        f"Entry: {format_price(signal.entry)}",
        f"SL: {format_price(signal.stop_loss)}",
        f"TP: {format_price(signal.take_profit)}",
    ]


def format_signal_created(signal: Signal, *, candle_ts: int | None = None) -> str:
    """Readable PENDING_ENTRY-created message (nothing sensitive, no raw AI text)."""
    lines = [_tag(signal) + " — NEW SIGNAL", "", f"Status: {signal.status}"]
    lines += _levels(signal)
    if signal.confidence is not None:
        lines.append(f"Confidence: {signal.confidence}%")
    if signal.risk_reward is not None:
        lines.append(f"Risk/Reward: {format_price(signal.risk_reward)}")
    lines += ["", f"Signal ID: {signal.id}", f"Candle: {format_epoch_ms(candle_ts)}"]
    if signal.created_at:
        lines.append(f"Created: {format_timestamp(signal.created_at)}")
    return "\n".join(lines)


def format_signal_opened(signal: Signal) -> str:
    lines = [_tag(signal) + " — POSITION OPEN", "", f"Status: {signal.status}"]
    lines += _levels(signal)
    lines += ["", f"Signal ID: {signal.id}"]
    if signal.opened_at:
        lines.append(f"Opened at: {format_timestamp(signal.opened_at)}")
    return "\n".join(lines)


def format_signal_tp(
    signal: Signal, *, pnl: Decimal | None = None, balance: Decimal | None = None
) -> str:
    lines = [f"🟢 {signal.symbol} {signal.direction} — TAKE PROFIT", "", "Status: TP_HIT"]
    lines += [
        f"Entry: {format_price(signal.entry)}",
        f"TP: {format_price(signal.take_profit)}",
        f"Close price: {format_price(signal.close_price)}",
    ]
    if pnl is not None:
        lines += ["", f"PnL: {format_amount(pnl)} USDT"]
    if balance is not None:
        lines.append(f"Balance: {format_amount(balance)} USDT")
    lines += ["", f"Signal ID: {signal.id}"]
    if signal.closed_at:
        lines.append(f"Closed at: {format_timestamp(signal.closed_at)}")
    return "\n".join(lines)


def format_signal_sl(
    signal: Signal, *, pnl: Decimal | None = None, balance: Decimal | None = None
) -> str:
    lines = [f"🔴 {signal.symbol} {signal.direction} — STOP LOSS", "", "Status: SL_HIT"]
    lines += [
        f"Entry: {format_price(signal.entry)}",
        f"SL: {format_price(signal.stop_loss)}",
        f"Close price: {format_price(signal.close_price)}",
    ]
    if pnl is not None:
        lines += ["", f"PnL: {format_amount(pnl)} USDT"]
    if balance is not None:
        lines.append(f"Balance: {format_amount(balance)} USDT")
    lines += ["", f"Signal ID: {signal.id}"]
    if signal.closed_at:
        lines.append(f"Closed at: {format_timestamp(signal.closed_at)}")
    return "\n".join(lines)


def format_signal_ambiguous(
    signal: Signal, *, candle_ts: int | None = None, reason: str = ""
) -> str:
    """Readable warning for an ambiguous candle (no transition was performed)."""
    lines = [f"⚠️ {signal.symbol} {signal.direction} — AMBIGUOUS CANDLE", ""]
    lines += [
        f"Signal ID: {signal.id}",
        f"Status: {signal.status}",
        f"Candle: {format_epoch_ms(candle_ts)}",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    lines += ["", "No automatic transition was performed."]
    return "\n".join(lines)


# -- Domain adapter ------------------------------------------------------------


class TelegramNotifier:
    """Optional domain notification adapter for the signal system.

    Injected (default ``None``) into the scheduler and monitor. All ``notify_*``
    methods return :class:`TelegramSendResult`: they never raise and are never
    allowed to fail the underlying trading operation. Disabled mode performs no
    network I/O at all.
    """

    def __init__(self, config: TelegramConfig, client: TelegramClient | None = None) -> None:
        self.config = config
        self.client = client if client is not None else TelegramClient(config)
        #: Heuristic guard against re-sending the SAME ambiguous candle on
        #: repeated monitor polls (ambiguity is not a state transition, so
        #: transition-semantics alone cannot deduplicate it).
        self._last_ambiguous_key: tuple[Any, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def send_message(self, text: str) -> TelegramSendResult:
        """Send one (truncated) message; never raises.

        The transport boundary catches everything the client itself cannot, so
        the scheduler/monitor callers stay simple and trading stays safe.
        """
        if not self.config.enabled:
            return TelegramSendResult(ok=False, error="telegram disabled")
        try:
            result = self.client.send_message(_truncate(text))
        except Exception:  # pragma: no cover - defensive transport boundary
            logger.warning("Telegram send failed without a result: %s", "transport failure")
            return TelegramSendResult(ok=False, error="Telegram send failed unexpectedly")
        if not result.ok:
            logger.warning("Telegram notification could not be sent: %s", result.error)
        return result

    def notify_signal_created(self, signal: Signal, *, candle_ts: int | None = None) -> TelegramSendResult:
        return self.send_message(format_signal_created(signal, candle_ts=candle_ts))

    def notify_signal_opened(self, signal: Signal) -> TelegramSendResult:
        return self.send_message(format_signal_opened(signal))

    def notify_signal_tp(
        self, signal: Signal, *, pnl: Decimal | None = None, balance: Decimal | None = None
    ) -> TelegramSendResult:
        return self.send_message(format_signal_tp(signal, pnl=pnl, balance=balance))

    def notify_signal_sl(
        self, signal: Signal, *, pnl: Decimal | None = None, balance: Decimal | None = None
    ) -> TelegramSendResult:
        return self.send_message(format_signal_sl(signal, pnl=pnl, balance=balance))

    def notify_ambiguous(
        self, signal: Signal, reason: str = "", *, candle_ts: int | None = None
    ) -> TelegramSendResult:
        if not self.config.enabled:
            return TelegramSendResult(ok=False, error="telegram disabled")
        key = (signal.id, candle_ts)
        if self._last_ambiguous_key == key:
            return TelegramSendResult(ok=False, error="duplicate ambiguous notification suppressed")
        self._last_ambiguous_key = key
        return self.send_message(format_signal_ambiguous(signal, candle_ts=candle_ts, reason=reason))


def telegram_notifier_from_env(env: dict | None = None) -> TelegramNotifier:
    """Build a notifier from the environment, ready for optional injection."""
    return TelegramNotifier(telegram_config_from_env(env))


class RefreshableTelegramNotifier:
    """Duck-typed notifier that re-resolves its target on every call.

    Takes a ``resolver`` callable returning the object the notification is
    delegated to (typically ``ConfigService.resolve_telegram_notifier``). This
    lets the daemon pick up a Telegram Settings-page save WITHOUT a restart or
    any shared mutable state, while keeping the exact same duck-typed contract
    as :class:`TelegramNotifier`: calls never raise and never fail the trading
    operation. A resolver returning ``None`` or raising is treated as
    "unavailable" and yields a disabled send result.
    """

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    def _current(self) -> Any:
        if self._resolver is None:
            return None
        try:
            return self._resolver()
        except Exception:  # pragma: no cover - defensive boundary
            logger.warning("Telegram notifier resolver failed; treating as unavailable")
            return None

    @staticmethod
    def _delegate(current: Any, name: str, *args: Any, **kwargs: Any) -> TelegramSendResult:
        method = getattr(current, name, None) if current is not None else None
        if method is None:
            return TelegramSendResult(ok=False, error="telegram disabled")
        return method(*args, **kwargs)

    def notify_signal_created(self, signal: Signal, *, candle_ts: int | None = None) -> TelegramSendResult:
        return self._delegate(
            self._current(), "notify_signal_created", signal, candle_ts=candle_ts
        )

    def notify_signal_opened(self, signal: Signal) -> TelegramSendResult:
        return self._delegate(self._current(), "notify_signal_opened", signal)

    def notify_signal_tp(
        self, signal: Signal, *, pnl: Decimal | None = None, balance: Decimal | None = None
    ) -> TelegramSendResult:
        return self._delegate(
            self._current(), "notify_signal_tp", signal, pnl=pnl, balance=balance
        )

    def notify_signal_sl(
        self, signal: Signal, *, pnl: Decimal | None = None, balance: Decimal | None = None
    ) -> TelegramSendResult:
        return self._delegate(
            self._current(), "notify_signal_sl", signal, pnl=pnl, balance=balance
        )

    def notify_ambiguous(
        self, signal: Signal, reason: str = "", *, candle_ts: int | None = None
    ) -> TelegramSendResult:
        return self._delegate(
            self._current(), "notify_ambiguous", signal, reason, candle_ts=candle_ts
        )


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENV_TELEGRAM_BACKOFF",
    "ENV_TELEGRAM_BOT_TOKEN",
    "ENV_TELEGRAM_CHAT_ID",
    "ENV_TELEGRAM_ENABLED",
    "ENV_TELEGRAM_MAX_RETRIES",
    "ENV_TELEGRAM_TIMEOUT",
    "MAX_MESSAGE_CHARS",
    "RefreshableTelegramNotifier",
    "TelegramClient",
    "TelegramConfig",
    "TelegramConfigError",
    "TelegramError",
    "TelegramNotifier",
    "TelegramSendResult",
    "format_amount",
    "format_epoch_ms",
    "format_price",
    "format_signal_ambiguous",
    "format_signal_created",
    "format_signal_opened",
    "format_signal_sl",
    "format_signal_tp",
    "format_timestamp",
    "telegram_config_from_env",
    "telegram_notifier_from_env",
    "validate_telegram_config",
]
