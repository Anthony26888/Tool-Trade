"""Phase 15: Settings & AI Provider configuration service.

This module is the single configuration boundary behind the Web Dashboard and
the daemon itself (AGENTS.md section 8). It follows the project philosophy of
reusing existing building blocks:

* the persisted store is SQLite (``app_settings`` + ``config_audit`` tables via
  :class:`database.database.ConfigRepository`);
* AI provider resolution reuses ``signal_engine.llm`` (``LLMConfig`` +
  ``build_llm_client``) and the shared TradingAgents provider layer — no second
  LLM implementation is created here;
* demo settings validation reuses ``demo.account.DemoConfig`` /
  ``validate_config``;
* Telegram testing reuses ``signal_engine.telegram.TelegramClient``.

Configuration precedence (highest first): **database settings > environment
variables > built-in defaults**. When no AI-provider setting has been saved,
the engine behaves exactly as before (env-driven). Values saved in the
database only ever affect FUTURE AI analysis/signals; existing signals,
positions, trades, entry/SL/TP, and PnL are never modified.

Security contract (AGENTS.md section 22 / Phase 15 spec section 7)
------------------------------------------------------------------
* API keys and Telegram bot tokens are NEVER written to SQLite, sent to the
  frontend, logged, echoed in errors, included in statistics, audit entries,
  or Telegram messages. They live in a 0600-permission JSON secrets file
  (``BTCUSDT_SECRETS_FILE``, default ``data/btcusdt_secrets.json``).
* The Settings API only ever returns ``configured`` flags plus masked samples.
* A config change never touches trading state: while a signal is active the AI
  provider change is staged as ``CONFIG_PENDING`` and applied the moment the
  system is idle; the Demo settings are refused while a signal is active.

Safe apply (AGENTS.md section 3, Phase 15 spec section 4)
---------------------------------------------------------
``update_ai_provider`` always stages the new value under ``ai_provider_pending``
and promotes it to the live ``ai_provider`` key only when no active signal
(PENDING_ENTRY or OPEN) exists. ``apply_pending_if_idle`` is called by the
runtime before each scheduler tick, so a change saved mid-trade becomes active
exactly when the position closes — never earlier, never for the running trade.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from binance.market_data import DEFAULT_SYMBOL
from database.database import (
    ConfigRepository,
    Database,
    DemoRepository,
    SignalRepository,
    iso_utc_now,
)
from demo.account import DemoConfig, DemoConfigError, validate_config

from .llm import LLMConfig, llm_config_from_env
from .multiagent import ENV_ANALYSIS_MODE

logger = logging.getLogger(__name__)

# -- Settings keys --------------------------------------------------------------

KEY_AI_PROVIDER = "ai_provider"
KEY_AI_PENDING = "ai_provider_pending"
KEY_DEMO = "demo"
KEY_TELEGRAM = "telegram"
KEY_RUNTIME = "runtime"
KEY_SYMBOL = "symbol"

#: Symbols selectable from the Settings page. Exactly one symbol is analysed at
#: a time; switching only affects the NEXT analysis once the system is idle.
SUPPORTED_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "XAUUSDT")

# -- Secret-file keys (never stored in SQLite) ------------------------------------

SECRET_API_KEY = "ai_provider.api_key"
SECRET_BOT_TOKEN = "telegram.bot_token"

ENV_SECRETS_FILE = "BTCUSDT_SECRETS_FILE"
DEFAULT_SECRETS_FILE = os.path.join("data", "btcusdt_secrets.json")

ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
DEFAULT_OLLAMA_SERVER = "http://localhost:11434"

#: Defaults for a fresh (never-saved) AI provider settings block. Production
#: default: Ollama + qwen3:4b with a configurable base URL.
DEFAULT_AI_PROVIDER = "ollama"
DEFAULT_AI_PRESET = "ollama"
DEFAULT_AI_MODEL = "qwen3:4b"
DEFAULT_AI_TIMEOUT = 120.0
DEFAULT_AI_TEMPERATURE = 0.2
DEFAULT_AI_MAX_TOKENS = 2048
#: Analysis pipelines the engine may use. ``single`` is the one-call structured
#: analysis; ``multi`` runs the Phase 5 five-agent committee debate.
ANALYSIS_MODES = ("single", "multi")
DEFAULT_AI_ANALYSIS_MODE = ANALYSIS_MODES[0]

#: API-page presets map to a supported TradingAgents provider id.
API_PRESET_PROVIDER: dict[str, str] = {
    "openai": "openai",
    "deepseek": "deepseek",
    "openai_compatible": "openai_compatible",
    "custom": "openai_compatible",
}

#: Default base URLs consulted by the API test-connection when the user leaves
#: the URL blank for a hosted preset.
API_PRESET_DEFAULT_BASE_URL: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}

_TEMPERATURE_MIN = 0.0
_TEMPERATURE_MAX = 5.0
_HTTP_TEST_TIMEOUT = 8.0


class SettingsError(Exception):
    """Base class for Phase 15 Settings errors."""


class SettingsValidationError(ValueError, SettingsError):
    """A submitted setting failed validation (HTTP 400)."""


class SettingsLockedError(SettingsError):
    """The setting cannot be changed while a signal is active (HTTP 409)."""


class SettingsTestError(SettingsError):
    """A live connectivity test failed without leaking secrets (HTTP 502)."""


def mask_secret(value: str) -> str:
    """Mask a credential for display: first 4 + * + last 4 (never plaintext)."""
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _ollama_server_base(raw: str, env: dict[str, Any]) -> str:
    """Normalize a user-supplied Ollama host to the server root (no ``/v1``)."""
    url = (raw or "").strip() or env.get(ENV_OLLAMA_BASE_URL, "") or DEFAULT_OLLAMA_SERVER
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def _ollama_chat_base(raw: str, env: dict[str, Any]) -> str:
    """Normalize a user-supplied Ollama host to the OpenAI-compatible ``/v1`` endpoint."""
    url = (raw or "").strip() or env.get(ENV_OLLAMA_BASE_URL, "") or DEFAULT_OLLAMA_SERVER
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _ollama_tags_url(raw: str, env: dict[str, Any]) -> str:
    return _ollama_server_base(raw, env) + "/api/tags"


# -- Secret store ----------------------------------------------------------------


class SecretStore:
    """0600-permission JSON secrets file (API keys / Telegram tokens).

    Values are cached per instance; writes are atomic (truncate + write through
    a single file descriptor opened with ``0o600``). This file is never
    committed, logged, or served to the frontend.
    """

    def __init__(self, path: str | None = None, *, env: dict[str, Any] | None = None) -> None:
        self.env = env if env is not None else os.environ
        self.path = os.path.abspath(
            path or self.env.get(ENV_SECRETS_FILE) or DEFAULT_SECRETS_FILE
        )
        self._cache: dict[str, str] | None = None
        self._cache_mtime: float | None = None

    def _refresh_if_changed(self) -> None:
        """Reload the secrets file when it changed on disk (writes).

        The web dashboard and the scheduler daemon run as separate processes
        and each owns its own :class:`SecretStore`; ``set`` on one instance
        must be observable by the other on the next ``get`` (e.g. an API key
        saved through the Settings page reaching the analysis pipeline without
        a container restart). The cached value is only served while the file's
        mtime is unchanged, so a stale in-memory copy never leaks through.
        """
        try:
            stat = os.stat(self.path)
            mtime = stat.st_mtime
        except OSError:
            mtime = None
        if self._cache is not None and mtime == self._cache_mtime:
            return
        self._cache_mtime = mtime
        if mtime is None:
            self._cache = {}
            return
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            self._cache = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            logger.warning("[Config] could not read secrets file %s", self.path)
            self._cache = {}

    def configured(self, key: str) -> bool:
        return bool(self.get(key))

    def get(self, key: str) -> str | None:
        self._refresh_if_changed()
        value = self._cache.get(key)
        return value if value else None

    def set(self, key: str, value: str) -> None:
        if not key or not value:
            return
        self._refresh_if_changed()
        data = dict(self._cache or {})
        data[key] = value
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
        finally:
            self._cache = data
            try:
                self._cache_mtime = os.stat(self.path).st_mtime
            except OSError:  # pragma: no cover - file was just written
                self._cache_mtime = None


# -- AI provider settings ----------------------------------------------------------


@dataclass(frozen=True)
class AIProviderSettings:
    """Validated AI provider settings submitted from the Settings page.

    ``provider`` selects the radio: ``ollama`` or ``api`` (OpenAI-compatible).
    ``preset`` selects the API preset (openai / deepseek / openai_compatible /
    custom). The API key never lives on this object — it is handled by
    :class:`SecretStore` and only observable as ``api_key_configured``.
    """

    provider: str = DEFAULT_AI_PROVIDER
    preset: str = DEFAULT_AI_PRESET
    base_url: str = ""
    model: str = DEFAULT_AI_MODEL
    timeout: float = DEFAULT_AI_TIMEOUT
    temperature: float = DEFAULT_AI_TEMPERATURE
    max_tokens: int = DEFAULT_AI_MAX_TOKENS
    analysis_mode: str = DEFAULT_AI_ANALYSIS_MODE
    api_key_configured: bool = False


def _required_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if value is None:
        raise SettingsValidationError(f"{field} is required")
    text = str(value).strip()
    if not text:
        raise SettingsValidationError(f"{field} is required")
    return text


def _optional_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    return str(value).strip() if value is not None else ""


def _float_setting(payload: dict[str, Any], field: str, default: float) -> float:
    value = payload.get(field)
    if value is None or str(value).strip() == "":
        return default
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise SettingsValidationError(f"{field} must be a number") from exc
    if not number == number:  # NaN
        raise SettingsValidationError(f"{field} must be a finite number")
    return number


def _int_setting(payload: dict[str, Any], field: str, default: int) -> int:
    value = payload.get(field)
    if value is None or str(value).strip() == "":
        return default
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise SettingsValidationError(f"{field} must be an integer") from exc
    return number


def validate_ai_provider(payload: dict[str, Any]) -> AIProviderSettings:
    """Validate a Settings-page AI provider payload.

    Raises :class:`SettingsValidationError` for missing/invalid fields. No
    network request and no credential check happens here.
    """
    provider = _required_str(payload, "provider").lower()
    if provider not in ("ollama", "api"):
        raise SettingsValidationError("provider must be 'ollama' or 'api'")
    preset = _optional_str(payload, "preset").lower() or ("ollama" if provider == "ollama" else "custom")
    if provider == "ollama":
        preset = "ollama"
    elif preset not in API_PRESET_PROVIDER:
        raise SettingsValidationError(
            f"preset must be one of {', '.join(sorted(API_PRESET_PROVIDER))}"
        )

    model = _required_str(payload, "model")
    if provider == "api" and preset in ("custom", "openai_compatible"):
        base_url = _required_str(payload, "base_url")
    else:
        base_url = _optional_str(payload, "base_url")
    if "://" not in base_url and base_url:
        raise SettingsValidationError("base_url must be a full URL (e.g. http://host:11434)")

    timeout = _float_setting(payload, "timeout", DEFAULT_AI_TIMEOUT)
    if timeout <= 0:
        raise SettingsValidationError("timeout must be positive")
    temperature = _float_setting(payload, "temperature", DEFAULT_AI_TEMPERATURE)
    if temperature < _TEMPERATURE_MIN or temperature > _TEMPERATURE_MAX:
        raise SettingsValidationError(
            f"temperature must be between {_TEMPERATURE_MIN} and {_TEMPERATURE_MAX}"
        )
    max_tokens = _int_setting(payload, "max_tokens", DEFAULT_AI_MAX_TOKENS)
    if max_tokens <= 0:
        raise SettingsValidationError("max_tokens must be positive")

    analysis_mode = _optional_str(payload, "analysis_mode")
    analysis_mode = analysis_mode or DEFAULT_AI_ANALYSIS_MODE
    if analysis_mode not in ANALYSIS_MODES:
        raise SettingsValidationError(
            f"analysis_mode must be one of {', '.join(ANALYSIS_MODES)}"
        )

    return AIProviderSettings(
        provider=provider,
        preset=preset,
        base_url=base_url,
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        analysis_mode=analysis_mode,
        api_key_configured=bool(payload.get("api_key_configured")),
    )


# -- Market symbol setting --------------------------------------------------------


def validate_symbol_setting(value: Any) -> str:
    """Normalize + validate a Settings-page symbol selection.

    Raises :class:`SettingsValidationError` when the value is missing or not in
    :data:`SUPPORTED_SYMBOLS`.
    """
    if not isinstance(value, str) or not value.strip():
        raise SettingsValidationError("symbol must be a non-empty string")
    symbol = value.strip().upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise SettingsValidationError(
            f"symbol must be one of {', '.join(SUPPORTED_SYMBOLS)}"
        )
    return symbol


# -- Demo settings -----------------------------------------------------------------


def _demo_decimal(payload: dict[str, Any], field: str) -> Decimal:
    value = payload.get(field)
    if value is None or str(value).strip() == "":
        raise SettingsValidationError(f"{field} is required")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise SettingsValidationError(f"{field} must be a decimal number") from exc
    if not number.is_finite():
        raise SettingsValidationError(f"{field} must be a finite number")
    return number


def demo_config_from_payload(payload: dict[str, Any]) -> DemoConfig:
    """Build a validated :class:`DemoConfig` from a Settings-page payload."""
    try:
        config = DemoConfig(
            initial_balance=_demo_decimal(payload, "initial_balance"),
            margin_per_trade=_demo_decimal(payload, "margin_per_trade"),
            leverage=_int_setting(payload, "leverage", 0),
            risk_percent=_demo_decimal(payload, "risk_percent"),
            fee_rate=_demo_decimal(payload, "fee_rate"),
        )
        validate_config(config)
    except (DemoConfigError, SettingsValidationError) as exc:
        raise SettingsValidationError(str(exc)) from exc
    return config


# -- Configuration service ------------------------------------------------------------


class ConfigService:
    """Read/update/validate the persistent Settings and resolve AI config.

    Precedence: database settings > environment > defaults. Secrecy is
    enforced at this boundary: the public dicts never contain a raw API key,
    bot token, or chat id.
    """

    def __init__(
        self,
        database: Database,
        *,
        secrets_path: str | None = None,
        env: dict[str, Any] | None = None,
        repo: ConfigRepository | None = None,
        secrets: SecretStore | None = None,
        transport_get: Callable[..., Any] | None = None,
        telegram_sender: Callable[[Any], Any] | None = None,
    ) -> None:
        self.database = database
        self.env = env if env is not None else os.environ
        self._repo = repo if repo is not None else ConfigRepository(database)
        self._secrets = secrets if secrets is not None else SecretStore(secrets_path, env=self.env)
        try:
            import requests
        except ImportError:  # pragma: no cover - requests is a core dependency
            requests = None
        self._transport_get = transport_get if transport_get is not None else requests.get
        self._telegram_sender = telegram_sender

    # -- Active-signal helpers -------------------------------------------------

    def active_signal(self) -> Any | None:
        """The active signal (PENDING_ENTRY/OPEN) that locks config changes."""
        return SignalRepository(self.database).get_active_signal()

    def has_active_signal(self) -> bool:
        return self.active_signal() is not None

    def has_stored_demo(self) -> bool:
        """True when the user saved demo settings through the Settings page."""
        return self._repo.get(KEY_DEMO) is not None

    def demo_config_object(self) -> DemoConfig | None:
        """The stored demo settings as a validated :class:`DemoConfig`, or None."""
        stored = self._stored_json(KEY_DEMO)
        if stored is None:
            return None
        return demo_config_from_payload(stored)

    # -- Public views -----------------------------------------------------------

    def get_settings(self) -> dict[str, Any]:
        """Full Settings document for the frontend (no secrets ever included)."""
        live = self._live_ai_record()
        pending = self._stored_json(KEY_AI_PENDING)
        return {
            "ai_provider": self._ai_public(live),
            "pending_ai_provider": self._ai_public(pending) if pending is not None else None,
            "config_state": self.config_state(),
            "symbol": self.get_symbol_public(),
            "demo": self.get_demo_public(),
            "telegram": self.get_telegram_public(),
            "runtime": self.get_runtime_public(),
            "audit": self.get_audit(limit=20),
        }

    def config_state(self) -> str:
        """``pending`` when an AI-provider change waits for the system to go idle."""
        return "pending" if self._repo.get(KEY_AI_PENDING) is not None else "applied"

    def get_audit(self, limit: int = 20) -> list[dict[str, str]]:
        return [
            {"namespace": n, "action": a, "summary": s, "created_at": t}
            for n, a, s, t in self._repo.list_audit(limit=limit)
        ]

    def _ai_public(self, record: dict[str, Any] | None) -> dict[str, Any]:
        if record is None:
            record = {
                "provider": DEFAULT_AI_PROVIDER,
                "preset": DEFAULT_AI_PRESET,
                "base_url": "",
                "model": DEFAULT_AI_MODEL,
                "timeout": DEFAULT_AI_TIMEOUT,
                "temperature": DEFAULT_AI_TEMPERATURE,
                "max_tokens": DEFAULT_AI_MAX_TOKENS,
                "analysis_mode": DEFAULT_AI_ANALYSIS_MODE,
                "api_key_configured": False,
            }
        configured = bool(record.get("api_key_configured")) or self._secrets.configured(
            SECRET_API_KEY
        )
        return {
            "provider": record.get("provider", DEFAULT_AI_PROVIDER),
            "preset": record.get("preset", DEFAULT_AI_PRESET),
            "base_url": record.get("base_url", "") or "",
            "model": record.get("model", DEFAULT_AI_MODEL),
            "timeout": record.get("timeout", DEFAULT_AI_TIMEOUT),
            "temperature": record.get("temperature", DEFAULT_AI_TEMPERATURE),
            "max_tokens": record.get("max_tokens", DEFAULT_AI_MAX_TOKENS),
            "analysis_mode": record.get("analysis_mode", DEFAULT_AI_ANALYSIS_MODE),
            "api_key_configured": configured,
            "api_key_masked": mask_secret(self._secrets.get(SECRET_API_KEY) or ""),
            "updated_at": record.get("updated_at"),
        }

    def get_demo_public(self) -> dict[str, Any]:
        """Effective demo settings: stored DB > account row > env > defaults."""
        stored = self._stored_json(KEY_DEMO)
        if stored is not None:
            values = {
                "initial_balance": str(stored.get("initial_balance", "")),
                "margin_per_trade": str(stored.get("margin_per_trade", "")),
                "leverage": int(stored.get("leverage", 10)),
                "risk_percent": str(stored.get("risk_percent", "")),
                "fee_rate": str(stored.get("fee_rate", "")),
                "updated_at": stored.get("updated_at"),
            }
        else:
            row = DemoRepository(self.database).get_account("demo")
            if row is not None:
                values = {
                    "initial_balance": str(row["initial_balance"]),
                    "margin_per_trade": str(row["margin_per_trade"]),
                    "leverage": int(row["leverage"]),
                    "risk_percent": str(row["risk_percent"]),
                    "fee_rate": str(row["fee_rate"]),
                    "updated_at": row["updated_at"],
                }
            else:
                from .runtime import demo_config_from_env

                demo = demo_config_from_env(self.env)
                values = {
                    "initial_balance": str(demo.initial_balance),
                    "margin_per_trade": str(demo.margin_per_trade),
                    "leverage": demo.leverage,
                    "risk_percent": str(demo.risk_percent),
                    "fee_rate": str(demo.fee_rate),
                    "updated_at": None,
                }
        values["locked"] = self.has_active_signal()
        return values

    def get_telegram_public(self) -> dict[str, Any]:
        stored = self._stored_json(KEY_TELEGRAM)
        env_enabled = str(self.env.get("BTCUSDT_TELEGRAM_ENABLED", "")).lower() in ("1", "true", "yes")
        env_token = self.env.get("BTCUSDT_TELEGRAM_BOT_TOKEN", "") or ""
        env_chat = self.env.get("BTCUSDT_TELEGRAM_CHAT_ID", "") or ""
        env_signals = self.env.get("BTCUSDT_TELEGRAM_CHAT_ID_SIGNALS", "") or ""
        env_reports = self.env.get("BTCUSDT_TELEGRAM_CHAT_ID_REPORTS", "") or ""
        enabled = bool(stored.get("enabled", env_enabled)) if stored is not None else env_enabled
        stored_legacy = (stored.get("chat_id", "") if stored is not None else "") or ""
        signals = (
            ((stored.get("chat_id_signals", "") if stored is not None else "") or "")
            or stored_legacy
            or env_signals
            or env_chat
        )
        reports = (
            ((stored.get("chat_id_reports", "") if stored is not None else "") or "")
            or stored_legacy
            or env_reports
            or env_chat
        )
        token_configured = bool(
            self._secrets.configured(SECRET_BOT_TOKEN) or env_token or (stored or {}).get("token_configured", False)
        )
        return {
            "enabled": enabled,
            "chat_id_configured": bool(signals),
            "chat_id_masked": "********" if signals else "",
            "chat_id_signals_configured": bool(signals),
            "chat_id_signals_masked": "********" if signals else "",
            "chat_id_reports_configured": bool(reports),
            "chat_id_reports_masked": "********" if reports else "",
            "token_configured": token_configured,
            "token_masked": mask_secret(self._secrets.get(SECRET_BOT_TOKEN) or env_token),
            "updated_at": (stored or {}).get("updated_at"),
        }

    def get_runtime_public(self) -> dict[str, Any]:
        from .runtime import runtime_config_from_env

        config = runtime_config_from_env(self.env)
        return {
            "symbol": self.resolve_symbol(),
            "timeframe": config.timeframe,
            "scheduler_poll": config.scheduler_poll,
            "monitor_poll": config.monitor_poll,
            "db_path": config.db_path,
            "db_connected": self._db_connected(),
        }

    def _db_connected(self) -> bool:
        try:
            with self.database.read() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:  # pragma: no cover - defensive
            return False

    # -- AI provider resolution ---------------------------------------------------

    def resolve_llm_config(self) -> LLMConfig:
        """Resolve the effective :class:`LLMConfig` for the next analysis.

        Precedence: saved database AI settings > environment (unchanged legacy
        behavior). The api key is injected only when the user configured one
        through the Settings page; otherwise the shared provider layer reads
        its own environment credential exactly as before.
        """
        record = self._live_ai_record()
        if record is None:
            return llm_config_from_env(self.env)

        provider = record.get("provider", "ollama")
        preset = record.get("preset", "custom")
        if provider == "ollama":
            llm_provider = "ollama"
            base_url = _ollama_chat_base(record.get("base_url", ""), self.env)
        else:
            llm_provider = API_PRESET_PROVIDER.get(preset, "openai_compatible")
            base_url = (record.get("base_url", "") or "").strip() or None

        api_key = (
            self._secrets.get(SECRET_API_KEY)
            if record.get("api_key_configured") or self._secrets.configured(SECRET_API_KEY)
            else None
        )
        return LLMConfig(
            provider=llm_provider,
            model=record.get("model", DEFAULT_AI_MODEL),
            base_url=base_url,
            timeout=self._as_float(record.get("timeout"), DEFAULT_AI_TIMEOUT),
            temperature=self._as_float(record.get("temperature"), None),
            max_tokens=self._as_int(record.get("max_tokens"), None),
            api_key=api_key,
        )

    @staticmethod
    def _as_float(value: Any, default: float | None) -> float | None:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int | None) -> int | None:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def resolve_analysis_mode(self) -> str:
        """Effective analysis pipeline: DB AI settings > env > default.

        Returns ``"multi"`` or ``"single"``. Resolved fresh on every scheduler
        pass so a Settings-page toggle applies to the NEXT eligible analysis
        without a daemon restart (mirrors ``resolve_llm_config``).
        """
        record = self._live_ai_record()
        if record is not None:
            mode = record.get("analysis_mode")
            if mode in ANALYSIS_MODES:
                return mode
        env_mode = str(self.env.get(ENV_ANALYSIS_MODE, "") or "").strip().lower()
        if env_mode in ANALYSIS_MODES:
            return env_mode
        return DEFAULT_AI_ANALYSIS_MODE

    # -- Market symbol resolution --------------------------------------------------

    def resolve_symbol(self) -> str:
        """The symbol the daemon should analyse: DB settings > env > default.

        Called fresh on every scheduler pass, so a Settings-page symbol change
        takes effect for the next eligible analysis without a daemon restart.
        """
        stored = self._stored_json(KEY_SYMBOL)
        if stored is not None and stored.get("symbol") in SUPPORTED_SYMBOLS:
            return stored["symbol"]
        return self._env_symbol()

    def _env_symbol(self) -> str:
        from .runtime import ENV_SYMBOL

        value = str(self.env.get(ENV_SYMBOL, "") or "").strip().upper()
        if value and value in SUPPORTED_SYMBOLS:
            return value
        return DEFAULT_SYMBOL

    def get_symbol_public(self) -> dict[str, Any]:
        """Effective symbol + the Settings dropdown choices (never sensitive)."""
        stored = self._stored_json(KEY_SYMBOL)
        return {
            "symbol": self.resolve_symbol(),
            "supported": list(SUPPORTED_SYMBOLS),
            "updated_at": (stored or {}).get("updated_at"),
        }

    def update_symbol(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist the analysed symbol for FUTURE analyses.

        This is never locked while a signal is active: the AI lock already
        prevents any new analysis, and an OPEN position is monitored through its
        own signal's symbol, so the running trade is never affected (AGENTS.md
        sections 3, 7, 9).
        """
        symbol = validate_symbol_setting(payload.get("symbol"))
        record = {"symbol": symbol, "updated_at": iso_utc_now()}
        self._repo.set(KEY_SYMBOL, json.dumps(record))
        self._audit("symbol", "update", f"Active symbol set to {symbol}.")
        logger.info("[Config] active symbol set to %s", symbol)
        return self.get_symbol_public()

    # -- Updates -------------------------------------------------------------------

    def update_ai_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and stage an AI provider change; promote when idle.

        Never touches an open signal. Returns the updated public settings.
        """
        settings = validate_ai_provider(payload)
        api_key = payload.get("api_key")
        key_changed = False
        if isinstance(api_key, str):
            candidate = api_key.strip()
            is_mask = "*" in candidate or not candidate
            if candidate and not is_mask:
                self._secrets.set(SECRET_API_KEY, candidate)
                key_changed = True

        api_key_configured = bool(
            settings.api_key_configured
            or self._secrets.configured(SECRET_API_KEY)
        )
        record = {
            "provider": settings.provider,
            "preset": settings.preset,
            "base_url": settings.base_url,
            "model": settings.model,
            "timeout": settings.timeout,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "analysis_mode": settings.analysis_mode,
            "api_key_configured": api_key_configured,
            "updated_at": iso_utc_now(),
        }
        self._repo.set(KEY_AI_PENDING, json.dumps(record))
        applied = self.has_active_signal() is False
        if applied:
            self._promote_ai()
        self._audit_ai(settings, applied=applied, key_changed=key_changed)
        return self.get_settings()

    def _promote_ai(self) -> None:
        pending = self._repo.get(KEY_AI_PENDING)
        if pending is None:
            return
        self._repo.promote(KEY_AI_PENDING, KEY_AI_PROVIDER)
        self._repo.delete(KEY_AI_PENDING)

    def apply_pending_if_idle(self) -> bool:
        """Promote a staged AI-provider change once the system is idle.

        Returns ``True`` when a pending change was applied. Called by the
        runtime before each scheduler tick and at startup.
        """
        if self._repo.get(KEY_AI_PENDING) is None or self.has_active_signal():
            return False
        self._promote_ai()
        self._audit(
            "ai_provider",
            "apply",
            "A previously staged AI provider configuration was applied (system idle).",
        )
        logger.info("[Config] applied pending AI provider configuration (system idle)")
        return True

    def update_demo(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist the Demo account settings for FUTURE trades.

        Refused (409) while a signal is active. The account row's configurable
        fields are updated in the same step; balance/initial_balance/peak_equity
        are never touched (AGENTS.md section 23).
        """
        if self.has_active_signal():
            raise SettingsLockedError(
                "Demo configuration is locked while a signal is active."
            )
        config = demo_config_from_payload(payload)
        self._update_account_config(config)
        record = {
            "initial_balance": str(config.initial_balance),
            "margin_per_trade": str(config.margin_per_trade),
            "leverage": config.leverage,
            "risk_percent": str(config.risk_percent),
            "fee_rate": str(config.fee_rate),
            "updated_at": iso_utc_now(),
        }
        self._repo.set(KEY_DEMO, json.dumps(record))
        self._audit(
            "demo",
            "update",
            f"Demo account settings updated (leverage {config.leverage}x, "
            f"margin {config.margin_per_trade} USDT, risk {config.risk_percent}%, "
            f"fee {config.fee_rate}).",
        )
        return self.get_demo_public()

    def reset_demo_account(self) -> dict[str, Any]:
        """Reset the demo account to a full fresh start (user-initiated).

        Sets ``initial_balance``/``balance``/``equity``/``peak_equity`` to the
        effective ``initial_balance`` and clears all demo positions and trades
        in one atomic transaction. The signal ledger is NEVER touched. Refused
        (409) while a signal is active so an in-memory runtime position can
        never be orphaned. The target balance is the stored demo setting, not
        the value currently on the account row.
        """
        if self.has_active_signal():
            raise SettingsLockedError(
                "Demo reset is locked while a signal is active."
            )
        repo = DemoRepository(self.database)
        row = repo.get_account("demo")
        if row is None:
            return self.get_demo_public()
        stored = self._stored_json(KEY_DEMO)
        if stored is not None and stored.get("initial_balance"):
            initial_balance: Any = stored.get("initial_balance")
        else:
            initial_balance = row["initial_balance"]
        repo.reset_account(int(row["id"]), initial_balance=initial_balance)
        self._audit(
            "demo",
            "reset",
            f"Demo account reset to a fresh start at {initial_balance} USDT "
            "(positions and demo trades cleared; signal ledger preserved).",
        )
        logger.info(
            "[Config] demo account %s reset (balance=equity=peak=%s)",
            row["id"],
            initial_balance,
        )
        return self.get_demo_public()

    def _update_account_config(self, config: DemoConfig) -> None:
        repo = DemoRepository(self.database)
        row = repo.get_account("demo")
        if row is None:
            return
        repo.update_account_config(
            int(row["id"]),
            margin_per_trade=config.margin_per_trade,
            leverage=config.leverage,
            risk_percent=config.risk_percent,
            fee_rate=config.fee_rate,
        )
        logger.info("[Config] demo account %s config updated for future trades", row["id"])

    def update_telegram(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist the Telegram notification settings (no token echoes)."""
        enabled = bool(payload.get("enabled", False))
        legacy = _optional_str(payload, "chat_id")
        signals = _optional_str(payload, "chat_id_signals") or legacy
        reports = _optional_str(payload, "chat_id_reports") or legacy
        token = payload.get("bot_token")
        if isinstance(token, str) and token.strip() and "*" not in token:
            self._secrets.set(SECRET_BOT_TOKEN, token.strip())
        existing_token = self._secrets.get(SECRET_BOT_TOKEN) or self.env.get("BTCUSDT_TELEGRAM_BOT_TOKEN", "")
        if enabled and (not (signals or reports) or not existing_token):
            raise SettingsValidationError(
                "Enabling Telegram requires a bot token and at least one chat id (signals or reports)."
            )
        token_configured = bool(existing_token)
        record = {
            "enabled": enabled,
            "chat_id": legacy,
            "chat_id_signals": signals,
            "chat_id_reports": reports,
            "token_configured": token_configured,
            "updated_at": iso_utc_now(),
        }
        self._repo.set(KEY_TELEGRAM, json.dumps(record))
        self._audit(
            "telegram",
            "update",
            f"Telegram notifications {'enabled' if enabled else 'disabled'}.",
        )
        return self.get_telegram_public()

    # -- Connectivity tests --------------------------------------------------------

    def test_llm_connection(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Test the submitted AI provider config live (never returns a secret)."""
        provider = str(payload.get("provider", "")).lower() or "ollama"
        model = str(payload.get("model", "")).strip() or DEFAULT_AI_MODEL
        started = time.monotonic()
        if provider == "ollama":
            base_url = str(payload.get("base_url", "") or "")
            return self._test_ollama(base_url, model, started)
        preset = str(payload.get("preset", "")).lower() or "custom"
        base_url = str(payload.get("base_url", "") or "")
        api_key = payload.get("api_key")
        return self._test_openai_compatible(base_url, model, preset, api_key, started)

    def _test_ollama(self, base_url: str, model: str, started: float) -> dict[str, Any]:
        url = _ollama_tags_url(base_url, self.env)
        try:
            response = self._transport_get(url, timeout=_HTTP_TEST_TIMEOUT)
        except Exception as exc:
            return {
                "success": False,
                "provider": "ollama",
                "model": model,
                "latency_ms": self._latency_ms(started),
                "model_available": False,
                "error": self._safe_error(exc),
            }
        if response.status_code != 200:
            return {
                "success": False,
                "provider": "ollama",
                "model": model,
                "latency_ms": self._latency_ms(started),
                "model_available": False,
                "error": f"Ollama server returned HTTP {response.status_code}",
            }
        names: list[str] = []
        try:
            data = response.json()
            names = sorted(str(m.get("name")) for m in data.get("models", []) if isinstance(m, dict))
        except Exception:  # pragma: no cover - malformed third-party payload
            names = []
        return {
            "success": True,
            "provider": "ollama",
            "model": model,
            "latency_ms": self._latency_ms(started),
            "model_available": model in names,
            "available_models": names[:100],
            "error": None,
        }

    def _test_openai_compatible(
        self, base_url: str, model: str, preset: str, api_key: Any, started: float
    ) -> dict[str, Any]:
        url = base_url.strip().rstrip("/") or API_PRESET_DEFAULT_BASE_URL.get(preset, "")
        if not url:
            return {
                "success": False,
                "provider": "api",
                "model": model,
                "latency_ms": self._latency_ms(started),
                "error": "base_url is required for this preset",
            }
        headers = {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key.strip() else {}
        try:
            response = self._transport_get(url + "/models", timeout=_HTTP_TEST_TIMEOUT, headers=headers)
        except Exception as exc:
            return {
                "success": False,
                "provider": "api",
                "model": model,
                "latency_ms": self._latency_ms(started),
                "error": self._safe_error(exc),
            }
        if response.status_code != 200:
            return {
                "success": False,
                "provider": "api",
                "model": model,
                "latency_ms": self._latency_ms(started),
                "error": f"API endpoint returned HTTP {response.status_code}",
            }
        return {
            "success": True,
            "provider": "api",
            "model": model,
            "latency_ms": self._latency_ms(started),
            "error": None,
        }

    def list_ollama_models(self, base_url: str) -> list[str]:
        """Installed Ollama models from ``/api/tags`` (sanitized failure)."""
        url = _ollama_tags_url(base_url, self.env)
        try:
            response = self._transport_get(url, timeout=_HTTP_TEST_TIMEOUT)
        except Exception as exc:
            raise SettingsTestError(self._safe_error(exc)) from exc
        if response.status_code != 200:
            raise SettingsTestError(f"Ollama server returned HTTP {response.status_code}")
        data = response.json()
        names = sorted(str(m.get("name")) for m in data.get("models", []) if isinstance(m, dict))
        return names

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        """A sanitized, secret-free error message for a transport failure."""
        return f"{type(exc).__name__}: connection failed"

    @staticmethod
    def _latency_ms(started: float) -> int:
        return max(0, int(round((time.monotonic() - started) * 1000)))

    def test_telegram(self, channel: str = "reports") -> dict[str, Any]:
        """Send a test Telegram message to one channel (never echoes secrets).

        ``channel`` is ``"signals"`` or ``"reports"`` (default); anything else
        resolves like the router (missing channel falls back to the other).
        """
        stored = self._stored_json(KEY_TELEGRAM) or {}
        legacy = stored.get("chat_id", "") or self.env.get("BTCUSDT_TELEGRAM_CHAT_ID", "") or ""
        if channel == "signals":
            chat_id = (
                stored.get("chat_id_signals", "")
                or legacy
                or self.env.get("BTCUSDT_TELEGRAM_CHAT_ID_SIGNALS", "")
                or self.env.get("BTCUSDT_TELEGRAM_CHAT_ID", "")
                or ""
            )
        else:
            chat_id = (
                stored.get("chat_id_reports", "")
                or legacy
                or self.env.get("BTCUSDT_TELEGRAM_CHAT_ID_REPORTS", "")
                or self.env.get("BTCUSDT_TELEGRAM_CHAT_ID", "")
                or ""
            )
        token = self._secrets.get(SECRET_BOT_TOKEN) or self.env.get("BTCUSDT_TELEGRAM_BOT_TOKEN", "") or ""
        if not token or not chat_id:
            return {"success": False, "message_id": None, "error": "Telegram bot token and chat id are required."}
        from .telegram import TelegramClient, TelegramConfig

        config = TelegramConfig(bot_token=token, chat_id=chat_id, enabled=True)
        if self._telegram_sender is not None:
            result = self._telegram_sender(config)
        else:
            result = TelegramClient(config).send_message("BTCUSDT Signal Engine — Telegram test message")
        if result.ok:
            return {"success": True, "message_id": result.message_id, "error": None}
        return {"success": False, "message_id": None, "error": result.error or "send failed"}

    def telegram_config_object(self) -> Any:
        """Effective Telegram config: stored Settings > env > defaults.

        The bot token always comes from the 0600 secrets file (falling back to
        the environment) and is never logged or echoed. ``timeout`` /
        ``max_retries`` / ``backoff`` keep their environment defaults. Each
        channel resolves new key > legacy key, env or stored; the legacy
        single chat fills both. Callers must treat the returned config as
        read-only.
        """
        from .telegram import TelegramConfig, telegram_config_from_env

        try:
            base = telegram_config_from_env(self.env)
        except Exception:  # pragma: no cover - malformed env must not break the daemon
            base = TelegramConfig()
        stored = self._stored_json(KEY_TELEGRAM)
        if stored is not None:
            enabled = bool(stored.get("enabled"))
            stored_legacy = str(stored.get("chat_id") or "")
            signals = (
                str(stored.get("chat_id_signals") or "")
                or stored_legacy
                or base.chat_id_signals
                or base.chat_id
            )
            reports = (
                str(stored.get("chat_id_reports") or "")
                or stored_legacy
                or base.chat_id_reports
                or base.chat_id
            )
            legacy_out = stored_legacy or base.chat_id
        else:
            enabled = base.enabled
            signals = base.chat_id_signals or base.chat_id
            reports = base.chat_id_reports or base.chat_id
            legacy_out = base.chat_id
        token = self._secrets.get(SECRET_BOT_TOKEN) or base.bot_token
        return TelegramConfig(
            bot_token=token,
            chat_id=legacy_out,
            chat_id_signals=signals,
            chat_id_reports=reports,
            enabled=enabled,
            timeout=base.timeout,
            max_retries=base.max_retries,
            backoff=base.backoff,
        )

    def resolve_telegram_notifier(self) -> Any:
        """A ``TelegramNotifier`` built from the effective stored/env settings.

        The daemon uses this so a Telegram save on the Settings page takes
        effect without a restart. Never logs or echoes the bot token.
        """
        from .telegram import TelegramNotifier

        return TelegramNotifier(self.telegram_config_object())

    def _audit_ai(self, settings: AIProviderSettings, *, applied: bool, key_changed: bool) -> None:
        action = "apply" if applied else "stage"
        summary = (
            f"AI provider settings {'applied' if applied else 'staged (applies when idle)'} "
            f"({settings.provider}"
            + (f" / {settings.preset}" if settings.provider == "api" else "")
            + f", model {settings.model})."
        )
        self._audit("ai_provider", action, summary)
        if key_changed:
            self._audit("ai_provider", "set_api_key", "API key updated (value not recorded).")

    def _audit(self, namespace: str, action: str, summary: str) -> None:
        try:
            self._repo.append_audit(namespace, action, summary)
        except Exception as exc:  # pragma: no cover - audit must never crash config
            logger.warning("[Config] audit write failed: %s", exc)

    def _stored_json(self, key: str) -> dict[str, Any] | None:
        raw = self._repo.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("[Config] stored settings %s are not valid JSON", key)
            return None
        return data if isinstance(data, dict) else None

    def _live_ai_record(self) -> dict[str, Any] | None:
        return self._stored_json(KEY_AI_PROVIDER)


__all__ = [
    "AIProviderSettings",
    "ConfigService",
    "DEFAULT_AI_MAX_TOKENS",
    "DEFAULT_AI_MODEL",
    "DEFAULT_AI_PRESET",
    "DEFAULT_AI_PROVIDER",
    "DEFAULT_AI_TEMPERATURE",
    "DEFAULT_AI_TIMEOUT",
    "DEFAULT_SECRETS_FILE",
    "ENV_SECRETS_FILE",
    "KEY_AI_PENDING",
    "KEY_AI_PROVIDER",
    "KEY_DEMO",
    "KEY_RUNTIME",
    "KEY_SYMBOL",
    "KEY_TELEGRAM",
    "SECRET_API_KEY",
    "SECRET_BOT_TOKEN",
    "SUPPORTED_SYMBOLS",
    "SettingsError",
    "SettingsLockedError",
    "SettingsTestError",
    "SettingsValidationError",
    "SecretStore",
    "demo_config_from_payload",
    "mask_secret",
    "validate_ai_provider",
    "validate_symbol_setting",
]
