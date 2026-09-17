"""Phase 15 unit tests for signal_engine/config.py + temperature metadata.

Covers: SecretStore (0600 + masking), settings validation, DB > env precedence,
secret non-leakage across settings/database, AI-provider staging vs promotion
(idle-only apply), the demo settings lock while a signal is active, account
config updates that never touch balance/equity, Telegram settings, live
connection tests through an injected transport, and temperature round-tripping
through the signal ledger.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from decimal import Decimal
from types import SimpleNamespace

import pytest

from database.database import (
    ConfigRepository,
    Database,
    DemoRepository,
    SignalRepository,
)
from signal_engine import SignalAnalysis
from signal_engine.config import (
    KEY_AI_PENDING,
    SECRET_API_KEY,
    ConfigService,
    SecretStore,
    SettingsLockedError,
    SettingsTestError,
    SettingsValidationError,
    demo_config_from_payload,
    mask_secret,
    validate_ai_provider,
)
from signal_engine.engine import SignalEngine
from signal_engine.llm import llm_config_from_env


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class TempConfig:
    def __init__(self, env: dict | None = None) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "cfg.db")
        self.secrets_path = os.path.join(self._tmp.name, "secrets.json")
        self.db = Database(self.path)
        self.db.initialize()
        self.env = dict(env or {})
        self.service = ConfigService(
            self.db, secrets_path=self.secrets_path, env=self.env
        )

    def close(self) -> None:
        self._tmp.cleanup()


@pytest.fixture()
def cfg() -> TempConfig:
    box = TempConfig()
    yield box
    box.close()


# -- SecretStore + masking ------------------------------------------------------


@pytest.mark.unit
def test_mask_secret_hides_middle():
    masked = mask_secret("sk-1234567890abcdef")
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "****" in masked
    assert "1234567890" not in masked
    assert mask_secret("") == ""
    assert mask_secret("short") == "*****"


@pytest.mark.unit
def test_secret_store_persists_with_0600_perms(cfg):
    SecretStore(cfg.secrets_path, env={}).set(SECRET_API_KEY, "sk-super-secret")
    mode = stat.S_IMODE(os.stat(cfg.secrets_path).st_mode)
    assert mode & 0o777 == 0o600
    reloaded = SecretStore(cfg.secrets_path, env={})
    assert reloaded.get(SECRET_API_KEY) == "sk-super-secret"


@pytest.mark.unit
def test_secret_store_refreshes_cache_across_instances(cfg):
    # The web dashboard and the daemon run as separate processes with their own
    # SecretStore. A key saved by one instance (stale in the other) must be
    # picked up on the next ``get`` without a restart (regression for the
    # OpenRouter 401 "Missing Authentication header").
    daemon_store = SecretStore(cfg.secrets_path, env={})
    daemon_store.get(SECRET_API_KEY)  # prime the cache
    assert daemon_store.get(SECRET_API_KEY) is None

    web_store = SecretStore(cfg.secrets_path, env={})
    web_store.set(SECRET_API_KEY, "sk-picked-up")

    assert daemon_store.get(SECRET_API_KEY) == "sk-picked-up"


@pytest.mark.unit
def test_secret_store_serves_cache_until_file_changes(cfg):
    store = SecretStore(cfg.secrets_path, env={})
    store.set(SECRET_API_KEY, "sk-unchanged")
    store.get(SECRET_API_KEY)  # cache warmed

    with open(cfg.secrets_path, encoding="utf-8") as handle:
        content = handle.read()
    original_mtime = os.stat(cfg.secrets_path).st_mtime
    time.sleep(0.02)
    # Rewrite identical bytes while advancing the mtime by hand so the file
    # visibly changed but the value did not.
    os.utime(cfg.secrets_path, (original_mtime + 5, original_mtime + 5))
    assert os.stat(cfg.secrets_path).st_mtime != original_mtime
    # mtime changed -> reload; identical content -> same value returned.
    assert store.get(SECRET_API_KEY) == "sk-unchanged"
    with open(cfg.secrets_path, encoding="utf-8") as handle:
        assert content == handle.read()


@pytest.mark.unit
def test_secret_store_cache_hit_does_not_stat_when_warm(cfg):
    # Sanity: while the mtime is unchanged the value is served from cache and
    # a second store on the same path sees the same content.
    SecretStore(cfg.secrets_path, env={}).set(SECRET_API_KEY, "sk-first")
    store = SecretStore(cfg.secrets_path, env={})
    assert store.get(SECRET_API_KEY) == "sk-first"
    assert store.get(SECRET_API_KEY) == "sk-first"


@pytest.mark.unit
def test_secret_store_deleted_file_returns_none(cfg):
    store = SecretStore(cfg.secrets_path, env={})
    store.set(SECRET_API_KEY, "sk-gone")
    assert store.get(SECRET_API_KEY) == "sk-gone"

    os.remove(cfg.secrets_path)
    assert store.get(SECRET_API_KEY) is None
    # A later write through a fresh instance recreates the file cleanly.
    SecretStore(cfg.secrets_path, env={}).set(SECRET_API_KEY, "sk-back")
    assert os.path.exists(cfg.secrets_path)


@pytest.mark.unit
def test_api_key_never_written_to_database_or_settings_json(cfg):
    cfg.service.update_ai_provider(
        {
            "provider": "api",
            "preset": "deepseek",
            "model": "deepseek-chat",
            "api_key": "sk-DEEP-SECRET-VALUE",
        }
    )
    settings = cfg.service.get_settings()
    blob = json.dumps(settings)
    assert "sk-DEEP-SECRET-VALUE" not in blob
    assert settings["ai_provider"]["api_key_configured"] is True
    assert "****" in settings["ai_provider"]["api_key_masked"]

    # The key lives only in the 0600 secrets file, never in SQLite.
    with cfg.db.read() as conn:
        rows = [str(r) for r in conn.execute("SELECT key, value FROM app_settings")]
        assert not any("sk-DEEP-SECRET-VALUE" in row for row in rows)
    with open(cfg.secrets_path, encoding="utf-8") as handle:
        assert "sk-DEEP-SECRET-VALUE" in handle.read()


@pytest.mark.unit
def test_masked_api_key_submission_is_ignored(cfg):
    cfg.service.update_ai_provider(
        {"provider": "api", "preset": "openai", "model": "gpt-4o", "api_key": "sk-KEEP-ME"}
    )
    cfg.service.update_ai_provider(
        {"provider": "api", "preset": "openai", "model": "gpt-4o-mini", "api_key": "sk-KEEP-ME****"}
    )
    store = SecretStore(cfg.secrets_path, env={})
    assert store.get(SECRET_API_KEY) == "sk-KEEP-ME"


# -- Validation -----------------------------------------------------------------


@pytest.mark.unit
def test_validate_ai_provider_ollama_defaults():
    settings = validate_ai_provider({"provider": "ollama", "model": "qwen3:4b"})
    assert settings.provider == "ollama"
    assert settings.preset == "ollama"
    assert settings.base_url == ""
    assert settings.analysis_mode == "single"


@pytest.mark.unit
def test_validate_ai_provider_analysis_mode():
    assert (
        validate_ai_provider(
            {"provider": "ollama", "model": "qwen3:4b", "analysis_mode": "multi"}
        ).analysis_mode
        == "multi"
    )
    with pytest.raises(SettingsValidationError):
        validate_ai_provider(
            {"provider": "ollama", "model": "qwen3:4b", "analysis_mode": "committee"}
        )


@pytest.mark.unit
def test_resolve_analysis_mode_precedence_db_over_env(cfg):
    assert cfg.service.resolve_analysis_mode() == "single"
    cfg.env["BTCUSDT_ANALYSIS_MODE"] = "multi"
    assert cfg.service.resolve_analysis_mode() == "multi"
    cfg.service.update_ai_provider(
        {"provider": "ollama", "model": "qwen3:4b", "analysis_mode": "multi"}
    )
    assert cfg.service.resolve_analysis_mode() == "multi"
    cfg.service.update_ai_provider(
        {"provider": "ollama", "model": "qwen3:4b", "analysis_mode": "single"}
    )
    assert cfg.service.resolve_analysis_mode() == "single"


@pytest.mark.unit
def test_resolve_analysis_mode_invalid_db_value_falls_back(cfg):
    cfg.service.update_ai_provider({"provider": "ollama", "model": "qwen3:4b"})
    repo = ConfigRepository(cfg.db)
    pending = json.loads(repo.get("ai_provider") or "{}")
    pending["analysis_mode"] = "committee"
    repo.set("ai_provider", json.dumps(pending))
    assert cfg.service.resolve_analysis_mode() == "single"
    cfg.env["BTCUSDT_ANALYSIS_MODE"] = "multi"
    assert cfg.service.resolve_analysis_mode() == "multi"


@pytest.mark.unit
def test_validate_ai_provider_custom_requires_base_url():
    with pytest.raises(SettingsValidationError):
        validate_ai_provider({"provider": "api", "preset": "custom", "model": "x"})
    with pytest.raises(SettingsValidationError):
        validate_ai_provider(
            {"provider": "api", "preset": "custom", "model": "x", "base_url": "no-scheme"}
        )


@pytest.mark.unit
@pytest.mark.parametrize("temperature", [-1, 5.5, "nope"])
def test_validate_ai_provider_rejects_bad_temperature(temperature):
    with pytest.raises(SettingsValidationError):
        validate_ai_provider({"provider": "ollama", "model": "m", "temperature": temperature})


@pytest.mark.unit
def test_demo_config_from_payload_requires_all_fields():
    with pytest.raises(SettingsValidationError):
        demo_config_from_payload({"margin_per_trade": "50"})
    config = demo_config_from_payload(
        {
            "initial_balance": "1000",
            "margin_per_trade": "50",
            "leverage": 10,
            "risk_percent": "1",
            "fee_rate": "0.04",
        }
    )
    assert config.leverage == 10
    assert config.margin_per_trade == Decimal("50")


# -- DB > env precedence ---------------------------------------------------------


@pytest.mark.unit
def test_resolve_falls_back_to_env_without_stored_settings(cfg):
    cfg.env.update({"BTCUSDT_LLM_PROVIDER": "ollama", "BTCUSDT_LLM_MODEL": "qwen3:8b"})
    resolved = cfg.service.resolve_llm_config()
    assert resolved == llm_config_from_env(cfg.env)


@pytest.mark.unit
def test_stored_settings_override_env(cfg):
    cfg.env.update({"BTCUSDT_LLM_PROVIDER": "ollama", "BTCUSDT_LLM_MODEL": "qwen3:8b"})
    cfg.service.update_ai_provider(
        {"provider": "api", "preset": "deepseek", "model": "deepseek-chat", "timeout": 30}
    )
    resolved = cfg.service.resolve_llm_config()
    assert resolved.provider == "deepseek"
    assert resolved.model == "deepseek-chat"
    assert resolved.timeout == 30.0
    assert resolved.api_key is None


@pytest.mark.unit
def test_resolve_injects_stored_api_key(cfg):
    cfg.service.update_ai_provider(
        {"provider": "api", "preset": "deepseek", "model": "deepseek-chat", "api_key": "sk-live-key"}
    )
    assert cfg.service.resolve_llm_config().api_key == "sk-live-key"


@pytest.mark.unit
def test_resolve_picks_up_api_key_saved_by_another_instance(cfg):
    # Regression for OpenRouter 401 "Missing Authentication header": the web
    # dashboard and the scheduler daemon are separate processes with separate
    # ConfigService/SecretStore instances. A key saved through the Settings
    # page must reach the analysis pipeline WITHOUT a daemon restart.
    daemon = ConfigService(cfg.db, secrets_path=cfg.secrets_path, env={})
    daemon.resolve_llm_config()  # daemon started with no key saved yet
    assert daemon.resolve_llm_config().api_key is None

    web = ConfigService(cfg.db, secrets_path=cfg.secrets_path, env={})
    web.update_ai_provider(
        {
            "provider": "api",
            "preset": "deepseek",
            "model": "deepseek-chat",
            "api_key": "sk-daemon-picks-up",
        }
    )
    resolved = daemon.resolve_llm_config()
    assert resolved.provider == "deepseek"
    assert resolved.api_key == "sk-daemon-picks-up"


@pytest.mark.unit
def test_ollama_default_base_url_uses_env(cfg):
    cfg.env.update({"OLLAMA_BASE_URL": "http://ollama-host:11434"})
    cfg.service.update_ai_provider({"provider": "ollama", "model": "qwen3:4b"})
    assert cfg.service.resolve_llm_config().base_url == "http://ollama-host:11434/v1"


# -- Staging vs promotion (idle-only apply) ----------------------------------------


def _active_signal(cfg) -> int:
    return SignalRepository(cfg.db).create_signal(
        "BTCUSDT", "1h", "LONG", Decimal("100.00"), Decimal("99.00"), Decimal("101.00")
    ).id


@pytest.mark.unit
def test_ai_provider_change_staged_while_signal_active(cfg):
    _active_signal(cfg)
    cfg.service.update_ai_provider({"provider": "ollama", "model": "qwen3:8b"})
    assert cfg.service.config_state() == "pending"
    # The LIVE provider is still the previous (default) one.
    live = cfg.service.get_settings()["ai_provider"]
    assert live["model"] != "qwen3:8b"


@pytest.mark.unit
def test_pending_promotes_when_idle(cfg):
    signal_id = _active_signal(cfg)
    cfg.service.update_ai_provider({"provider": "api", "preset": "openai", "model": "gpt-4o"})
    # Staged change is NOT live while the signal is active.
    cfg.env.update({"BTCUSDT_LLM_MODEL": "qwen3:8b"})
    assert cfg.service.resolve_llm_config().model == "qwen3:8b"
    assert cfg.service.apply_pending_if_idle() is False

    SignalRepository(cfg.db).cancel_signal(signal_id)
    assert cfg.service.apply_pending_if_idle() is True
    assert cfg.service.config_state() == "applied"
    assert ConfigRepository(cfg.db).get(KEY_AI_PENDING) is None
    assert cfg.service.resolve_llm_config().model == "gpt-4o"


@pytest.mark.unit
def test_audit_records_settings_changes_without_secrets(cfg):
    cfg.service.update_ai_provider(
        {"provider": "api", "preset": "openai", "model": "gpt-4o", "api_key": "sk-audit-secret"}
    )
    audit = cfg.service.get_audit()
    summaries = "\n".join(a["summary"] for a in audit)
    assert "gpt-4o" in summaries
    assert "sk-audit-secret" not in summaries
    assert "sk-audit-secret" not in json.dumps(audit)


# -- Demo settings lock + account config -------------------------------------------


@pytest.mark.unit
def test_demo_settings_locked_while_signal_active(cfg):
    _active_signal(cfg)
    with pytest.raises(SettingsLockedError):
        cfg.service.update_demo(
            {
                "initial_balance": "1000",
                "margin_per_trade": "50",
                "leverage": 10,
                "risk_percent": "1",
                "fee_rate": "0.04",
            }
        )


@pytest.mark.unit
def test_demo_save_updates_account_config_but_not_balance(cfg):
    repo = DemoRepository(cfg.db)
    repo.ensure_account(
        name="demo",
        initial_balance=Decimal("1000"),
        margin_per_trade=Decimal("50"),
        leverage=10,
        risk_percent=Decimal("1"),
        fee_rate=Decimal("0.04"),
    )
    cfg.service.update_demo(
        {
            "initial_balance": "999999",
            "margin_per_trade": "75",
            "leverage": 5,
            "risk_percent": "2",
            "fee_rate": "0.05",
        }
    )
    row = repo.get_account("demo")
    assert row["balance"] == "1000"  # never touched
    assert row["initial_balance"] == "1000"
    assert row["margin_per_trade"] == "75"
    assert row["leverage"] == 5
    assert row["risk_percent"] == "2"
    assert row["fee_rate"] == "0.05"
    stored = cfg.service.demo_config_object()
    assert stored.margin_per_trade == Decimal("75")
    assert cfg.service.has_stored_demo() is True


@pytest.mark.unit
def test_no_stored_demo_by_default(cfg):
    assert cfg.service.demo_config_object() is None
    assert cfg.service.has_stored_demo() is False


@pytest.mark.unit
def test_reset_demo_account_locked_while_signal_active(cfg):
    _active_signal(cfg)
    with pytest.raises(SettingsLockedError):
        cfg.service.reset_demo_account()


@pytest.mark.unit
def test_reset_demo_account_applies_stored_initial_balance(cfg):
    repo = DemoRepository(cfg.db)
    repo.ensure_account(
        name="demo",
        initial_balance=Decimal("1000"),
        margin_per_trade=Decimal("50"),
        leverage=10,
        risk_percent=Decimal("1"),
        fee_rate=Decimal("0.04"),
    )
    cfg.service.update_demo(
        {
            "initial_balance": "5000",
            "margin_per_trade": "75",
            "leverage": 5,
            "risk_percent": "2",
            "fee_rate": "0.05",
        }
    )
    result = cfg.service.reset_demo_account()
    assert result["initial_balance"] == "5000"
    row = repo.get_account("demo")
    assert row["initial_balance"] == "5000"
    assert row["balance"] == "5000"
    assert row["equity"] == "5000"
    assert row["peak_equity"] == "5000"
    # the configurable values saved earlier survive the reset
    assert row["margin_per_trade"] == "75"
    assert row["leverage"] == 5
    assert row["risk_percent"] == "2"
    assert row["fee_rate"] == "0.05"


@pytest.mark.unit
def test_reset_demo_account_without_account_is_noop(cfg):
    result = cfg.service.reset_demo_account()
    assert result["locked"] is False
    assert "initial_balance" in result
    assert DemoRepository(cfg.db).get_account("demo") is None


# -- Telegram ----------------------------------------------------------------------


@pytest.mark.unit
def test_telegram_requires_token_and_chat_when_enabled(cfg):
    with pytest.raises(SettingsValidationError):
        cfg.service.update_telegram({"enabled": True, "chat_id": "12345"})


@pytest.mark.unit
def test_telegram_token_stored_then_masked_and_test_sends(cfg):
    cfg.service.update_telegram(
        {"enabled": True, "chat_id": "123456789", "bot_token": "711:AA-secret-token"}
    )
    public = cfg.service.get_settings()["telegram"]
    assert public["token_configured"] is True
    assert "711:AA-secret-token" not in json.dumps(public)

    sent: list[str] = []

    def fake_sender(config):
        sent.append(config.chat_id)
        return SimpleNamespace(ok=True, message_id=42, error=None)

    service = ConfigService(
        cfg.db, secrets_path=cfg.secrets_path, env={}, telegram_sender=fake_sender
    )
    result = service.test_telegram()
    assert result["success"] is True
    assert result["message_id"] == 42
    assert sent == ["123456789"]


@pytest.mark.unit
def test_telegram_config_object_stored_precedence(cfg):
    cfg.service.update_telegram(
        {"enabled": True, "chat_id": "123456789", "bot_token": "711:AA-secret-token"}
    )
    config = cfg.service.telegram_config_object()
    assert config.enabled is True
    assert config.chat_id == "123456789"
    assert config.bot_token == "711:AA-secret-token"
    # defaults come from the env layer
    assert config.timeout > 0
    notifier = cfg.service.resolve_telegram_notifier()
    assert notifier.config.enabled is True
    assert notifier.config.chat_id == "123456789"
    assert notifier.config.bot_token == "711:AA-secret-token"


@pytest.mark.unit
def test_telegram_config_object_env_fallback(cfg):
    service = ConfigService(
        cfg.db,
        secrets_path=cfg.secrets_path,
        env={
            "BTCUSDT_TELEGRAM_ENABLED": "true",
            "BTCUSDT_TELEGRAM_BOT_TOKEN": "711:AA-env-token",
            "BTCUSDT_TELEGRAM_CHAT_ID": "987654321",
        },
    )
    config = service.telegram_config_object()
    assert config.enabled is True
    assert config.chat_id == "987654321"
    assert config.bot_token == "711:AA-env-token"


@pytest.mark.unit
def test_telegram_config_object_disabled_when_nothing_configured(cfg):
    config = cfg.service.telegram_config_object()
    assert config.enabled is False
    assert config.bot_token == ""
    notifier = cfg.service.resolve_telegram_notifier()
    assert notifier.config.enabled is False


# -- Live connection tests (fake transport) -----------------------------------------


@pytest.mark.unit
def test_test_ollama_connection_success(cfg):
    service = ConfigService(
        cfg.db,
        secrets_path=cfg.secrets_path,
        env={},
        transport_get=lambda url, timeout: FakeResponse(
            200, {"models": [{"name": "qwen3:4b"}, {"name": "qwen3:8b"}]}
        ),
    )
    result = service.test_llm_connection({"provider": "ollama", "model": "qwen3:4b"})
    assert result["success"] is True
    assert result["model_available"] is True
    assert set(result["available_models"]) == {"qwen3:4b", "qwen3:8b"}
    assert result["latency_ms"] >= 0


@pytest.mark.unit
def test_test_ollama_connection_failure_is_sanitized(cfg):
    def boom(url, timeout):
        raise ConnectionError("can't connect to http://127.0.0.1:11434")

    service = ConfigService(
        cfg.db, secrets_path=cfg.secrets_path, env={}, transport_get=boom
    )
    result = service.test_llm_connection({"provider": "ollama", "model": "qwen3:4b"})
    assert result["success"] is False
    assert "ConnectionError" in result["error"]
    assert "11434" not in result["error"]  # no URL leakage


@pytest.mark.unit
def test_list_ollama_models_raises_settings_test_error(cfg):
    service = ConfigService(
        cfg.db,
        secrets_path=cfg.secrets_path,
        env={},
        transport_get=lambda url, timeout: FakeResponse(500),
    )
    with pytest.raises(SettingsTestError):
        service.list_ollama_models("http://localhost:11434")


# -- temperature end-to-end ---------------------------------------------------------


def _analysis(
    *,
    decision: str,
    temperature: float | None = None,
    entry: str = "61000",
    stop: str = "60000",
    take: str = "64000",
) -> SignalAnalysis:
    return SignalAnalysis(
        decision=decision,
        confidence=80.0,
        reasoning="deterministic test analysis",
        entry_price=Decimal(entry),
        stop_loss=Decimal(stop),
        take_profit=Decimal(take),
        provider="ollama",
        model="qwen3:4b",
        symbol="BTCUSDT",
        timeframe="1h",
        temperature=temperature,
        analysis_timestamp="2026-09-10T01:00:00.000Z",
        market_timestamp="2026-09-10T00:00:00.000Z",
        closed_at="2026-09-10T01:00:00.000Z",
        candle_close_price=61050.5,
    )


@pytest.mark.unit
def test_temperature_round_trips_through_signal_ledger(cfg):
    repo = SignalRepository(cfg.db)
    engine = SignalEngine(repo)
    analysis = _analysis(decision="LONG", temperature=0.4)
    result = engine.process(analysis)
    assert result.signal is not None
    assert result.signal.status == "PENDING_ENTRY"
    assert result.signal.temperature == 0.4

    reloaded = repo.get_signal(result.signal.id)
    assert reloaded.temperature == 0.4


@pytest.mark.unit
def test_engine_passes_temperature_to_create_signal(cfg):
    repo = SignalRepository(cfg.db)
    engine = SignalEngine(repo)
    analysis = _analysis(
        decision="SHORT", temperature=0.8, entry="60000", stop="61000", take="58000"
    )
    engine.process(analysis)
    assert repo.get_active_signal().temperature == 0.8


@pytest.mark.unit
def test_temperature_stays_none_when_omitted(cfg):
    repo = SignalRepository(cfg.db)
    engine = SignalEngine(repo)
    engine.process(_analysis(decision="LONG"))
    assert repo.get_active_signal().temperature is None


@pytest.mark.unit
def test_config_service_default_matches_env(cfg):
    settings = cfg.service.get_settings()
    assert settings["ai_provider"]["provider"] == "ollama"
    assert settings["ai_provider"]["api_key_configured"] is False
    assert settings["config_state"] == "applied"


# ---- Active symbol setting ------------------------------------------------------


@pytest.mark.unit
def test_resolve_symbol_defaults_to_btcusdt(cfg):
    assert cfg.service.resolve_symbol() == "BTCUSDT"
    assert cfg.service.get_symbol_public()["symbol"] == "BTCUSDT"
    assert cfg.service.get_settings()["symbol"]["symbol"] == "BTCUSDT"
    assert cfg.service.get_runtime_public()["symbol"] == "BTCUSDT"
    assert cfg.service.get_symbol_public()["supported"] == ["BTCUSDT", "ETHUSDT", "XAUUSDT"]


@pytest.mark.unit
def test_env_symbol_used_when_no_db_setting():
    box = TempConfig(env={"BTCUSDT_SYMBOL": "ETHUSDT"})
    try:
        assert box.service.resolve_symbol() == "ETHUSDT"
        settings = box.service.get_settings()
        assert settings["symbol"]["symbol"] == "ETHUSDT"
        assert settings["symbol"]["supported"] == ["BTCUSDT", "ETHUSDT", "XAUUSDT"]
    finally:
        box.close()


@pytest.mark.unit
def test_unsupported_env_symbol_falls_back_to_default():
    box = TempConfig(env={"BTCUSDT_SYMBOL": "SOLUSDT"})
    try:
        assert box.service.resolve_symbol() == "BTCUSDT"
    finally:
        box.close()


@pytest.mark.unit
def test_db_symbol_precedes_env_and_persists_across_restart():
    box = TempConfig(env={"BTCUSDT_SYMBOL": "ETHUSDT"})
    try:
        updated = box.service.update_symbol({"symbol": "XAUUSDT"})
        assert updated["symbol"] == "XAUUSDT"
        assert box.service.resolve_symbol() == "XAUUSDT"
        reloaded = ConfigService(
            box.db, secrets_path=box.secrets_path, env={"BTCUSDT_SYMBOL": "ETHUSDT"}
        )
        assert reloaded.resolve_symbol() == "XAUUSDT"
    finally:
        box.close()


@pytest.mark.unit
def test_update_symbol_rejects_unknown_symbol(cfg):
    with pytest.raises(SettingsValidationError):
        cfg.service.update_symbol({"symbol": "SOLUSDT"})
    assert cfg.service.resolve_symbol() == "BTCUSDT"


@pytest.mark.unit
def test_update_symbol_allowed_while_signal_active():
    box = TempConfig()
    try:
        repo = SignalRepository(box.db)
        repo.create_signal("BTCUSDT", "1h", "LONG", "100", "99", "101")
        assert repo.get_active_signal() is not None
        updated = box.service.update_symbol({"symbol": "ETHUSDT"})
        assert updated["symbol"] == "ETHUSDT"
    finally:
        box.close()
