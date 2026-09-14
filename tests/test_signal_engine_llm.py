"""Phase 4 unit tests: signal-pipeline LLM provider configuration.

These tests exercise the thin configuration layer under ``signal_engine/llm.py``
— model registry, environment resolution, eager validation, and the forward to
the existing TradingAgents provider factory. No LLM API is ever called: client
construction is synchronous and local; no ``invoke()``/network path runs in
these tests.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from signal_engine import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENV_LLM_BASE_URL,
    ENV_LLM_MAX_RETRIES,
    ENV_LLM_MAX_TOKENS,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    ENV_LLM_TIMEOUT,
    SIGNAL_MODELS,
    LLMConfig,
    LLMConfigError,
    build_llm_client,
    get_model_spec,
    llm_config_from_env,
    validate_llm_config,
)
from tradingagents.llm_clients.base_client import BaseLLMClient
from tradingagents.llm_clients.factory import create_llm_client


def _ollama_compatible_classes():
    """Resolve the chat classes from the live client module.

    Other test modules (e.g. test_ollama_base_url) ``importlib.reload`` the
    openai_client module, so class objects imported at module scope here can
    go stale and break ``isinstance`` in a full-suite run. Reading them back
    from ``sys.modules`` at call time keeps the assertions correct under any
    ordering.
    """
    import tradingagents.llm_clients.openai_client as mod

    return mod.DeepSeekChatOpenAI, mod.NormalizedChatOpenAI


@pytest.mark.unit
class TestModelRegistry(unittest.TestCase):
    def test_all_required_models_registered(self):
        self.assertEqual(
            set(SIGNAL_MODELS),
            {"deepseek-v4-flash", "qwen3-8b", "qwen3-4b", "gemma3-4b"},
        )

    def test_deepseek_spec(self):
        spec = get_model_spec("deepseek-v4-flash")
        self.assertEqual(spec.provider, "deepseek")
        self.assertEqual(spec.model, "deepseek-v4-flash")

    def test_qwen3_8b_uses_ollama(self):
        spec = get_model_spec("qwen3-8b")
        self.assertEqual(spec.provider, "ollama")
        self.assertEqual(spec.model, "qwen3:8b")

    def test_qwen3_4b_uses_ollama(self):
        spec = get_model_spec("qwen3-4b")
        self.assertEqual(spec.provider, "ollama")
        self.assertEqual(spec.model, "qwen3:4b")

    def test_unknown_model_key_raises(self):
        with self.assertRaises(LLMConfigError) as ctx:
            get_model_spec("does-not-exist")
        self.assertIn("does-not-exist", str(ctx.exception))
        self.assertIn("deepseek-v4-flash", str(ctx.exception))


@pytest.mark.unit
class TestEnvResolution(unittest.TestCase):
    def test_defaults_with_no_env(self):
        config = llm_config_from_env({})
        self.assertEqual(config.provider, DEFAULT_PROVIDER)
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertIsNone(config.base_url)

    def test_select_qwen3_8b_by_model_key(self):
        config = llm_config_from_env({ENV_LLM_MODEL: "qwen3-8b"})
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "qwen3:8b")

    def test_select_qwen3_4b_by_model_key(self):
        config = llm_config_from_env({ENV_LLM_MODEL: "qwen3-4b"})
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "qwen3:4b")

    def test_provider_default_model(self):
        config = llm_config_from_env({ENV_LLM_PROVIDER: "ollama"})
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "qwen3:8b")

    def test_switching_models_needs_no_code_change(self):
        for model_key, expected_model in (
            ("deepseek-v4-flash", "deepseek-v4-flash"),
            ("qwen3-8b", "qwen3:8b"),
            ("qwen3-4b", "qwen3:4b"),
        ):
            config = llm_config_from_env({ENV_LLM_MODEL: model_key})
            self.assertEqual(config.model, expected_model)

    def test_explicit_provider_overrides_registry_provider(self):
        config = llm_config_from_env({ENV_LLM_PROVIDER: "ollama", ENV_LLM_MODEL: "deepseek-v4-flash"})
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "deepseek-v4-flash")

    def test_custom_model_via_compatible_endpoint(self):
        config = llm_config_from_env(
            {
                ENV_LLM_PROVIDER: "openai_compatible",
                ENV_LLM_MODEL: "my-local-model",
                ENV_LLM_BASE_URL: "http://localhost:8000/v1",
            }
        )
        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.model, "my-local-model")
        self.assertEqual(config.base_url, "http://localhost:8000/v1")

    def test_provider_is_case_insensitive(self):
        config = llm_config_from_env({ENV_LLM_PROVIDER: "OLLAMA"})
        self.assertEqual(config.provider, "ollama")

    def test_empty_values_fall_back_to_defaults(self):
        config = llm_config_from_env({ENV_LLM_PROVIDER: "", ENV_LLM_MODEL: "  "})
        self.assertEqual(config.provider, DEFAULT_PROVIDER)
        self.assertEqual(config.model, DEFAULT_MODEL)

    def test_direct_construction(self):
        config = LLMConfig(provider="deepseek", model="deepseek-v4-flash")
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-v4-flash")


@pytest.mark.unit
class TestEnvCoercion(unittest.TestCase):
    def test_numeric_settings_parsed(self):
        config = llm_config_from_env(
            {
                ENV_LLM_MAX_RETRIES: "4",
                ENV_LLM_MAX_TOKENS: "8192",
                ENV_LLM_TIMEOUT: "30",
            }
        )
        self.assertEqual(config.max_retries, 4)
        self.assertEqual(config.max_tokens, 8192)
        self.assertIsInstance(config.timeout, float)
        self.assertEqual(config.timeout, 30.0)

    def test_bad_max_retries_raises_with_var_name(self):
        with self.assertRaises(LLMConfigError) as ctx:
            llm_config_from_env({ENV_LLM_MAX_RETRIES: "many"})
        self.assertIn(ENV_LLM_MAX_RETRIES, str(ctx.exception))

    def test_bad_timeout_raises(self):
        with self.assertRaises(LLMConfigError):
            llm_config_from_env({ENV_LLM_TIMEOUT: "30s"})


@pytest.mark.unit
class TestBuildClient(unittest.TestCase):
    def _deepseek_config(self):
        return LLMConfig(provider="deepseek", model="deepseek-v4-flash")

    def test_returns_shared_client_abstraction(self):
        client = build_llm_client(self._deepseek_config())
        self.assertIsInstance(client, BaseLLMClient)

    def test_deepseek_uses_deepseek_chat_openai(self):
        deepseek_cls, _ = _ollama_compatible_classes()
        llm = build_llm_client(self._deepseek_config()).get_llm()
        self.assertIsInstance(llm, deepseek_cls)
        self.assertEqual(llm.model_name, "deepseek-v4-flash")
        self.assertEqual(llm.openai_api_base, "https://api.deepseek.com")

    def test_ollama_uses_normalized_chat_openai(self):
        _, normalized_cls = _ollama_compatible_classes()
        config = LLMConfig(provider="ollama", model="qwen3:8b")
        llm = build_llm_client(config).get_llm()
        self.assertIsInstance(llm, normalized_cls)
        self.assertEqual(llm.model_name, "qwen3:8b")

    def test_ollama_default_base_url(self):
        llm = build_llm_client(LLMConfig(provider="ollama", model="qwen3:8b")).get_llm()
        self.assertEqual(llm.openai_api_base, "http://localhost:11434/v1")

    def test_ollama_respects_base_url_override(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://remote-ollama:11434/v1"}):
            llm = build_llm_client(LLMConfig(provider="ollama", model="qwen3:8b")).get_llm()
            self.assertEqual(llm.openai_api_base, "http://remote-ollama:11434/v1")

    def test_explicit_base_url_wins(self):
        config = LLMConfig(provider="ollama", model="qwen3:8b", base_url="http://explicit:11434/v1")
        llm = build_llm_client(config).get_llm()
        self.assertEqual(llm.openai_api_base, "http://explicit:11434/v1")

    def test_timeout_retries_tokens_forwarded(self):
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-v4-flash",
            max_retries=4,
            max_tokens=2048,
            temperature=0.0,
        )
        llm = build_llm_client(config).get_llm()
        self.assertEqual(llm.max_retries, 4)
        self.assertEqual(llm.max_tokens, 2048)
        self.assertEqual(llm.temperature, 0.0)

    def test_matches_direct_factory_output(self):
        """Backward compatibility: the wrapper and the shared factory produce
        the same configured LLM for identical inputs."""
        config = LLMConfig(provider="deepseek", model="deepseek-v4-flash")
        wrapped = build_llm_client(config).get_llm()
        direct = create_llm_client("deepseek", "deepseek-v4-flash", base_url=None)
        direct_llm = direct.get_llm()
        self.assertIsInstance(wrapped, type(direct_llm))
        self.assertEqual(wrapped.model_name, direct_llm.model_name)
        self.assertEqual(wrapped.openai_api_base, direct_llm.openai_api_base)

    def test_llm_is_chat_openai_compatible(self):
        llm = build_llm_client(self._deepseek_config()).get_llm()
        self.assertIsInstance(llm, ChatOpenAI)


@pytest.mark.unit
class TestEarlyValidation(unittest.TestCase):
    def test_missing_deepseek_key_raises_clear_error(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), self.assertRaises(LLMConfigError) as ctx:
            validate_llm_config(LLMConfig(provider="deepseek", model="deepseek-v4-flash"))
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))
        self.assertIn("not set", str(ctx.exception))

    def test_ollama_needs_no_deepseek_credentials(self):
        _, normalized_cls = _ollama_compatible_classes()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            llm = build_llm_client(LLMConfig(provider="ollama", model="qwen3:8b")).get_llm()
            self.assertIsInstance(llm, normalized_cls)

    def test_deepseek_needs_no_ollama_config(self):
        deepseek_cls, _ = _ollama_compatible_classes()
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": ""}):
            llm = build_llm_client(LLMConfig(provider="deepseek", model="deepseek-v4-flash")).get_llm()
            self.assertIsInstance(llm, deepseek_cls)

    def test_unknown_provider_raises_clear_error(self):
        with self.assertRaises(LLMConfigError) as ctx:
            build_llm_client(LLMConfig(provider="not-a-provider", model="x"))
        self.assertIn("Unsupported LLM provider", str(ctx.exception))

    def test_validate_passes_for_ollama(self):
        validate_llm_config(LLMConfig(provider="ollama", model="qwen3:8b"))


SECRET = "sk-super-secret-9876543210"


@pytest.mark.unit
class TestNoSecretLeakage(unittest.TestCase):
    def test_coercion_error_never_echoes_value(self):
        with self.assertRaises(LLMConfigError) as ctx:
            llm_config_from_env({ENV_LLM_MAX_RETRIES: SECRET})
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertIn(ENV_LLM_MAX_RETRIES, str(ctx.exception))

    def test_no_secret_in_stdout_or_logs(self):
        import contextlib
        import io
        import logging

        class _Capture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []

            def emit(self, record):
                self.records.append(self.getMessage(record))

        handler = _Capture()
        root = logging.getLogger()
        out, err = io.StringIO(), io.StringIO()
        root.addHandler(handler)
        try:
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": SECRET}), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err), self.assertRaises(LLMConfigError):
                validate_llm_config(LLMConfig(provider="not-a-provider", model="x"))
            stream = "".join(handler.records) + out.getvalue() + err.getvalue()
            self.assertNotIn(SECRET, stream)
        finally:
            root.removeHandler(handler)

    def test_validation_error_mentions_env_var_not_value(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "OLLAMA_BASE_URL": f"http://{SECRET}/v1"}), \
                self.assertRaises(LLMConfigError) as ctx:
            validate_llm_config(LLMConfig(provider="deepseek", model="deepseek-v4-flash"))
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))
