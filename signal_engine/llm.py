"""LLM provider configuration for the BTCUSDT signal pipeline.

This is a thin, configuration-only layer on top of TradingAgents' existing
provider layer (``tradingagents.llm_clients``). It deliberately does NOT
re-implement provider clients, API-key handling, the provider factory, or the
capability dispatch — all of those stay in the shared provider layer, which
already supports DeepSeek and Ollama (OpenAI-compatible).

It adds only:

* a curated registry of the signal pipeline's supported models (DeepSeek V4
  Flash, Qwen3 8B, Qwen3 4B via Ollama), keyed so the pipeline can switch
  models by configuration instead of by editing code;
* an environment-driven loader (``BTCUSDT_LLM_*``) that resolves a provider +
  model pair and optional request settings already supported by the shared
  architecture (base URL, timeout, retries, temperature, output-token cap);
* eager validation that fails with a clear error as soon as the *selected*
  provider is missing what it needs (e.g. ``DEEPSEEK_API_KEY``), without
  touching the other provider's configuration.

Credentials are never defined or logged here. Each provider's key is read from
its existing environment variable (``tradingagents.llm_clients.api_key_env``);
Ollama requires no key. All resolved values are handed to the shared factory
untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tradingagents.llm_clients import BaseLLMClient, create_llm_client

# Environment variable names for the signal pipeline's LLM configuration.
# Selecting a model or provider is purely configuration-driven; application
# code does not need to change to switch between DeepSeek and Ollama.
ENV_LLM_PROVIDER = "BTCUSDT_LLM_PROVIDER"
ENV_LLM_MODEL = "BTCUSDT_LLM_MODEL"
ENV_LLM_BASE_URL = "BTCUSDT_LLM_BASE_URL"
ENV_LLM_TIMEOUT = "BTCUSDT_LLM_TIMEOUT"
ENV_LLM_MAX_RETRIES = "BTCUSDT_LLM_MAX_RETRIES"
ENV_LLM_TEMPERATURE = "BTCUSDT_LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "BTCUSDT_LLM_MAX_TOKENS"

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"

# Default model per provider when only ``BTCUSDT_LLM_PROVIDER`` is set.
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "deepseek": "deepseek-v4-flash",
    "ollama": "qwen3:8b",
}


class LLMConfigError(ValueError):
    """Invalid or unusable LLM configuration for the signal pipeline."""


@dataclass(frozen=True)
class ModelSpec:
    """One curated model in the signal pipeline's registry."""

    key: str
    label: str
    provider: str
    model: str


SIGNAL_MODELS: dict[str, ModelSpec] = {
    "deepseek-v4-flash": ModelSpec(
        key="deepseek-v4-flash",
        label="DeepSeek V4 Flash",
        provider="deepseek",
        model="deepseek-v4-flash",
    ),
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        label="Qwen3 8B via Ollama",
        provider="ollama",
        model="qwen3:8b",
    ),
    "qwen3-4b": ModelSpec(
        key="qwen3-4b",
        label="Qwen3 4B via Ollama",
        provider="ollama",
        model="qwen3:4b",
    ),
    "gemma3-4b": ModelSpec(
        key="gemma3-4b",
        label="Gemma 4B via Ollama",
        provider="ollama",
        model="gemma3:4b",
    ),
}

KNOWN_MODEL_KEYS: tuple[str, ...] = tuple(SIGNAL_MODELS)


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM configuration for one signal-pipeline run.

    ``provider`` and ``model`` are the only required fields. The remaining
    settings are optional refinements; ``None`` leaves the shared provider
    client at its own default, so an empty configuration always produces the
    pipeline's default model (DeepSeek V4 Flash).

    ``api_key`` is optional and only used by the Phase 15 Settings page: when
    set, it is forwarded to the shared factory (``api_key`` is a passthrough
    kwarg); when ``None`` the provider layer reads its own environment-variable
    credential exactly as before. It is never echoed anywhere.
    """

    provider: str
    model: str
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None


def get_model_spec(model_key: str) -> ModelSpec:
    """Return the curated ``ModelSpec`` for a signal model key."""
    try:
        return SIGNAL_MODELS[model_key]
    except KeyError:
        raise LLMConfigError(
            f"Unknown signal model key: {model_key!r}. "
            f"Known keys: {', '.join(KNOWN_MODEL_KEYS)}."
        ) from None


def _env_float(env_var: str, raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        # The offending value is deliberately NOT echoed: an env var may hold
        # a credential by mistake, and it must never surface in an error.
        raise LLMConfigError(f"Invalid value for {env_var}: expected a number.") from exc


def _env_int(env_var: str, raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMConfigError(f"Invalid value for {env_var}: expected an integer.") from exc


def _resolve_provider_model(env: dict[str, Any]) -> tuple[str, str]:
    """Resolve a (provider, model) pair from env vars and the model registry.

    Each lookup is case-insensitive for the provider name; model ids are used
    verbatim. Rules:

    * a known model key with no explicit provider -> the registry's provider
      and canonical model id (e.g. ``qwen3-8b`` -> ollama / ``qwen3:8b``);
    * a known model key with an explicit provider -> the explicit provider wins,
      model id stays the registry's canonical id;
    * anything else -> the raw ``BTCUSDT_LLM_MODEL`` as model id (or the
      provider default), with ``BTCUSDT_LLM_PROVIDER`` or ``DEFAULT_PROVIDER``.
    """
    provider = (env.get(ENV_LLM_PROVIDER) or "").strip().lower() or None
    model_key = (env.get(ENV_LLM_MODEL) or "").strip() or None

    spec = SIGNAL_MODELS.get(model_key) if model_key else None
    if spec is not None and provider is None:
        return spec.provider, spec.model
    if spec is not None:
        return provider, spec.model  # type: ignore[return-value]

    provider = provider or DEFAULT_PROVIDER
    model = model_key or PROVIDER_DEFAULT_MODEL.get(provider, DEFAULT_MODEL)
    return provider, model


def llm_config_from_env(env: dict[str, Any] | None = None) -> LLMConfig:
    """Build an ``LLMConfig`` from the ``BTCUSDT_LLM_*`` environment variables.

    ``env`` defaults to ``os.environ``; passing a dict is useful for tests and
    for programmatic injection. Missing/empty variables fall back to the
    pipeline default (DeepSeek V4 Flash), so the pipeline runs out of the box
    once the selected provider's credential env var is set.
    """
    if env is None:
        env = os.environ
    provider, model = _resolve_provider_model(env)

    base_url = (env.get(ENV_LLM_BASE_URL) or "").strip() or None
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        timeout=_env_float(ENV_LLM_TIMEOUT, env.get(ENV_LLM_TIMEOUT)),
        max_retries=_env_int(ENV_LLM_MAX_RETRIES, env.get(ENV_LLM_MAX_RETRIES)),
        temperature=_env_float(ENV_LLM_TEMPERATURE, env.get(ENV_LLM_TEMPERATURE)),
        max_tokens=_env_int(ENV_LLM_MAX_TOKENS, env.get(ENV_LLM_MAX_TOKENS)),
    )


def build_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Construct the configured LLM client via the shared TradingAgents factory.

    This is the pipeline's single touch point against the provider layer. The
    client's ``get_llm()`` is invoked once eagerly so that missing credentials
    or a missing required base URL surface here (early, at startup) with the
    provider layer's own clear error — and without any network I/O.
    """
    kwargs: dict[str, Any] = {
        "base_url": config.base_url,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    try:
        client = create_llm_client(config.provider, config.model, **kwargs)
    except (ValueError, TypeError) as exc:
        raise LLMConfigError(str(exc)) from exc
    try:
        client.get_llm()
    except (ValueError, TypeError) as exc:
        raise LLMConfigError(str(exc)) from exc
    return client


def validate_llm_config(config: LLMConfig) -> None:
    """Validate an ``LLMConfig`` without calling any LLM API.

    Raises ``LLMConfigError`` when the selected provider is missing what it
    needs (e.g. ``DEEPSEEK_API_KEY`` for DeepSeek) or the provider/model is
    unsupported. Configuration for unused providers is never inspected, so
    using Ollama never requires DeepSeek credentials and vice versa.
    """
    build_llm_client(config)
