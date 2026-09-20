"""Structured LLM analysis of the BTCUSDT closed-candle context.

Phase 5: this module turns the deterministic :class:`AnalysisContext` into a
strict LONG / SHORT / WAIT decision by calling a TradingAgents-provider LLM.

Strictness contract
-------------------
- Structured output is used when the provider supports it, otherwise a strict
  JSON object matching the pydantic schema is required. There is **no** free-text
  fallback: malformed or unparsable output raises :class:`SignalAnalysisError`
  and never produces a signal.
- WAIT must not propose entry/stop-loss/take-profit prices.
- An LLM failure raises :class:`SignalAnalysisError`; it can never silently
  degrade into LONG or SHORT.
- The analysis is purely in-memory (Phase 6 persists and validates it): this
  module never writes to the database, never calls the Binance API, and never
  invokes TradingAgents' stock/news/social/fundamental dataflows.

The TradingAgents ``structured`` helper lives under ``tradingagents.agents``,
whose package import would pull in the full stock pipeline (yfinance + vendor
data tools). To keep this integration free of those dataflows, the tiny binding
helper is re-implemented here with the same semantics instead of imported.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_core import ValidationError

from binance.market_data import Candle

from .context import (
    DEFAULT_MAX_CANDLES,
    AnalysisContext,
    build_analysis_context,
)
from .llm import LLMConfig, build_llm_client

logger = logging.getLogger(__name__)

DECISIONS = ("LONG", "SHORT", "WAIT")
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 100.0

# Same constraint the TradingAgents agents put in their prompts: the model may
# only reason from the supplied evidence, never reach for external tools.
_NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)

# LLMs sometimes write a placeholder string into an optional numeric field.
# Coerce those to None (mirroring the TradingAgents schemas convention).
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value: Any) -> Any:
    """Normalise an LLM-written optional price field before pydantic validates it."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    return cleaned or None


class SignalDecision(str, Enum):
    """The only legal decisions the signal pipeline accepts."""

    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


class SignalDecisionModel(BaseModel):
    """Strict structured-output contract for a single BTCUSDT 1H analysis.

    Field descriptions double as the model's output instructions for providers
    that use schema-driven structured output.
    """

    decision: SignalDecision = Field(
        description="Exactly one of LONG, SHORT, or WAIT. "
        "LONG = the supplied 1H closed-candle evidence supports a long entry; "
        "SHORT = it supports a short entry; WAIT = the evidence is unclear, "
        "conflicting, or insufficient."
    )
    confidence: float = Field(
        ge=CONFIDENCE_MIN,
        le=CONFIDENCE_MAX,
        description=f"How strongly the supplied data supports the decision, "
        f"from {CONFIDENCE_MIN:.0f} (no conviction) to {CONFIDENCE_MAX:.0f}"
        f" (maximum conviction).",
    )
    reasoning: str = Field(
        min_length=1,
        description="Reasoning based ONLY on the closed candles and indicators "
        "supplied in this prompt. Never invent prices, levels, news, or data.",
    )
    entry_price: float | None = Field(
        default=None,
        gt=0,
        description="Absolute price in USDT for an indicative entry. "
        "Only for LONG/SHORT; must be null for WAIT. Not a guarantee of fill.",
    )
    stop_loss: float | None = Field(
        default=None,
        gt=0,
        description="Absolute price in USDT for an indicative stop loss. "
        "Only for LONG/SHORT; must be null for WAIT.",
    )
    take_profit: float | None = Field(
        default=None,
        gt=0,
        description="Absolute price in USDT for an indicative take profit. "
        "Only for LONG/SHORT; must be null for WAIT.",
    )

    @field_validator("entry_price", "stop_loss", "take_profit", mode="before")
    @classmethod
    def _coerce_price(cls, value: Any) -> Any:
        return _coerce_optional_float(value)


SYSTEM_PROMPT = (
    "You are a technical analyst for the BTCUSDT pair on Binance USDT-M "
    "Futures, analysing the 1H timeframe.\n"
    "The data you receive consists ONLY of closed 1H candles and indicators "
    "that were computed from those closed candles by deterministic Python "
    "code. You have no news, no social-media sentiment, no fundamentals, no "
    "insider information, and no real-time access beyond this context.\n"
    "Rules:\n"
    "1. Decide exactly one of LONG, SHORT, or WAIT.\n"
    "2. confidence must be a number from 0 to 100 expressing how strongly the "
    "supplied data supports your decision (100 = maximum conviction).\n"
    "3. For LONG/SHORT you may propose entry_price, stop_loss, and take_profit "
    "as absolute USDT prices. These are only indicative planning levels, never "
    "guaranteed fills. For WAIT they must be null.\n"
    "4. Base every statement on the supplied market data and indicators. Never "
    "invent prices, levels, or data.\n"
    + _NO_EXTERNAL_TOOLS
)


def build_prompt(context: AnalysisContext) -> list[dict[str, str]]:
    """Build the chat messages handed to the LLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context.rendered},
    ]


class SignalAnalysisError(Exception):
    """The LLM analysis failed or produced output that cannot be a signal."""


class RateLimitedError(SignalAnalysisError):
    """The LLM provider refused the call with a rate limit (HTTP 429 etc.).

    Retrying the same candle on every poll would deepen the limit, so the
    scheduler skips the candle instead of retrying it.
    """


#: Lowercased substrings identifying a provider rate-limit / capacity error.
#: The LLM layer collapses transport errors to message strings (no status
#: code), so classification is message-based by necessity.
_RATE_LIMIT_HINTS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "capacity",
    "overloaded",
)


def _is_rate_limited(text: str) -> bool:
    """Return True when an LLM failure message looks like a rate limit."""
    lowered = text.lower()
    return any(hint in lowered for hint in _RATE_LIMIT_HINTS)


def _reset_hint(text: str) -> str:
    """Extract a provider retry/reset hint (e.g. retry-after seconds)."""
    match = re.search(
        r"(?:retry[\s_-]?after|reset(?:\s+in)?|retry in)\s*[:=]?\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return f" (provider asks to retry after {match.group(1)}s)"
    return ""


def _bind_structured(llm: Any, schema: type[BaseModel], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` when unsupported.

    Semantics mirror TradingAgents' ``bind_structured``; the optional third
    argument ``typed`` / other provider quirks are delegated to the LLM itself.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "using a strict JSON contract instead",
            agent_name, exc,
        )
        return None


def _bind_structured_raw(
    llm: Any, schema: type[BaseModel], agent_name: str
) -> Any | None:
    """Return ``with_structured_output(schema, include_raw=True)`` or None.

    The raw envelope keeps the provider's usage metadata that the parsed-only
    path drops (this is why token columns are NULL on OpenRouter free). Any
    provider quirk (unexpected kwarg, unimplemented, missing method) falls
    back to the parsed-only binding instead of raising.
    """
    try:
        return llm.with_structured_output(schema, include_raw=True)
    except (NotImplementedError, AttributeError, TypeError) as exc:
        logger.debug(
            "%s: provider does not support include_raw (%s); "
            "usage metadata will be estimated",
            agent_name, exc,
        )
        return None


def _prompt_text(prompt: Any) -> str:
    """Flatten a chat prompt (list of message dicts or plain str) to text."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(getattr(message, "content", message)))
        return "\n".join(parts)
    return str(prompt)


def _estimate_usage(prompt_text: str, parsed: Any) -> dict[str, int]:
    """Rough token estimate (~4 chars/token) when the provider reports none.

    Always labeled estimated downstream (``~`` in the UI, ``tokens_estimated``
    in storage): better than NULL for quota review, never confused with a
    metered count.
    """
    try:
        rendered = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else str(parsed)
    except Exception:
        rendered = ""
    prompt_tokens = max(1, len(prompt_text) // 4)
    completion_tokens = max(1, len(rendered) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _call_structured(
    structured: Any,
    structured_raw: Any,
    prompt: Any,
    agent_name: str,
) -> tuple[Any, dict[str, int | None] | None]:
    """Invoke structured output, preferring the raw envelope for usage.

    Returns ``(content, usage)`` with the provider count (or None when the
    envelope carries none — callers apply the shared estimate fallback).
    A non-dict result (e.g. test doubles bound without ``include_raw``)
    degrades to the parsed-only path. Raises are left to the caller, which
    already maps rate limits and failures.
    """
    if structured_raw is not None:
        result = structured_raw.invoke(prompt)
        if isinstance(result, dict) and result.get("parsed") is not None:
            raw_message = result.get("raw")
            return result["parsed"], _extract_usage(raw_message)
        if isinstance(result, dict) and result.get("parsing_error") is not None:
            raise SignalAnalysisError(
                f"{agent_name} structured-output parsing failed: "
                f"{result.get('parsing_error')}"
            )
        if result is None:
            raise SignalAnalysisError(
                f"{agent_name} structured-output invocation returned no parsed result"
            )
        # Unexpected shape: fall through to the parsed-only binding.
    if structured is None:
        raise SignalAnalysisError(
            f"{agent_name} has no structured-output binding"
        )
    content = structured.invoke(prompt)
    if content is None:
        raise SignalAnalysisError(
            f"{agent_name} structured-output invocation returned no parsed result"
        )
    return content, None


def _validate_decision_rules(model: SignalDecisionModel) -> SignalDecisionModel:
    """Enforce cross-field decision rules that pydantic cannot express."""
    if model.decision == SignalDecision.WAIT:
        proposed = {
            "entry_price": model.entry_price,
            "stop_loss": model.stop_loss,
            "take_profit": model.take_profit,
        }
        non_null = [key for key, value in proposed.items() if value is not None]
        if non_null:
            raise SignalAnalysisError(
                "WAIT must not propose price levels: "
                + ", ".join(sorted(non_null))
            )
    return model


def _coerce_decision(raw: Any) -> SignalDecisionModel:
    """Parse raw LLM output into a valid SignalDecisionModel or raise."""
    try:
        if isinstance(raw, SignalDecisionModel):
            model = raw
        elif isinstance(raw, dict):
            model = SignalDecisionModel.model_validate(raw)
        elif isinstance(raw, str):
            text = raw.strip()
            if not text:
                raise SignalAnalysisError("LLM returned empty output")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SignalAnalysisError(
                    "LLM returned invalid JSON instead of the required schema"
                ) from exc
            if not isinstance(data, dict):
                raise SignalAnalysisError(
                    "LLM returned a JSON value that is not an object"
                )
            model = SignalDecisionModel.model_validate(data)
        else:
            raise SignalAnalysisError(
                f"LLM returned an unsupported output type: {type(raw).__name__}"
            )
    except ValidationError as exc:
        raise SignalAnalysisError(f"malformed signal analysis output: {exc}") from exc
    return _validate_decision_rules(model)


@dataclass(frozen=True)
class SignalAnalysis:
    """In-memory result of one AI analysis. Not persisted or opened in Phase 5."""

    decision: str
    confidence: float
    reasoning: str
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    provider: str
    model: str
    symbol: str
    timeframe: str
    analysis_timestamp: str
    market_timestamp: str
    closed_at: str
    candle_close_price: float
    temperature: float | None = None
    #: ATR(14) of the analysis candle, threaded from the deterministic
    #: indicator snapshot so Phase 6 guardrails can bound stop/take distances.
    atr: float | None = None
    #: Quota audit: LLM calls spent producing this analysis (1 single-call,
    #: up to 5 for the multi-agent debate) plus best-effort token counts
    #: (NULL when the provider path drops the usage metadata).
    llm_calls: int = 1
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: Live Binance funding rate (/8h) at analysis time, threaded from the
    #: Phase P1 positioning snapshot so the crowded-side guardrail can reject
    #: entries leaning into an overcrowded side. None = unavailable = no check.
    funding_rate: float | None = None
    #: 4H regime ("UP"/"DOWN") threaded from the Phase P2 HTF bias so the
    #: regime guardrail can veto counter-trend entries. None = unclear or
    #: unavailable = no check.
    regime: str | None = None
    #: True when the token counts are a ~4-chars-per-token estimate (the
    #: provider dropped the usage metadata). The UI prefixes such counts
    #: with "~" so they are never confused with metered counts.
    tokens_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "provider": self.provider,
            "model": self.model,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "analysis_timestamp": self.analysis_timestamp,
            "market_timestamp": self.market_timestamp,
            "closed_at": self.closed_at,
            "candle_close_price": self.candle_close_price,
            "temperature": self.temperature,
            "atr": self.atr,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "funding_rate": self.funding_rate,
            "regime": self.regime,
            "tokens_estimated": self.tokens_estimated,
        }


def _snapshot_atr(context: AnalysisContext) -> float | None:
    """Extract a finite ATR(14) from the analysis-candle snapshot, if any."""
    try:
        value = float(context.snapshot.atr14)
    except (TypeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _extract_usage(response: Any) -> dict[str, int | None] | None:
    """Best-effort token-usage extraction from a raw LLM response.

    Provider metadata shapes differ (``response_metadata.token_usage`` on
    OpenAI-compatible paths, ``usage_metadata`` elsewhere) and the
    structured-output path drops the envelope entirely. Anything missing or
    malformed yields NULL quota columns — never an exception.
    """
    try:
        meta = getattr(response, "response_metadata", None) or {}
        usage = dict(meta.get("token_usage") or {})
        alt = getattr(response, "usage_metadata", None) or {}

        def _pick(*keys: str) -> int | None:
            for key in keys:
                for source in (usage, alt):
                    value = source.get(key)
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, int) and value >= 0:
                        return value
            return None

        prompt = _pick("prompt_tokens", "input_tokens")
        completion = _pick("completion_tokens", "output_tokens")
        total = _pick("total_tokens")
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        if prompt is None and completion is None and total is None:
            return None
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
    except Exception:
        return None


class SignalAnalyzer:
    """Runs the TradingAgents LLM against a closed-candle context."""

    def __init__(self, config: LLMConfig, llm: Any | None = None) -> None:
        self.config = config
        self.llm = llm if llm is not None else build_llm_client(config).get_llm()
        self.structured_llm = _bind_structured(
            self.llm, SignalDecisionModel, "BTCUSDT Signal Analyzer"
        )
        #: Raw-envelope variant (include_raw) for usage metadata; None when
        #: the provider cannot bind it (parsed-only path + estimate then).
        self.structured_llm_raw = _bind_structured_raw(
            self.llm, SignalDecisionModel, "BTCUSDT Signal Analyzer"
        )

    def analyze(
        self,
        candles: list[Candle],
        indicators,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = DEFAULT_MAX_CANDLES,
        event_note: str | None = None,
        positioning: str | None = None,
        funding_rate: float | None = None,
        htf_note: str | None = None,
        regime: str | None = None,
    ) -> SignalAnalysis:
        """Analyse the latest closed candle and return a strictly parsed result.

        Raises:
            AnalysisContextError: for invalid market/indicator input.
            SignalAnalysisError: for LLM invocation or parsing failure (never
                yields LONG/SHORT on failure).
        """
        context = build_analysis_context(
            candles,
            indicators,
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
            event_note=event_note,
            positioning=positioning,
            htf_note=htf_note,
        )
        decision_model, usage, estimated = self._invoke(context)
        return self._to_analysis(
            context, decision_model, usage=usage, funding_rate=funding_rate,
            regime=regime, tokens_estimated=estimated,
        )

    def _invoke(
        self, context: AnalysisContext
    ) -> tuple[SignalDecisionModel, dict[str, int | None] | None, bool]:
        prompt = build_prompt(context)
        usage: dict[str, int | None] | None = None
        estimated = False
        try:
            if self.structured_llm is not None or self.structured_llm_raw is not None:
                raw, usage = _call_structured(
                    self.structured_llm,
                    self.structured_llm_raw,
                    prompt,
                    "BTCUSDT Signal Analyzer",
                )
            else:
                response = self.llm.invoke(prompt)
                usage = _extract_usage(response)
                raw = getattr(response, "content", response)
            if usage is None:
                usage = _estimate_usage(_prompt_text(prompt), raw)
                estimated = True
        except SignalAnalysisError:
            raise
        except Exception as exc:
            text = str(exc)
            if _is_rate_limited(text):
                raise RateLimitedError(
                    f"LLM rate limit{_reset_hint(text)}: {exc}"
                ) from exc
            raise SignalAnalysisError(f"LLM analysis failed: {exc}") from exc
        return _coerce_decision(raw), usage, estimated

    def _to_analysis(
        self,
        context: AnalysisContext,
        model: SignalDecisionModel,
        *,
        usage: dict[str, int | None] | None = None,
        funding_rate: float | None = None,
        regime: str | None = None,
        tokens_estimated: bool = False,
    ) -> SignalAnalysis:
        usage = usage or {}
        return SignalAnalysis(
            decision=model.decision.value,
            confidence=float(model.confidence),
            reasoning=model.reasoning,
            entry_price=model.entry_price,
            stop_loss=model.stop_loss,
            take_profit=model.take_profit,
            provider=self.config.provider,
            model=self.config.model,
            symbol=context.symbol,
            timeframe=context.timeframe,
            analysis_timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            market_timestamp=context.market_timestamp,
            closed_at=context.closed_at,
            candle_close_price=context.last_close,
            temperature=self.config.temperature,
            atr=_snapshot_atr(context),
            llm_calls=1,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            funding_rate=funding_rate,
            regime=regime,
            tokens_estimated=tokens_estimated,
        )


def analyze_signal(
    config: LLMConfig,
    candles: list[Candle],
    indicators,
    *,
    llm: Any | None = None,
    mode: str = "single",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    max_candles: int = DEFAULT_MAX_CANDLES,
    event_note: str | None = None,
    positioning: str | None = None,
    funding_rate: float | None = None,
    htf_note: str | None = None,
    regime: str | None = None,
) -> SignalAnalysis:
    """Convenience wrapper: build an analyzer and run a single analysis.

    ``mode`` selects the pipeline: ``"single"`` is the classic one-call
    structured analysis; ``"multi"`` runs the Phase 5 five-agent debate
    (analyst > bull > bear > trader > risk manager) against the same context.
    Any value other than ``"multi"`` resolves to the single-agent path.
    """
    if mode == "multi":
        from .multiagent import MultiAgentSignalAnalyzer

        analyzer = MultiAgentSignalAnalyzer(config, llm=llm)
    else:
        analyzer = SignalAnalyzer(config, llm=llm)
    return analyzer.analyze(
        candles,
        indicators,
        symbol=symbol,
        timeframe=timeframe,
        max_candles=max_candles,
        event_note=event_note,
        positioning=positioning,
        funding_rate=funding_rate,
        htf_note=htf_note,
        regime=regime,
    )
