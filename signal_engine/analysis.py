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
        }


class SignalAnalyzer:
    """Runs the TradingAgents LLM against a closed-candle context."""

    def __init__(self, config: LLMConfig, llm: Any | None = None) -> None:
        self.config = config
        self.llm = llm if llm is not None else build_llm_client(config).get_llm()
        self.structured_llm = _bind_structured(
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
        )
        decision_model = self._invoke(context)
        return self._to_analysis(context, decision_model)

    def _invoke(self, context: AnalysisContext) -> SignalDecisionModel:
        prompt = build_prompt(context)
        try:
            if self.structured_llm is not None:
                raw = self.structured_llm.invoke(prompt)
                if raw is None:
                    raise SignalAnalysisError(
                        "structured-output invocation returned no parsed result"
                    )
            else:
                response = self.llm.invoke(prompt)
                raw = getattr(response, "content", response)
        except SignalAnalysisError:
            raise
        except Exception as exc:
            raise SignalAnalysisError(f"LLM analysis failed: {exc}") from exc
        return _coerce_decision(raw)

    def _to_analysis(
        self, context: AnalysisContext, model: SignalDecisionModel
    ) -> SignalAnalysis:
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
    )
