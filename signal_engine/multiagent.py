"""Phase 5 multi-agent analysis of the closed-candle context.

This module implements the TradingAgents-style committee debate on top of the
single-provider LLM already configured for the engine:

    Technical Analyst  ->  Bull Researcher  ->  Bear Researcher
                        ->  Trader           ->  Risk Manager  ->  signal

Every agent runs against the SAME :class:`LLMConfig` (one provider / one model /
one API key) and the SAME deterministic :class:`AnalysisContext`. Each role is a
separate structured LLM call; the later agents receive the earlier agents' typed
outputs as JSON so the debate is visible to the model.

Strictness contract (mirrors ``signal_engine.analysis``)
--------------------------------------------------------
- Every agent uses structured output when supported, otherwise a strict JSON
  object matching its pydantic schema. There is no free-text fallback:
  malformed output raises :class:`SignalAnalysisError`.
- The Trader and Risk Manager reuse the exact ``SignalDecisionModel`` schema of
  the single-agent path, so WAIT-never-carries-prices and the strict
  LONG/SHORT/SL/Entry/TP ordering rules hold for the final decision.
- The Risk Manager is the final gate: it must re-validate the decision and the
  ordering; any violation aborts the whole analysis (no signal is produced).
- An LLM failure in ANY agent raises :class:`SignalAnalysisError` and can never
  silently degrade into LONG or SHORT.
- The analysis is purely in-memory: it never writes to the database, never calls
  the Binance API, and never invokes TradingAgents' stock/news/social/fundamental
  dataflows (same constraint as ``signal_engine.analysis``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from binance.market_data import Candle

from .analysis import (
    _NO_EXTERNAL_TOOLS,
    RateLimitedError,
    SignalAnalysis,
    SignalAnalysisError,
    SignalDecision,
    SignalDecisionModel,
    _bind_structured,
    _extract_usage,
    _is_rate_limited,
    _reset_hint,
    _snapshot_atr,
    _validate_decision_rules,
)
from .context import DEFAULT_MAX_CANDLES, AnalysisContext, build_analysis_context
from .llm import LLMConfig, build_llm_client

logger = logging.getLogger(__name__)

ENV_ANALYSIS_MODE = "BTCUSDT_ANALYSIS_MODE"
SINGLE_AGENT_MODE = "single"
MULTI_AGENT_MODE = "multi"


def resolve_analysis_mode(env: Any | None = None) -> bool:
    """Return whether the multi-agent pipeline is enabled.

    Reads ``BTCUSDT_ANALYSIS_MODE`` from ``env`` (defaults to ``os.environ``);
    only the exact (case-insensitive) value ``multi`` enables multi-agent.
    Anything else, including an unset variable, keeps the single-agent path.
    """
    source = os.environ if env is None else env
    value = source.get(ENV_ANALYSIS_MODE)
    if not value:
        return False
    return str(value).strip().lower() == MULTI_AGENT_MODE


class AnalystBias(str, Enum):
    """The Technical Analyst's directional read of the supplied data only."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class AnalystStep(BaseModel):
    """Structured output of the Technical Analyst agent."""

    bias: AnalystBias = Field(
        description="Directional read of the supplied closed-candle evidence: "
        "LONG, SHORT, or NEUTRAL."
    )
    summary: str = Field(
        min_length=1,
        description="Concise, evidence-only summary of the technical setup, "
        "referencing the supplied indicators and candle structure.",
    )


class CaseStep(BaseModel):
    """Structured output of a Bull or Bear researcher agent."""

    thesis: str = Field(
        min_length=1,
        description="The directional thesis, argued ONLY from the supplied "
        "closed candles and indicator values.",
    )
    key_evidence: list[str] = Field(
        default_factory=list,
        description="Bullet points of evidence already present in the prompt. "
        "Never invent prices, levels, news, or data.",
    )


class RiskStep(SignalDecisionModel):
    """The Risk Manager's final decision plus its risk assessment note."""

    risk_note: str = Field(
        min_length=1,
        description="A short risk assessment of the proposal: position risk vs "
        "stop distance, ordering validity, and why the final decision is "
        "acceptable (or why it was turned into WAIT).",
    )


_ANALYST_SYSTEM = (
    "You are the Technical Analyst in a conservative BTCUSDT USDT-M Futures "
    "signal committee. You analyse ONLY closed 1H candles and the indicator "
    "values (EMA20/50/200, RSI14, MACD, ATR14, volume) computed from those "
    "candles by deterministic Python code. You have no news, no sentiment, no "
    "fundamentals, and no real-time data.\n"
    "Return a directional bias (LONG/SHORT/NEUTRAL) and a short summary that "
    "cites only the supplied values. If the evidence is messy, NEUTRAL is safe."
    "\n"
    + _NO_EXTERNAL_TOOLS
)

_BULL_SYSTEM = (
    "You are the Bull Researcher in a conservative BTCUSDT USDT-M Futures "
    "signal committee. Your job is to argue the strongest LONG case using ONLY "
    "the supplied closed 1H candles, indicator values, and the Technical "
    "Analyst's read. Do not invent prices, levels, news, or data; if evidence "
    "for a LONG is missing, say so explicitly.\n"
    + _NO_EXTERNAL_TOOLS
)

_BEAR_SYSTEM = (
    "You are the Bear Researcher in a conservative BTCUSDT USDT-M Futures "
    "signal committee. Your job is to argue the strongest SHORT case using ONLY "
    "the supplied closed 1H candles, indicator values, and the Technical "
    "Analyst's read. Do not invent prices, levels, news, or data; if evidence "
    "for a SHORT is missing, say so explicitly.\n"
    + _NO_EXTERNAL_TOOLS
)

_TRADER_SYSTEM = (
    "You are the Trader in a conservative BTCUSDT USDT-M Futures signal "
    "committee. Decide exactly one of LONG, SHORT, or WAIT based ONLY on the "
    "supplied closed-candle data, the indicator values, and the Analyst/Bull/"
    "Bear research.\n"
    "Rules:\n"
    "1. confidence must be 0-100 (100 = maximum conviction).\n"
    "2. For LONG/SHORT propose entry_price, stop_loss, and take_profit as "
    "absolute USDT prices with LONG: stop_loss < entry_price < take_profit and "
    "SHORT: take_profit < entry_price < stop_loss. These are indicative "
    "planning levels, not guaranteed fills.\n"
    "3. For WAIT all three price fields must be null.\n"
    "4. Base every statement on the supplied data; never invent anything.\n"
    + _NO_EXTERNAL_TOOLS
)

_RISK_SYSTEM = (
    "You are the Risk Manager in a conservative BTCUSDT USDT-M Futures signal "
    "committee. You receive the trader's proposal and the committee research. "
    "Review the risk: stop distance vs the ATR/volatility evidence supplied, "
    "ordering validity, and position-sizing sanity.\n"
    "Return the FINAL decision using the same structured fields as the trader, "
    "enforcing:\n"
    "- LONG: stop_loss < entry_price < take_profit; "
    "SHORT: take_profit < entry_price < stop_loss.\n"
    "- WAIT must keep entry_price/stop_loss/take_profit null.\n"
    "- confidence 0-100 and every statement grounded ONLY in the supplied data.\n"
    "Write a concise risk_note summarising the assessment.\n"
    + _NO_EXTERNAL_TOOLS
)


def _render_step(label: str, model: BaseModel) -> str:
    return (
        f"\n### {label}\n"
        + json.dumps(model.model_dump(), ensure_ascii=False, indent=2)
    )


def _accumulate_totals(
    totals: dict[str, int | None] | None,
    usage: dict[str, int | None] | None,
) -> dict[str, int | None] | None:
    """Sum per-step token usage across debate agents (NULL-safe).

    A step without metadata contributes nothing; fields stay NULL unless at
    least one step reported them.
    """
    if not usage:
        return totals
    merged = dict(totals or {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            current = merged.get(key)
            merged[key] = value if current is None else current + value
    return merged or None


def _analyst_prompt(context: AnalysisContext) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _ANALYST_SYSTEM},
        {"role": "user", "content": context.rendered},
    ]


def _case_prompt(
    system: str, context: AnalysisContext, analyst: AnalystStep
) -> list[dict[str, str]]:
    content = context.rendered + _render_step("Technical Analyst", analyst)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _trader_prompt(
    context: AnalysisContext, analyst: AnalystStep, bull: CaseStep, bear: CaseStep
) -> list[dict[str, str]]:
    content = (
        context.rendered
        + _render_step("Technical Analyst", analyst)
        + _render_step("Bull Researcher", bull)
        + _render_step("Bear Researcher", bear)
    )
    return [{"role": "system", "content": _TRADER_SYSTEM}, {"role": "user", "content": content}]


def _risk_prompt(
    context: AnalysisContext,
    analyst: AnalystStep,
    bull: CaseStep,
    bear: CaseStep,
    trader: SignalDecisionModel,
) -> list[dict[str, str]]:
    content = (
        context.rendered
        + _render_step("Technical Analyst", analyst)
        + _render_step("Bull Researcher", bull)
        + _render_step("Bear Researcher", bear)
        + _render_step("Trader Proposal", trader)
    )
    return [{"role": "system", "content": _RISK_SYSTEM}, {"role": "user", "content": content}]


def _coerce_step(raw: Any, schema: type[BaseModel], agent_name: str) -> BaseModel:
    """Parse raw LLM output into ``schema`` or raise :class:`SignalAnalysisError`."""
    try:
        if isinstance(raw, schema):
            return raw
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                raise SignalAnalysisError(f"{agent_name} returned empty output")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SignalAnalysisError(
                    f"{agent_name} returned invalid JSON instead of the required schema"
                ) from exc
            if not isinstance(data, dict):
                raise SignalAnalysisError(
                    f"{agent_name} returned a JSON value that is not an object"
                )
            return schema.model_validate(data)
    except ValidationError as exc:
        raise SignalAnalysisError(f"{agent_name} returned malformed output: {exc}") from exc
    raise SignalAnalysisError(
        f"{agent_name} returned an unsupported output type: {type(raw).__name__}"
    )


def _validate_ordering(model: SignalDecisionModel) -> SignalDecisionModel:
    """Enforce the strict SL/Entry/TP price ordering for LONG/SHORT."""
    if model.decision == SignalDecision.WAIT:
        return model
    prices = (model.entry_price, model.stop_loss, model.take_profit)
    if any(price is None for price in prices):
        raise SignalAnalysisError(
            f"{model.decision.value} requires entry, stop-loss, and take-profit levels"
        )
    entry, stop_loss, take_profit = prices
    if model.decision == SignalDecision.LONG:
        valid = stop_loss < entry < take_profit
    else:
        valid = take_profit < entry < stop_loss
    if not valid:
        raise SignalAnalysisError(
            f"{model.decision.value} violates the stop-loss/entry/take-profit ordering rule"
        )
    return model


_ROLE_NAMES = {
    "analyst": "Technical Analyst",
    "bull": "Bull Researcher",
    "bear": "Bear Researcher",
    "trader": "Trader",
    "risk": "Risk Manager",
}


class MultiAgentSignalAnalyzer:
    """Runs the five-agent debate against one closed-candle context.

    Mirrors the ``SignalAnalyzer.analyze`` signature so the scheduler/runtime
    and any test fake can drop it in transparently.
    """

    def __init__(self, config: LLMConfig, llm: Any | None = None) -> None:
        self.config = config
        self.llm = llm if llm is not None else build_llm_client(config).get_llm()
        self.structured_llm = {
            name: _bind_structured(self.llm, schema, agent_name)
            for name, schema, agent_name in (
                ("analyst", AnalystStep, _ROLE_NAMES["analyst"]),
                ("bull", CaseStep, _ROLE_NAMES["bull"]),
                ("bear", CaseStep, _ROLE_NAMES["bear"]),
                ("trader", SignalDecisionModel, _ROLE_NAMES["trader"]),
                ("risk", RiskStep, _ROLE_NAMES["risk"]),
            )
        }

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
        """Run the full committee debate and return a strictly parsed result.

        Raises:
            AnalysisContextError: for invalid market/indicator input.
            SignalAnalysisError: for any agent failure or invalid final
                decision (never yields LONG/SHORT on failure).
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
        analyst, usage = self._step("analyst", AnalystStep, _analyst_prompt(context))
        calls, totals = 1, _accumulate_totals(None, usage)
        bull, usage = self._step("bull", CaseStep, _case_prompt(_BULL_SYSTEM, context, analyst))
        calls, totals = calls + 1, _accumulate_totals(totals, usage)
        bear, usage = self._step("bear", CaseStep, _case_prompt(_BEAR_SYSTEM, context, analyst))
        calls, totals = calls + 1, _accumulate_totals(totals, usage)
        trader, usage = self._step(
            "trader",
            SignalDecisionModel,
            _trader_prompt(context, analyst, bull, bear),
        )
        calls, totals = calls + 1, _accumulate_totals(totals, usage)
        risk, usage = self._step(
            "risk", RiskStep, _risk_prompt(context, analyst, bull, bear, trader)
        )
        calls, totals = calls + 1, _accumulate_totals(totals, usage)
        decision_model = _validate_decision_rules(risk)
        _validate_ordering(decision_model)
        return self._to_analysis(
            context, decision_model, trader, risk, llm_calls=calls, usage=totals,
            funding_rate=funding_rate, regime=regime,
        )

    def _step(
        self,
        name: str,
        schema: type[BaseModel],
        prompt: list[dict[str, str]],
    ) -> tuple[BaseModel, dict[str, int | None] | None]:
        agent_name = _ROLE_NAMES[name]
        structured = self.structured_llm[name]
        usage: dict[str, int | None] | None = None
        try:
            if structured is not None:
                raw = structured.invoke(prompt)
                if raw is None:
                    raise SignalAnalysisError(
                        f"{agent_name} structured-output invocation returned "
                        "no parsed result"
                    )
            else:
                response = self.llm.invoke(prompt)
                usage = _extract_usage(response)
                raw = getattr(response, "content", response)
        except SignalAnalysisError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any LLM error
            text = str(exc)
            if _is_rate_limited(text):
                raise RateLimitedError(
                    f"{agent_name} LLM rate limit{_reset_hint(text)}: {exc}"
                ) from exc
            raise SignalAnalysisError(f"{agent_name} LLM analysis failed: {exc}") from exc
        return _coerce_step(raw, schema, agent_name), usage

    def _to_analysis(
        self,
        context: AnalysisContext,
        model: SignalDecisionModel,
        trader: SignalDecisionModel,
        risk: RiskStep,
        *,
        llm_calls: int = 5,
        usage: dict[str, int | None] | None = None,
        funding_rate: float | None = None,
        regime: str | None = None,
    ) -> SignalAnalysis:
        reasoning = f"{risk.reasoning}\n[Risk note] {risk.risk_note}"
        usage = usage or {}
        return SignalAnalysis(
            decision=model.decision.value,
            confidence=float(model.confidence),
            reasoning=reasoning,
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
            llm_calls=llm_calls,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            funding_rate=funding_rate,
            regime=regime,
        )
