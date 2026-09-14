"""Phase 13 LLM decision provider with caching, latency and failure safety.

Implements the Phase 12 :class:`BacktestDecisionProvider` protocol so the
isolated backtest engine drives the benchmark exactly like production: one
closed-candle window at a time, no look-ahead, and single-active-signal
blocking. On top of the raw ``SignalAnalyzer`` call it adds:

- a per-model :class:`DecisionCache` layer — a cache hit replays the recorded
  decision without re-calling the LLM;
- per-call latency accounting (including failures) for the latency report;
- failure safety — an LLM/parse failure is recorded as ``FAILED`` and the
  engine is handed a deliberately invalid LONG (``entry == stop_loss``) so the
  engine records an explicit rejected event and keeps running. A failure
  **never** becomes WAIT and **never** becomes a position;
- invalid-output tagging — a structurally valid but rule-invalid LONG/SHORT
  (e.g. ``entry`` on the wrong side of ``stop_loss``) is recorded as ``INVALID``
  and rejected by the engine naturally, never auto-converted to a valid trade.

Only the analysis invocation (``analyze_fn``) is ever measured for latency; the
whole decision is synchronous and sequential.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from backtest.config import BacktestConfigError
from database.models import SignalValidationError
from signal_engine.analysis import (
    SignalAnalysis,
    SignalAnalysisError,
    SignalAnalyzer,
    SignalDecision,
    SignalDecisionModel,
)
from signal_engine.context import DEFAULT_MAX_CANDLES
from signal_engine.llm import LLMConfig
from signal_engine.validator import validate_analysis

from .cache import DecisionCache, compute_scope
from .models import CONTEXT_VERSION, DecisionStatus, ModelDecision, RunningInferenceStats

logger = logging.getLogger(__name__)

#: Maximum characters kept in a record's ``error`` field.
_ERROR_CAP = 500


#: Callable matching the analyzer contract used by the Phase 5 engine.
AnalyzeFn = Callable[
    [list, pd.DataFrame],
    Any,
]


class BenchmarkConfigError(BacktestConfigError):
    """Invalid configuration for a Phase 13 benchmark run."""


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _model_to_analysis(
    decision_model: SignalDecisionModel,
    *,
    candle,
    symbol: str,
    timeframe: str,
    provider: str,
    model: str,
) -> SignalAnalysis:
    return SignalAnalysis(
        decision=decision_model.decision.value,
        confidence=float(decision_model.confidence),
        reasoning=decision_model.reasoning,
        entry_price=decision_model.entry_price,
        stop_loss=decision_model.stop_loss,
        take_profit=decision_model.take_profit,
        provider=provider,
        model=model,
        symbol=symbol,
        timeframe=timeframe,
        analysis_timestamp=_iso(candle.timestamp),
        market_timestamp=_iso(candle.timestamp),
        closed_at=_iso(candle.close_time),
        candle_close_price=float(candle.close),
    )


def _record_to_analysis(
    record: ModelDecision,
    *,
    candle,
    symbol: str,
    timeframe: str,
    provider: str,
    model: str,
) -> SignalAnalysis:
    """Rebuild the exact analysis a record represents for the engine.

    The record's decision fields are validated through the same strict pydantic
    contract used by the live path, so a malformed/corrupt cached record cannot
    silently become a signal.
    """
    payload: dict[str, Any] = {
        "decision": record.decision,
        "confidence": record.confidence if record.confidence is not None else 0.0,
        "reasoning": record.reasoning or "",
        "entry_price": record.entry_price,
        "stop_loss": record.stop_loss,
        "take_profit": record.take_profit,
    }
    try:
        decision_model = SignalDecisionModel.model_validate(payload)
    except PydanticValidationError as exc:
        raise BenchmarkConfigError(
            f"malformed cached decision for candle {record.candle_ts}: {exc}"
        ) from exc
    return _model_to_analysis(
        decision_model,
        candle=candle,
        symbol=symbol,
        timeframe=timeframe,
        provider=provider,
        model=model,
    )


class LLMDecisionProvider:
    """Benchmark decision provider backed by a local LLM through ``SignalAnalyzer``.

    Args:
        model: the Ollama/Local model id (e.g. ``qwen3:4b``).
        provider: the provider name (default ``ollama``).
        cache: the per-model :class:`DecisionCache` used as both store and cache.
        analyze_fn: injectable analyzer with the ``SignalAnalyzer.analyze``
            signature; defaults to a real ``SignalAnalyzer`` built from
            ``config``.
        config: resolved :class:`LLMConfig` used for both the analyzer and the
            cache scope.
        context_version: prompt/context version used in the cache scope.
        dataset_id: optional cache-scope discriminant (same dataset label).
        force_fresh: ignore the cache and call the LLM for every candle.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        cache: DecisionCache,
        config: LLMConfig | None = None,
        analyze_fn: AnalyzeFn | None = None,
        context_version: int = CONTEXT_VERSION,
        dataset_id: str | None = None,
        force_fresh: bool = False,
    ) -> None:
        if not model or not isinstance(model, str):
            raise BenchmarkConfigError("a non-empty model id is required")
        self.model = model
        self.provider = provider
        self.cache = cache
        self.context_version = context_version
        self.dataset_id = dataset_id
        self.force_fresh = force_fresh
        self.stats = RunningInferenceStats()

        if analyze_fn is not None:
            self._analyze_fn = analyze_fn
        elif config is not None:
            self._analyze_fn = SignalAnalyzer(config).analyze
        else:
            raise BenchmarkConfigError(
                "either an analyze_fn or a resolved LLMConfig is required"
            )
        self._config = config

    def decide(
        self,
        candles: list,
        indicators: pd.DataFrame,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> SignalAnalysis:
        candle = candles[-1]
        if not self.force_fresh:
            cached = self.cache.get(self._scope(symbol, timeframe, max_candles), int(candle.timestamp))
            if cached is not None:
                self.stats.add(
                    0.0, cached=True, failed=cached.status == DecisionStatus.FAILED
                )
                return _record_to_analysis(
                    cached,
                    candle=candle,
                    symbol=symbol,
                    timeframe=timeframe,
                    provider=self.provider,
                    model=self.model,
                )
        return self._fresh(candles, indicators, symbol, timeframe, max_candles, candle)

    def _scope(self, symbol: str, timeframe: str, max_candles: int) -> str:
        config = self._config
        return compute_scope(
            provider=self.provider,
            model=self.model,
            base_url=config.base_url if config else None,
            temperature=config.temperature if config else None,
            timeout=config.timeout if config else None,
            max_retries=config.max_retries if config else None,
            max_tokens=config.max_tokens if config else None,
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
            context_version=self.context_version,
            dataset_id=self.dataset_id,
        )

    def _fresh(
        self,
        candles: list,
        indicators: pd.DataFrame,
        symbol: str,
        timeframe: str,
        max_candles: int,
        candle,
    ) -> SignalAnalysis:
        start = time.perf_counter()
        try:
            raw = self._analyze_fn(
                candles, indicators, symbol=symbol, timeframe=timeframe, max_candles=max_candles
            )
            latency = time.perf_counter() - start
            analysis = _coerce_analysis(raw, candle, symbol, timeframe, self.provider, self.model)
            status = _classify(analysis)
            self._put_record(analysis, status, candle, symbol, timeframe, max_candles, latency)
            self.stats.add(latency, cached=False, failed=False)
            return analysis
        except Exception as exc:
            latency = time.perf_counter() - start
            logger.debug(
                "benchmark model %s failed for candle %s: %s",
                self.model, candle.timestamp, exc,
            )
            fallback = self._fallback_analysis(candle, symbol, timeframe)
            try:
                self._put_record(
                    fallback, DecisionStatus.FAILED, candle, symbol, timeframe,
                    max_candles, latency, error=str(exc)[:_ERROR_CAP],
                )
            except Exception as put_exc:  # pragma: no cover - best-effort write
                logger.warning("could not record benchmark failure: %s", put_exc)
            self.stats.add(latency, cached=False, failed=True)
            return fallback

    def _put_record(
        self,
        analysis: SignalAnalysis,
        status: str,
        candle,
        symbol: str,
        timeframe: str,
        max_candles: int,
        latency: float,
        *,
        error: str | None = None,
    ) -> None:
        record = ModelDecision(
            model=self.model,
            provider=self.provider,
            candle_ts=int(candle.timestamp),
            candle_close=float(candle.close),
            status=status,
            decision=analysis.decision,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
            entry_price=analysis.entry_price,
            stop_loss=analysis.stop_loss,
            take_profit=analysis.take_profit,
            latency_sec=round(latency, 6),
            cached=False,
            scope=self._scope(symbol, timeframe, max_candles),
            context_version=self.context_version,
            analysis_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            error=error,
        )
        self.cache.put(record)

    def _fallback_analysis(self, candle, symbol: str, timeframe: str) -> SignalAnalysis:
        """A deliberately invalid LONG (entry == stop) the engine will reject.

        Producing WAIT here would silently mislabel an LLM failure as the model
        choosing to wait; producing no analysis would abort the whole run. An
        entry-equals-stop LONG fails Phase 12 validation, surfaces as an explicit
        rejected event, and keeps the benchmark running.
        """
        close = float(candle.close)
        return _model_to_analysis(
            SignalDecisionModel(
                decision=SignalDecision.LONG,
                confidence=0.0,
                reasoning="LLM inference failed; invalid fallback LONG (entry == "
                "stop_loss) so the engine records an explicit rejected event",
                entry_price=close,
                stop_loss=close,
                take_profit=round(close * 1.02, 2),
            ),
            candle=candle,
            symbol=symbol,
            timeframe=timeframe,
            provider=self.provider,
            model=self.model,
        )


def _coerce_analysis(
    raw: Any,
    candle,
    symbol: str,
    timeframe: str,
    provider: str,
    model: str,
) -> SignalAnalysis:
    """Normalize an analyze_fn result into a model-tagged SignalAnalysis."""
    if isinstance(raw, SignalAnalysis):
        return raw
    try:
        decision_model = (
            raw
            if isinstance(raw, SignalDecisionModel)
            else SignalDecisionModel.model_validate(raw)
        )
    except (PydanticValidationError, ValueError, TypeError) as exc:
        raise SignalAnalysisError(f"invalid analysis output: {exc}") from exc
    return _model_to_analysis(
        decision_model,
        candle=candle,
        symbol=symbol,
        timeframe=timeframe,
        provider=provider,
        model=model,
    )


def _classify(analysis: SignalAnalysis) -> str:
    """Tag a live LONG/SHORT as valid (OK) or rule-invalid (INVALID)."""
    if analysis.decision not in ("LONG", "SHORT"):
        return DecisionStatus.OK
    try:
        validate_analysis(analysis)
    except SignalValidationError:
        return DecisionStatus.INVALID
    return DecisionStatus.OK


__all__ = [
    "AnalyzeFn",
    "BenchmarkConfigError",
    "LLMDecisionProvider",
    "_record_to_analysis",
]
