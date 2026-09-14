"""Decision providers for the backtest engine (Phase 12).

A :class:`BacktestDecisionProvider` mirrors the Phase 5 ``SignalAnalyzer`` call:
it turns a bounded window of CLOSED decision-timeframe candles plus their
deterministic indicators into a :class:`SignalAnalysis`. The engine passes only
``candles[:i+1]`` for the candle being decided, so a provider can never see a
future candle.

Phase 12 ships deterministic, no-LLM providers:

- :class:`WaitDecisionProvider` — always WAIT (used to smoke-test the pipeline).
- :class:`FunctionDecisionProvider` — delegates to a pure function, so tests and
  later phases can reuse the real TradingAgents analyzer or a rule-based policy.
- :class:`ReplayDecisionProvider` — replays a precomputed mapping of candle open
  times to decisions (a record of a previous real run). Missing candle keys
  produce WAIT.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd
from pydantic_core import ValidationError as PydanticValidationError

from binance.market_data import Candle
from signal_engine.analysis import SignalAnalysis, SignalDecision, SignalDecisionModel
from signal_engine.context import DEFAULT_MAX_CANDLES

from .config import BacktestExecutionError

_DECISION_ALIASES = {
    "long": SignalDecision.LONG,
    "short": SignalDecision.SHORT,
    "wait": SignalDecision.WAIT,
}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _model_to_analysis(
    model: SignalDecisionModel, *, candle: Candle, symbol: str, timeframe: str
) -> SignalAnalysis:
    return SignalAnalysis(
        decision=model.decision.value,
        confidence=float(model.confidence),
        reasoning=model.reasoning,
        entry_price=model.entry_price,
        stop_loss=model.stop_loss,
        take_profit=model.take_profit,
        provider="backtest",
        model="backtest",
        symbol=symbol,
        timeframe=timeframe,
        analysis_timestamp=_iso(candle.timestamp),
        market_timestamp=_iso(candle.timestamp),
        closed_at=_iso(candle.close_time),
        candle_close_price=float(candle.close),
    )


def coerce_decision(
    value: Any,
    *,
    candle: Candle,
    symbol: str,
    timeframe: str,
) -> SignalAnalysis:
    """Normalize any supported decision payload into a :class:`SignalAnalysis`.

    Accepted inputs mirror the Phase 5 output contract: a ``SignalAnalysis``,
    a ``SignalDecisionModel``, a dict validable by that model, or a plain
    ``LONG`` / ``SHORT`` / ``WAIT`` string. Anything else raises
    :class:`BacktestExecutionError`; a backtest never guesses.
    """
    if isinstance(value, SignalAnalysis):
        return SignalAnalysis(
            decision=value.decision,
            confidence=value.confidence,
            reasoning=value.reasoning,
            entry_price=value.entry_price,
            stop_loss=value.stop_loss,
            take_profit=value.take_profit,
            provider=value.provider,
            model=value.model,
            symbol=symbol,
            timeframe=timeframe,
            analysis_timestamp=_iso(candle.timestamp),
            market_timestamp=_iso(candle.timestamp),
            closed_at=_iso(candle.close_time),
            candle_close_price=float(candle.close),
        )
    if isinstance(value, SignalDecisionModel):
        return _model_to_analysis(value, candle=candle, symbol=symbol, timeframe=timeframe)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise BacktestExecutionError("replayed decision string must not be empty")
        decision = _DECISION_ALIASES.get(text.lower())
        if decision is None:
            raise BacktestExecutionError(
                f"replayed decision must be LONG, SHORT, or WAIT, got {value!r}"
            )
        confidence = 100.0 if decision is SignalDecision.WAIT else 0.0
        model = SignalDecisionModel(
            decision=decision,
            confidence=confidence,
            reasoning="replayed backtest decision",
        )
        return _model_to_analysis(model, candle=candle, symbol=symbol, timeframe=timeframe)
    if isinstance(value, dict):
        try:
            model = SignalDecisionModel.model_validate(value)
        except PydanticValidationError as exc:
            raise BacktestExecutionError(
                f"malformed replayed decision for candle {candle.timestamp}: {exc}"
            ) from exc
        return _model_to_analysis(model, candle=candle, symbol=symbol, timeframe=timeframe)
    raise BacktestExecutionError(
        f"unsupported decision payload {type(value).__name__}"
    )


class BacktestDecisionProvider(Protocol):
    """Decision contract that mirrors ``SignalAnalyzer.analyze``.

    ``candles`` is the window ``candles[:i+1]`` (all closed) and ``indicators``
    is ``compute_indicator_matrix(candles)`` computed from exactly that window,
    guaranteeing no look-ahead for rule-based providers.
    """

    provider: str
    model: str

    def decide(
        self,
        candles: list[Candle],
        indicators: pd.DataFrame,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> SignalAnalysis: ...


@dataclass(frozen=True)
class WaitDecisionProvider:
    """Always returns a WAIT analysis; never opens a position."""

    provider: str = "wait"
    model: str = "wait"

    def decide(
        self,
        candles: list[Candle],
        indicators: pd.DataFrame,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> SignalAnalysis:
        return coerce_decision(
            SignalDecisionModel(
                decision=SignalDecision.WAIT,
                confidence=0.0,
                reasoning="wait policy for backtest",
            ),
            candle=candles[-1],
            symbol=symbol,
            timeframe=timeframe,
        )


@dataclass(frozen=True)
class FunctionDecisionProvider:
    """Delegates decisions to a pure function with the same signature as the protocol.

    ``fn`` may return any ``coerce_decision``-compatible payload, so rule-based
    strategies can return plain ``"LONG"`` strings while the real analyzer can be
    wrapped as ``FunctionDecisionProvider(SignalAnalyzer(...).analyze)``.
    """

    fn: Callable[..., Any]
    provider: str = "function"
    model: str = "none"

    def decide(
        self,
        candles: list[Candle],
        indicators: pd.DataFrame,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> SignalAnalysis:
        result = self.fn(
            candles,
            indicators,
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
        )
        return coerce_decision(
            result, candle=candles[-1], symbol=symbol, timeframe=timeframe
        )


@dataclass(frozen=True)
class ReplayDecisionProvider:
    """Replays a precomputed ``{candle_open_time_ms: decision}`` mapping.

    Keys are open-time timestamps of the decision candles. Values may be any
    ``coerce_decision`` payload. A candle with no key in the mapping is WAIT.
    An unrecognized payload raises :class:`BacktestExecutionError`.
    """

    decisions: Mapping[int, Any]
    provider: str = "replay"
    model: str = "replay"

    def decide(
        self,
        candles: list[Candle],
        indicators: pd.DataFrame,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = DEFAULT_MAX_CANDLES,
    ) -> SignalAnalysis:
        candle = candles[-1]
        value = self.decisions.get(candle.timestamp)
        if value is None:
            return coerce_decision(
                SignalDecisionModel(
                    decision=SignalDecision.WAIT,
                    confidence=0.0,
                    reasoning="no replay entry for this candle; wait",
                ),
                candle=candle,
                symbol=symbol,
                timeframe=timeframe,
            )
        return coerce_decision(value, candle=candle, symbol=symbol, timeframe=timeframe)


__all__ = [
    "BacktestDecisionProvider",
    "FunctionDecisionProvider",
    "ReplayDecisionProvider",
    "WaitDecisionProvider",
    "coerce_decision",
]
