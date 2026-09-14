"""Signal Engine — the authoritative boundary between AI output and persistence.

The AI is never authoritative: this engine validates Phase 5 analysis results,
enforces the one-active-signal invariant, and persists signals atomically
through the Phase 3 ``SignalRepository`` (SQLite is the source of truth). A
validated LONG/SHORT is stored as ``PENDING_ENTRY`` — the AI never opens a
position directly; an entry monitor later promotes PENDING_ENTRY -> OPEN.

Pipeline
--------
::

    Phase 5 analysis
        → process/analyze_and_create (this engine)
        → validation            (validator.py, Decimals only)
        → one-active gate       (state.py + repository transaction)
        → SignalRepository      (atomic PENDING_ENTRY insert)
        → CREATED / WAIT / BLOCKED_OPEN_SIGNAL / REJECTED / ERROR

Invariants
----------
- At most one active signal (``PENDING_ENTRY`` or ``OPEN``). The authoritative
  guarantee is the transactional check inside ``SignalRepository.create_signal``
  (``BEGIN IMMEDIATE``), so a concurrent second creation fails safely with
  ``SignalExistsError`` and never overwrites either signal. The Python
  pre-check is an optimization only.
- AI lock: when an active signal exists the engine returns
  ``BLOCKED_OPEN_SIGNAL`` *before* invoking the Phase 5 analyzer, so no LLM
  call is wasted and an already-PENDING signal is never re-analyzed.
- Immutability: the engine never updates a signal with a later AI result;
  entry/SL/TP/direction/timeframe are fixed at creation and the repository
  exposes no price-updating operation.
- Fail-safe: LLM error, malformed/invalid AI output, invalid prices, or a
  database failure all result in NO signal creation and are surfaced as
  ``ERROR``/``REJECTED`` results, never as success.
- Phase 6 places no orders, executes no Demo trades, monitors no prices, and
  has no scheduler; it only validates and persists signals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from database.database import SignalRepository
from database.models import Signal, SignalExistsError, SignalValidationError

from .analysis import SignalAnalysis
from .state import SignalState
from .validator import validate_analysis, validate_decision


class SignalOutcome(str, Enum):
    """Structured outcome vocabulary consumed by later phases."""

    CREATED = "CREATED"
    WAIT = "WAIT"
    BLOCKED_OPEN_SIGNAL = "BLOCKED_OPEN_SIGNAL"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SignalEngineResult:
    """Result of the Signal Engine, without exposing raw database exceptions."""

    outcome: SignalOutcome
    signal: Signal | None = None
    message: str = ""
    errors: tuple[str, ...] = ()

    @classmethod
    def created(cls, signal: Signal) -> SignalEngineResult:
        return cls(
            outcome=SignalOutcome.CREATED,
            signal=signal,
            message="signal created and persisted as PENDING_ENTRY (awaiting its Entry trigger)",
        )

    @classmethod
    def wait(cls) -> SignalEngineResult:
        return cls(
            outcome=SignalOutcome.WAIT,
            message="AI returned WAIT; no signal created",
        )

    @classmethod
    def blocked(cls) -> SignalEngineResult:
        return cls(
            outcome=SignalOutcome.BLOCKED_OPEN_SIGNAL,
            message="an active signal (PENDING_ENTRY or OPEN) already exists; no new signal was created",
        )

    @classmethod
    def rejected(cls, *errors: str) -> SignalEngineResult:
        return cls(
            outcome=SignalOutcome.REJECTED,
            message="; ".join(errors),
            errors=errors,
        )

    @classmethod
    def error(cls, message: str) -> SignalEngineResult:
        return cls(outcome=SignalOutcome.ERROR, message=message)


class SignalEngine:
    """Validates Phase 5 analyses and persists PENDING_ENTRY signals."""

    def __init__(
        self,
        repository: SignalRepository,
        state: SignalState | None = None,
    ) -> None:
        self.repository = repository
        self.state = state if state is not None else SignalState(repository)

    def process(self, analysis: SignalAnalysis) -> SignalEngineResult:
        """Convert a single Phase 5 analysis result into a validated signal.

        Ordering (highest priority first):
        1. WAIT -> WAIT result (never writes or modifies anything).
        2. Active signal exists (PENDING_ENTRY or OPEN) -> BLOCKED_OPEN_SIGNAL
           (no validation performed; the AI must never run while one is active).
        3. Validation failure -> REJECTED (never downgrades to WAIT).
        4. Atomic persist via repository -> CREATED (status PENDING_ENTRY),
           or BLOCKED/REJECTED/ERROR.
        """
        if not isinstance(analysis, SignalAnalysis):
            return SignalEngineResult.rejected(
                "analysis must be a SignalAnalysis result from Phase 5"
            )

        try:
            decision = validate_decision(analysis.decision)
        except SignalValidationError as exc:
            return SignalEngineResult.rejected(str(exc))

        if decision == "WAIT":
            return SignalEngineResult.wait()

        if self.state.is_active():
            return SignalEngineResult.blocked()

        try:
            candidate = validate_analysis(analysis)
        except SignalValidationError as exc:
            return SignalEngineResult.rejected(str(exc))

        try:
            signal = self.repository.create_signal(
                symbol=candidate.symbol,
                timeframe=candidate.timeframe,
                direction=candidate.decision,
                entry=candidate.entry,
                stop_loss=candidate.stop_loss,
                take_profit=candidate.take_profit,
                confidence=candidate.confidence,
                rationale=candidate.reasoning,
                provider=candidate.provider,
                model_name=candidate.model_name,
                temperature=candidate.temperature,
                analysis_timestamp=candidate.analysis_timestamp,
                market_timestamp=candidate.market_timestamp,
                candle_close_price=candidate.candle_close_price,
            )
        except SignalExistsError:
            # A concurrent creator won the race: both signals are safe, we block.
            return SignalEngineResult.blocked()
        except SignalValidationError as exc:
            # Defense in depth: repository-level validation also failed.
            return SignalEngineResult.rejected(str(exc))
        except Exception as exc:
            # Database failure must never look like a successful creation.
            return SignalEngineResult.error(f"failed to persist signal: {exc}")

        return SignalEngineResult.created(signal)

    def analyze_and_create(
        self,
        candles: Any,
        indicators: Any,
        analyzer: Callable[[Any, Any], SignalAnalysis],
    ) -> SignalEngineResult:
        """Gate the AI first, then analyze and create.

        When an active signal (PENDING_ENTRY or OPEN) already exists,
        ``BLOCKED_OPEN_SIGNAL`` is returned and ``analyzer`` is never called,
        so no LLM call is wasted. ``analyzer`` must accept ``(candles, indicators)``
        positionally and return a Phase 5 ``SignalAnalysis`` (e.g. a wrapped
        ``signal_engine.analysis.analyze_signal``). A race after the gate is
        still caught by the repository transaction.
        """
        if self.state.is_active():
            return SignalEngineResult.blocked()

        try:
            analysis = analyzer(candles, indicators)
        except Exception as exc:
            return SignalEngineResult.error(f"AI analysis failed: {exc}")

        return self.process(analysis)


__all__ = [
    "SignalEngine",
    "SignalEngineResult",
    "SignalOutcome",
]
