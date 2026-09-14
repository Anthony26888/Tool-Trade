"""Startup recovery for the BTCUSDT signal system (Phase 9).

On process start the system must re-establish its trading state from SQLite,
because SQLite — never Python memory, cached variables, or the scheduler — is
the source of truth.

What recovery does
------------------
- Reads the database for the single active signal — ``PENDING_ENTRY`` (awaits
  its Entry trigger) or ``OPEN`` (awaits TP/SL) — via ``SignalRepository``.
- Exactly one active signal: returns it verbatim (``RECOVERED_OPEN``), so the
  ``SignalMonitor`` can resume — entry monitoring for a PENDING_ENTRY signal,
  TP/SL monitoring for an OPEN signal. No AI, no new signal, no overwrite of
  entry/SL/TP, no promotion or closure.
- No active signal: returns ``NO_OPEN_SIGNAL``. A later (Phase 10) analysis
  scheduler decides when the AI may run; this module never schedules it.
- More than one active signal (inconsistent state that violates the
  single-active invariant): refuses to pick one arbitrarily and returns
  ``RECOVERY_ERROR``.

What recovery does NOT do
-------------------------
- It never calls the AI (no LLM, no ``SignalAnalyzer``, no
  ``SignalEngine.analyze_and_create``), never fetches Binance data, never
  places/orders, never creates a signal, and never updates a signal. It is a
  read/resume-only pass over SQLite.
- It never reopens a closed signal. ``TP_HIT``/``SL_HIT``/``CANCELLED`` remain
  closed; creating the next signal is the later scheduler's decision.
- It never promotes a recovered ``PENDING_ENTRY`` to ``OPEN``: entry
  confirmation is the monitor's job, driven by real closed candles.
- It introduces no new persistence: no recovery/startup tables, no second
  signal table. The existing Phase 3 schema is the only state store.

Concurrency and races
---------------------
Recovery only issues SQLite reads. A closed/promoted-during-race (e.g. another
process' ``SignalMonitor`` promotes PENDING_ENTRY -> OPEN or closes OPEN
between startup and the first observer poll) is handled by the repository's
atomic ``BEGIN IMMEDIATE`` transitions; recovery never caches active state in
memory or assumes a recovered object is still active. The caller should always
re-read through the repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from database.database import SignalRepository, validate_signal_prices
from database.models import (
    DIRECTIONS,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    Signal,
    SignalValidationError,
)

from .monitor import DEFAULT_CANDLE_LIMIT, SignalMonitor

logger = logging.getLogger(__name__)


class RecoveryOutcome(str, Enum):
    """Structured result of a startup recovery pass."""

    RECOVERED_OPEN = "RECOVERED_OPEN"
    NO_OPEN_SIGNAL = "NO_OPEN_SIGNAL"
    RECOVERY_ERROR = "RECOVERY_ERROR"


@dataclass(frozen=True)
class RecoveryResult:
    """Result of the recovery pass, without leaking raw database exceptions.

    ``signal`` holds the recovered active signal (``PENDING_ENTRY`` or ``OPEN``)
    for ``RECOVERED_OPEN`` — its ``status`` field tells the caller which monitor
    path to resume — and is ``None`` otherwise. ``errors`` carries safe
    diagnostic text only; raw exceptions are never re-raised.
    """

    outcome: RecoveryOutcome
    signal: Signal | None = None
    message: str = ""
    errors: tuple[str, ...] = ()

    @classmethod
    def recovered(cls, signal: Signal) -> RecoveryResult:
        return cls(
            outcome=RecoveryOutcome.RECOVERED_OPEN,
            signal=signal,
            message=(
                f"recovered active signal {signal.id} ({signal.status}); "
                "AI must not run"
            ),
        )

    @classmethod
    def no_open_signal(cls) -> RecoveryResult:
        return cls(
            outcome=RecoveryOutcome.NO_OPEN_SIGNAL,
            message="no active signal; system may proceed to analysis scheduling later",
        )

    @classmethod
    def error(cls, message: str) -> RecoveryResult:
        return cls(
            outcome=RecoveryOutcome.RECOVERY_ERROR,
            message=message,
            errors=(message,),
        )


class RecoveryService:
    """Reads SQLite on startup and returns the single active signal, if any.

    The service is deliberately read-only: it never calls the AI, never fetches
    Binance data, and never creates/updates signals. It also exposes a monitor
    factory so the caller can immediately resume observation with the existing
    ``SignalMonitor`` (no duplicated monitoring logic): entry monitoring for a
    recovered ``PENDING_ENTRY`` signal, TP/SL monitoring for a recovered OPEN
    signal.
    """

    def __init__(self, repository: SignalRepository) -> None:
        self.repository = repository

    def recover(self) -> RecoveryResult:
        """Determine the startup trading state from SQLite (read-only).

        Returns:
            RecoveryResult: ``RECOVERED_OPEN`` (with the active signal, whose
            ``status`` is ``PENDING_ENTRY`` or ``OPEN``), ``NO_OPEN_SIGNAL``,
            or ``RECOVERY_ERROR`` for database/inconsistent-state failures.
        """
        try:
            pending = self.repository.list_signals(status=STATUS_PENDING_ENTRY)
            opened = self.repository.list_signals(status=STATUS_OPEN)
        except Exception as exc:
            logger.error("recovery failed while reading SQLite: %s", exc)
            return RecoveryResult.error(f"failed to read the active signal: {exc}")

        active = pending + opened
        if not active:
            logger.info("recovery: no active signal; analysis may be scheduled later")
            return RecoveryResult.no_open_signal()

        if len(active) > 1:
            # Invariant violation. Never silently choose one of them.
            ids = ", ".join(str(signal.id) for signal in active)
            logger.error(
                "recovery: inconsistent state; %d active signals found: [%s]",
                len(active),
                ids,
            )
            return RecoveryResult.error(
                f"inconsistent state: {len(active)} active signals found; "
                f"refusing to recover a signal"
            )

        signal = active[0]
        if signal.status == STATUS_PENDING_ENTRY and signal.opened_at is not None:
            # A pending signal must not carry an open timestamp; a corrupted row
            # must never be silently "fixed".
            logger.error(
                "recovery: PENDING_ENTRY signal %s has an opened_at timestamp", signal.id
            )
            return RecoveryResult.error(
                f"recovered PENDING_ENTRY signal {signal.id} is inconsistent: "
                f"it already has an opened_at timestamp"
            )
        if not self._is_valid(signal):
            logger.error(
                "recovery: active signal %s violates the signal invariant", signal.id
            )
            return RecoveryResult.error(
                f"recovered active signal {signal.id} is inconsistent with the "
                f"signal ordering rules"
            )

        logger.info(
            "recovery: active signal %s (%s %s, status %s) recovered; AI must not run",
            signal.id,
            signal.direction,
            signal.symbol,
            signal.status,
        )
        return RecoveryResult.recovered(signal)

    def resume_monitor(
        self, market_data=None, *, candle_limit: int = DEFAULT_CANDLE_LIMIT
    ) -> SignalMonitor:
        """Build a :class:`SignalMonitor` bound to the same repository.

        The monitor re-reads the active signal from SQLite on every poll, so it
        automatically observes a signal recovered by this service — promoting a
        PENDING_ENTRY at its entry or closing an OPEN signal at its exit, or
        observing a concurrently modified state — without duplicating logic.
        """
        return SignalMonitor(self.repository, market_data=market_data, candle_limit=candle_limit)

    def _is_valid(self, signal: Signal) -> bool:
        """True when the persisted active signal still satisfies the invariant."""
        if signal.direction not in DIRECTIONS:
            return False
        if not signal.symbol or not signal.symbol.strip():
            return False
        try:
            validate_signal_prices(
                signal.direction, signal.entry, signal.stop_loss, signal.take_profit
            )
        except SignalValidationError:
            return False
        return True


__all__ = [
    "RecoveryOutcome",
    "RecoveryResult",
    "RecoveryService",
]
