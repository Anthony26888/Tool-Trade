"""Signal state gate and transition vocabulary for the Signal Engine (Phase 6).

This module owns the application-level view of trading state that the engine
uses to (a) gate AI invocation and (b) decide whether a signal may be created.

Correctness contract
--------------------
- SQLite is the source of truth. This gate never caches active state in memory:
  every call reads the database through the Phase 3 ``SignalRepository``.
- The atomic guarantee (at most one active signal globally) comes from
  ``SignalRepository.create_signal`` running inside a ``BEGIN IMMEDIATE``
  transaction, never from a Python-level check. The gate here is therefore an
  optimization (avoid burning an LLM call) and a clean API, not the authority.
- Status-transition rules (``PENDING_ENTRY -> OPEN -> TP_HIT/SL_HIT/CANCELLED``,
  and terminal locks) are Phase 3 constants; they are re-exported here so later
  phases and the engine share one vocabulary without duplicating it.
"""

from __future__ import annotations

from binance.market_data import validate_symbol
from database.database import SignalRepository
from database.models import (
    ACTIVE_STATUSES,
    ALLOWED_TRANSITIONS,
    STATUS_CANCELLED,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    TERMINAL_STATUSES,
    Signal,
)


class SignalState:
    """Read-only query + transition delegate over the Phase 3 repository.

    Active means ``PENDING_ENTRY`` or ``OPEN``: while either exists the AI must
    not be invoked and no new signal may be created. ``is_open()``/``open_signal()``
    keep their OPEN-only meaning (used by later position/demo phases); the engine
    and recovery drive the AI lock through ``is_active()``/``active_signal()``.

    Plan B': the ``*_for_symbol`` variants scope the slot per symbol (one
    active signal per symbol). The unsuffixed methods keep the legacy global
    meaning for locks, CLI, and single-symbol compat.
    """

    def __init__(self, repository: SignalRepository) -> None:
        self.repository = repository

    def open_signal(self) -> Signal | None:
        """Return the current OPEN signal, or None when none is OPEN (always fresh)."""
        return self.repository.get_open_signal()

    def open_signal_for(self, symbol: str) -> Signal | None:
        """Return the OPEN signal for one symbol, or None (plan B')."""
        return self.repository.get_open_signal_for_symbol(symbol)

    def active_signal(self) -> Signal | None:
        """Return the single active signal (PENDING_ENTRY or OPEN), or None."""
        return self.repository.get_active_signal()

    def active_signal_for(self, symbol: object) -> Signal | None:
        """Return the active signal for one symbol, or None (plan B').

        A malformed symbol falls back to the global lookup (fail safe toward
        blocking, never toward creating): the engine must never raise for a
        bad analysis when it can safely report BLOCKED instead.
        """
        try:
            return self.repository.get_active_signal_for_symbol(
                validate_symbol(symbol)  # type: ignore[arg-type]
            )
        except Exception:
            return self.repository.get_active_signal()

    def is_open(self) -> bool:
        """True when an OPEN signal exists (a confirmed, entry-touched position)."""
        return self.open_signal() is not None

    def is_active(self) -> bool:
        """True when any active signal exists; the AI must not be invoked then."""
        return self.active_signal() is not None

    def can_open(self) -> bool:
        """True when no active signal exists and a new signal may be considered."""
        return self.active_signal() is None

    def transition(self, signal_id: int, new_status: str, **kwargs) -> Signal:
        """Delegate to the Phase 3 transition rule."""
        return self.repository.transition_signal(signal_id, new_status, **kwargs)

    def cancel(self, signal_id: int, **kwargs) -> Signal:
        """Delegate to the Phase 3 cancel rule."""
        return self.repository.cancel_signal(signal_id, **kwargs)


__all__ = [
    "ACTIVE_STATUSES",
    "ALLOWED_TRANSITIONS",
    "STATUS_CANCELLED",
    "STATUS_OPEN",
    "STATUS_PENDING_ENTRY",
    "TERMINAL_STATUSES",
    "SignalState",
]
