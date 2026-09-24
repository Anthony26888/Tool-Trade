"""Entry + TP/SL Monitor for the active BTCUSDT signal (Phases 7 and 9.2).

The monitor is an independent observer: it reads the single active signal from
SQLite (Phase 3 ``SignalRepository``), watches Binance public 1-minute market
data (Phase 1), and drives the signal through its lifecycle atomically via the
repository state machine:

- ``PENDING_ENTRY``: awaited until the configured Entry price is touched, then
  promoted to ``OPEN`` (``ENTRY_HIT``). The entry must be touched by a CLOSED
  1m candle; a forming candle never confirms.
- ``OPEN``: closed at its configured exit (``TP_HIT``/``SL_HIT``) exactly as in
  Phase 7.

The monitor NEVER analyzes: it does not call the LLM, does not invoke the
Signal Engine, does not recalculate or modify ``entry``/``stop_loss``/
``take_profit``, and places no Binance orders.

Data source and the closed-candle rule
--------------------------------------
Basis data: ``BinanceMarketData.fetch_klines`` for ``1m`` (the public klines
endpoint). Only candles whose ``is_closed`` flag is set (derived by Phase 1
parsing from ``now_ms``) are eligible. The currently-forming 1m candle is never
used for OHLC-based entry or exit confirmation; if no closed 1m candle is
available the monitor returns ``DATA_UNAVAILABLE``. The monitor evaluates ONLY
the most recent closed 1m candle and never looks ahead into future candles. It
is fully independent from the 1H AI-analysis cycle and never uses a forming 1H
candle.

Entry trigger (``Decimal`` comparisons only, never binary float)
-----------------------------------------------------------------
LONG:  entry touched when ``high >= entry`` (``high == entry`` counts).
SHORT: entry touched when ``low  <= entry`` (``low  == entry`` counts).
When the trigger candle ALSO touches either exit level (TP or SL) in the same
candle, the intrabar order is unknowable from OHLC alone: the monitor returns
``AMBIGUOUS``, does NOT promote the signal, and keeps monitoring.

Same-candle exit ambiguity
--------------------------
A closed 1m candle whose OHLC touches BOTH the take-profit and the stop-loss
cannot be resolved from OHLC alone (the intrabar order is unknown). The monitor
must not guess: it returns ``AMBIGUOUS``, does NOT close the signal, and keeps
monitoring, so a later unambiguous candle can still resolve the exit. Later
backtesting phases may resolve intrabar order with finer-grained data; the
live/demo monitor deliberately keeps this conservative deterministic behavior.

Exit rules (``Decimal`` comparisons only, never binary float)
-------------------------------------------------------------
LONG:  TP_HIT when ``high >= take_profit``; SL_HIT when ``low <= stop_loss``.
SHORT: TP_HIT when ``low  <= take_profit``; SL_HIT when ``high >= stop_loss``.

Open record (PENDING_ENTRY -> OPEN)
-----------------------------------
- status: ``PENDING_ENTRY -> OPEN`` (atomic repository transition).
- opened_at: ISO-8601 UTC timestamp written by the repository transition.
- ``entry``/``stop_loss``/``take_profit``/``direction``/analysis metadata are
  never modified.

Close record (OPEN -> TP_HIT / SL_HIT)
--------------------------------------
- status: ``OPEN -> TP_HIT`` or ``OPEN -> SL_HIT`` (atomic repository transition).
- close_price: the configured ``take_profit`` price on TP_HIT and the
  configured ``stop_loss`` price on SL_HIT — never an invented intrabar price.
- close_reason: ``TP`` / ``SL``.
- closed_at: ISO-8601 UTC timestamp written by the repository transition.
- ``entry``/``stop_loss``/``take_profit``/``direction`` (and analysis metadata)
  are never modified.

Safety
------
- No active signal exists -> ``NO_ACTIVE_SIGNAL``; nothing is fetched, nothing
  is created, nothing is closed, and no AI is invoked.
- A terminal (already closed) signal is never reopened.
- A ``PENDING_ENTRY`` signal never closes straight to TP/SL: if no entry touch
  is ever confirmed the signal stays pending rather than inventing an exit.
- Missing/malformed/failed market data -> ``DATA_UNAVAILABLE``/``ERROR``; the
  signal never changes state on missing or invalid data.
- Concurrency: the repository ``BEGIN IMMEDIATE`` transitions are atomic; a
  lost race observes the already-changed state and never creates a second
  conflicting position.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING

from binance.market_data import BinanceError, BinanceMarketData, validate_symbol
from database.database import SignalRepository
from database.models import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    STATUS_OPEN,
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    InvalidTransitionError,
    Signal,
    SignalNotFoundError,
    SignalValidationError,
)

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from .telegram import TelegramNotifier

logger = logging.getLogger(__name__)

#: The observation timeframe. Entry/exit confirmation is deliberately
#: independent from the 1H AI analysis cycle.
MONITOR_INTERVAL = "1m"

#: How many 1m klines are fetched per poll. The monitor keeps only the newest
#: CLOSED candle, so the extra candles cover kline-boundary edge cases.
DEFAULT_CANDLE_LIMIT = 5

CLOSE_REASON_TP = "TP"
CLOSE_REASON_SL = "SL"

#: Reason strings for the Phase 11 ambiguous-candle Telegram warning. These are
#: human-readable and share no text with the internal MonitorResult messages.
AMBIGUITY_REASON_ENTRY = (
    "The candle touched the entry and an exit level in the same candle; the "
    "execution order is unknown, so the signal stayed PENDING_ENTRY."
)
AMBIGUITY_REASON_EXIT = (
    "The candle touched both the take-profit and the stop-loss in the same "
    "candle; the execution order is unknown, so nothing was closed."
)


class MonitorOutcome(str, Enum):
    """Structured outcome vocabulary of a single monitor poll."""

    NO_ACTIVE_SIGNAL = "NO_ACTIVE_SIGNAL"
    #: Backward-compatible alias for Phase 7 callers.
    NO_OPEN_SIGNAL = "NO_ACTIVE_SIGNAL"
    ENTRY_HIT = "ENTRY_HIT"
    MONITORING = "MONITORING"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    AMBIGUOUS = "AMBIGUOUS"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class MonitorResult:
    """Result of a monitor poll, without leaking raw implementation exceptions.

    ``signal`` is the considered active signal for MONITORING/AMBIGUOUS, the
    promoted OPEN signal for ENTRY_HIT, and the resulting CLOSED signal for
    TP_HIT/SL_HIT; otherwise ``None``.
    """

    outcome: MonitorOutcome
    signal: Signal | None = None
    message: str = ""
    errors: tuple[str, ...] = ()

    @classmethod
    def no_active_signal(cls) -> MonitorResult:
        return cls(
            outcome=MonitorOutcome.NO_ACTIVE_SIGNAL,
            message="no active signal (PENDING_ENTRY or OPEN); nothing to monitor, "
            "nothing created, nothing closed",
        )

    @classmethod
    def no_open_signal(cls) -> MonitorResult:
        """Backward-compatible alias of :meth:`no_active_signal`."""
        return cls.no_active_signal()

    @classmethod
    def monitoring(cls, signal: Signal) -> MonitorResult:
        return cls(
            outcome=MonitorOutcome.MONITORING,
            signal=signal,
            message="no entry/exit touch in the latest closed 1m candle",
        )

    @classmethod
    def entry_hit(cls, signal: Signal) -> MonitorResult:
        return cls(
            outcome=MonitorOutcome.ENTRY_HIT,
            signal=signal,
            message=f"signal {signal.id} promoted from PENDING_ENTRY to OPEN; entry touched",
        )

    @classmethod
    def closed(cls, outcome: MonitorOutcome, signal: Signal) -> MonitorResult:
        return cls(
            outcome=outcome,
            signal=signal,
            message=f"signal {signal.id} closed as {outcome.value}",
        )

    @classmethod
    def ambiguous(cls, signal: Signal) -> MonitorResult:
        return cls(
            outcome=MonitorOutcome.AMBIGUOUS,
            signal=signal,
            message="one closed 1m candle touched both TP and SL; execution "
            "order is unknown from OHLC data, so nothing was closed",
        )

    @classmethod
    def entry_ambiguous(cls, signal: Signal) -> MonitorResult:
        return cls(
            outcome=MonitorOutcome.AMBIGUOUS,
            signal=signal,
            message="one closed 1m candle touched the entry AND an exit level; "
            "execution order is unknown from OHLC data, so the signal stayed "
            "PENDING_ENTRY",
        )

    @classmethod
    def data_unavailable(cls, message: str) -> MonitorResult:
        return cls(
            outcome=MonitorOutcome.DATA_UNAVAILABLE,
            message=message,
            errors=(message,),
        )

    @classmethod
    def error(cls, message: str) -> MonitorResult:
        return cls(outcome=MonitorOutcome.ERROR, message=message, errors=(message,))


def evaluate_candle(signal: Signal, high: Decimal, low: Decimal) -> MonitorOutcome:
    """Classify one closed candle against the signal's exit levels.

    Prices MUST be ``Decimal``; the caller performs the OHLC conversion. Returns
    ``AMBIGUOUS`` when the same candle touches both levels (order unknowable),
    ``TP_HIT``/``SL_HIT`` for a clean contact, or ``MONITORING`` for no contact.
    """
    if signal.direction == DIRECTION_LONG:
        tp_hit = high >= signal.take_profit
        sl_hit = low <= signal.stop_loss
    elif signal.direction == DIRECTION_SHORT:
        tp_hit = low <= signal.take_profit
        sl_hit = high >= signal.stop_loss
    else:
        raise SignalValidationError(f"unknown signal direction {signal.direction!r}")
    if tp_hit and sl_hit:
        return MonitorOutcome.AMBIGUOUS
    if tp_hit:
        return MonitorOutcome.TP_HIT
    if sl_hit:
        return MonitorOutcome.SL_HIT
    return MonitorOutcome.MONITORING


def evaluate_entry(signal: Signal, high: Decimal, low: Decimal) -> MonitorOutcome:
    """Classify one closed candle against a PENDING_ENTRY signal's entry level.

    Prices MUST be ``Decimal``; the caller performs the OHLC conversion.
    LONG entry is touched when ``high >= entry`` (``==`` counts); SHORT when
    ``low <= entry`` (``==`` counts). If the same candle ALSO touches either
    configured exit level (TP or SL) the execution order is unknowable from
    OHLC alone, so it returns ``AMBIGUOUS`` (the signal stays PENDING_ENTRY)
    rather than inventing an intrabar order. ``ENTRY_HIT`` only when the entry
    is touched cleanly; ``MONITORING`` otherwise.
    """
    if signal.direction == DIRECTION_LONG:
        entry_hit = high >= signal.entry
        touched_exit = high >= signal.take_profit or low <= signal.stop_loss
    elif signal.direction == DIRECTION_SHORT:
        entry_hit = low <= signal.entry
        touched_exit = low <= signal.take_profit or high >= signal.stop_loss
    else:
        raise SignalValidationError(f"unknown signal direction {signal.direction!r}")
    if entry_hit and touched_exit:
        return MonitorOutcome.AMBIGUOUS
    if entry_hit:
        return MonitorOutcome.ENTRY_HIT
    return MonitorOutcome.MONITORING


class SignalMonitor:
    """Observes public 1m market data and drives the active signal's lifecycle.

    A ``PENDING_ENTRY`` signal is promoted to ``OPEN`` when its Entry is touched
    by a closed 1m candle; an ``OPEN`` signal is closed at its TP/SL exit.
    ``market_data`` must expose Phase 1's ``fetch_klines`` contract; it is
    injected so tests never touch the network. ``candle_limit`` bounds the 1m
    fetch window (only the newest CLOSED candle is evaluated).

    ``notifier`` is optional (``TelegramNotifier`` or any object exposing the
    ``notify_signal_opened`` / ``notify_signal_tp`` / ``notify_signal_sl`` /
    ``notify_ambiguous`` methods). When provided, notifications fire ONLY after
    a successful repository transition — the concurrent loser never re-emits —
    and a notification failure never affects the transition result.
    """

    def __init__(
        self,
        repository: SignalRepository,
        market_data: BinanceMarketData | None = None,
        *,
        candle_limit: int = DEFAULT_CANDLE_LIMIT,
        notifier: TelegramNotifier | None = None,
        symbol: str | Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(candle_limit, int) or candle_limit <= 0:
            raise ValueError("candle_limit must be a positive integer")
        self.repository = repository
        self.market_data = BinanceMarketData() if market_data is None else market_data
        self.candle_limit = candle_limit
        self.notifier = notifier
        #: Which symbol this monitor owns. ``None`` keeps the legacy global
        #: behavior (single-symbol deployments, all existing tests). A plain
        #: string is validated now; a callable is resolved on every poll so a
        #: Settings symbol switch applies without a restart (plan B').
        if symbol is None or callable(symbol):
            self._symbol_source = symbol
        else:
            self._symbol_source = validate_symbol(symbol)

    def _symbol(self) -> str | None:
        """Resolve the owned symbol (None = legacy global scope)."""
        source = self._symbol_source
        if source is None:
            return None
        if callable(source):
            return validate_symbol(source())
        return source

    def _active(self):
        """Read the owned active signal (None = idle)."""
        symbol = self._symbol()
        if symbol is None:
            return self.repository.get_active_signal()
        return self.repository.get_active_signal_for_symbol(symbol)

    def poll(self, *, now_ms: int | None = None) -> MonitorResult:
        """Check the active signal against the latest closed 1m candle."""
        try:
            active = self._active()
        except Exception as exc:
            logger.warning("[Monitor] symbol resolution failed: %s", exc)
            return MonitorResult.no_active_signal()
        if active is None:
            # Idle: no fetch, no LLM, no signal creation — the critical rule.
            return MonitorResult.no_active_signal()

        try:
            candles = self.market_data.fetch_klines(
                active.symbol,
                MONITOR_INTERVAL,
                limit=self.candle_limit,
                now_ms=now_ms,
            )
        except BinanceError as exc:
            # Covers EmptyKlineError, MalformedKlineError, HTTP, rate-limit,
            # server, and connection failures raised by the Phase 1 adapter.
            return MonitorResult.data_unavailable(f"{exc.__class__.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive boundary
            return MonitorResult.error(f"unexpected market-data failure: {exc}")

        closed = [candle for candle in candles if candle.is_closed]
        if not closed:
            # The forming 1m candle is never used for OHLC confirmation.
            return MonitorResult.data_unavailable(
                f"no closed {MONITOR_INTERVAL} candle available; the currently "
                "forming candle is never used for OHLC confirmation"
            )
        latest = closed[-1]

        try:
            high = Decimal(str(latest.high))
            low = Decimal(str(latest.low))
        except (InvalidOperation, ValueError, TypeError):
            return MonitorResult.data_unavailable(
                "malformed candle OHLC values; signal state unchanged"
            )

        if active.status == STATUS_PENDING_ENTRY:
            return self._evaluate_pending_entry(active, high, low, latest.timestamp)
        return self._evaluate_open_position(active, high, low, latest.timestamp)

    def _evaluate_pending_entry(
        self, signal: Signal, high: Decimal, low: Decimal, candle_ts: int | None = None
    ) -> MonitorResult:
        """Decide entry promotion for a PENDING_ENTRY signal."""
        try:
            decision = evaluate_entry(signal, high, low)
        except (InvalidOperation, ValueError, TypeError):
            return MonitorResult.data_unavailable(
                "malformed candle OHLC values; signal state unchanged"
            )
        if decision is MonitorOutcome.AMBIGUOUS:
            # The candle touched the entry AND an exit level: never guess.
            self._notify_ambiguous(signal, AMBIGUITY_REASON_ENTRY, candle_ts)
            return MonitorResult.entry_ambiguous(signal)
        if decision is MonitorOutcome.MONITORING:
            return MonitorResult.monitoring(signal)
        return self._open(signal)

    def _evaluate_open_position(
        self, signal: Signal, high: Decimal, low: Decimal, candle_ts: int | None = None
    ) -> MonitorResult:
        """Decide TP/SL closure for an OPEN signal (Phase 7 logic)."""
        try:
            decision = evaluate_candle(signal, high, low)
        except (InvalidOperation, ValueError, TypeError):
            return MonitorResult.data_unavailable(
                "malformed candle OHLC values; signal not closed"
            )
        if decision is MonitorOutcome.AMBIGUOUS:
            # Conservative: never guess the intrabar execution order.
            self._notify_ambiguous(signal, AMBIGUITY_REASON_EXIT, candle_ts)
            return MonitorResult.ambiguous(signal)
        if decision is MonitorOutcome.MONITORING:
            return MonitorResult.monitoring(signal)
        return self._close(signal, decision)

    def _open(self, signal: Signal) -> MonitorResult:
        """Atomically promote PENDING_ENTRY to OPEN at the confirmed entry touch."""
        try:
            opened = self.repository.transition_signal(signal.id, STATUS_OPEN)
        except InvalidTransitionError as exc:
            # A concurrent poll already promoted (or closed) the signal. Never
            # create a second position: re-read and observe the changed state.
            # Scoped to this signal's own symbol so a sibling symbol's
            # position is never mistaken for this one (plan B').
            current = self.repository.get_active_signal_for_symbol(signal.symbol)
            if current is None:
                return MonitorResult.no_active_signal()
            if current.status == STATUS_OPEN:
                return MonitorResult.entry_hit(current)
            return MonitorResult.error(f"could not open signal {signal.id}: {exc}")
        except SignalNotFoundError:
            return MonitorResult.no_active_signal()
        except Exception as exc:
            # A database failure must never look like a successful opening.
            return MonitorResult.error(f"failed to open signal {signal.id}: {exc}")
        self._notify(opened, "notify_signal_opened")
        return MonitorResult.entry_hit(opened)

    def _close(self, signal: Signal, outcome: MonitorOutcome) -> MonitorResult:
        """Atomically transition OPEN to the confirmed terminal state."""
        if outcome is MonitorOutcome.TP_HIT:
            new_status = STATUS_TP_HIT
            close_price = signal.take_profit
            reason = CLOSE_REASON_TP
        else:
            new_status = STATUS_SL_HIT
            close_price = signal.stop_loss
            reason = CLOSE_REASON_SL
        try:
            closed = self.repository.transition_signal(
                signal.id,
                new_status,
                close_price=close_price,
                close_reason=reason,
            )
        except InvalidTransitionError as exc:
            # A concurrent poll already closed the signal into a terminal
            # state. Never overwrite it: if nothing is OPEN now we are done.
            # Scoped to this signal's own symbol (plan B').
            if self.repository.get_open_signal_for_symbol(signal.symbol) is None:
                return MonitorResult.no_active_signal()
            return MonitorResult.error(f"could not close signal {signal.id}: {exc}")
        except SignalNotFoundError:
            return MonitorResult.no_active_signal()
        except Exception as exc:
            # A database failure must never look like a successful close.
            return MonitorResult.error(f"failed to close signal {signal.id}: {exc}")
        if outcome is MonitorOutcome.TP_HIT:
            self._notify(closed, "notify_signal_tp")
        else:
            self._notify(closed, "notify_signal_sl")
        return MonitorResult.closed(outcome, closed)

    def _notify(self, signal: Signal, method_name: str) -> None:
        """Call a notifier method for a just-transitioned signal; failures never raise.

        Notifications are best-effort by design: a Telegram outage must never
        roll back or re-report a successfully transitioned signal, so any
        exception is reduced to a warning log and the transition stands. The
        method is looked up by name so ``notifier=None`` and duck-typed fakes
        both behave: no notification, no error.
        """
        if self.notifier is None:
            return
        method = getattr(self.notifier, method_name, None)
        if method is None:
            return
        try:
            method(signal)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning(
                "Telegram notification failed after signal %s transitioned: %s",
                signal.id,
                exc,
            )

    def _notify_ambiguous(self, signal: Signal, reason: str, candle_ts: int | None) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.notify_ambiguous(signal, reason, candle_ts=candle_ts)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning(
                "Telegram ambiguous warning failed for signal %s: %s", signal.id, exc
            )


__all__ = [
    "CLOSE_REASON_SL",
    "CLOSE_REASON_TP",
    "DEFAULT_CANDLE_LIMIT",
    "MONITOR_INTERVAL",
    "MonitorOutcome",
    "MonitorResult",
    "SignalMonitor",
    "evaluate_candle",
    "evaluate_entry",
]
