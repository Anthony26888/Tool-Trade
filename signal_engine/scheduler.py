"""Phase 10: the 1H analysis scheduler (OneHourScheduler).

The scheduler is the ONLY component allowed to trigger AI analysis. It runs at
most once per *closed* 1H candle and only while no active signal (``PENDING_ENTRY``
or ``OPEN``) exists. It is deliberately independent from the TP/SL monitor
(``signal_engine/monitor.py``), which observes 1m data and never analyzes.

Pipeline of a single ``tick()``::

    Detect latest closed 1H candle vs the persisted marker
        -> NO_NEW_CANDLE / ALREADY_PROCESSED / DATA_UNAVAILABLE / proceed
    Active-signal gate (repo read, fresh)
        -> BLOCKED_ACTIVE_SIGNAL
    Fetch >= MIN_REQUIRED_CANDLES closed 1H candles
        -> DATA_UNAVAILABLE on failure
    Deterministic indicators              (binance.indicators)
    AnalysisContext -> LLM -> SignalAnalysis
        -> DATA_UNAVAILABLE (context), ERROR (LLM/parsing), never a signal
    Signal Engine validation + persist    (PENDING_ENTRY) or WAIT
    Mark the candle processed             (scheduler_state, SQLite)
        -> idempotency + restart safety

Guarantees
----------
- Closed-candle rule: detection and the analysis window use only CLOSED candles
  (``fetch_closed_klines``). A forming 1H candle never reaches the AI.
- Idempotency: the same candle is analyzed at most once per process lifetime
  and across restarts, thanks to the SQLite marker row. Multiple ``tick()``
  calls for one candle yield exactly one AI invocation.
- AI lock: when ``PENDING_ENTRY`` or ``OPEN`` exists, the scheduler returns
  ``BLOCKED_ACTIVE_SIGNAL`` BEFORE any market-data fetch and never calls the
  LLM. Only after the position is closed (TP/SL) may a later candle be analyzed.
- Failure safety: a market-data/indicator/context failure is
  ``DATA_UNAVAILABLE`` and an LLM/database failure is ``ERROR``; neither marks
  the candle processed, and a failure is never reported as WAIT.
- Timezone: only timezone-aware UTC datetimes are accepted; naive datetimes are
  rejected so no naive-vs-aware comparison is possible.
- No scheduling of the TP/SL monitor and no orders: this module only decides
  when the AI may analyze a closed 1H candle. Telegram notifications (Phase 11)
  are emitted for a newly created PENDING_ENTRY signal only, after it persists.

Concurrency
-----------
A module-level lock serializes the analyze+persist section across threads (and
across scheduler instances) within one process, so concurrent ``tick()`` calls
for the same candle produce exactly one AI call. The authoritative
single-active-signal guarantee remains the repository's ``BEGIN IMMEDIATE``
create/transition (in-process and out-of-process).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from binance.indicators import (
    MIN_REQUIRED_CANDLES,
    IndicatorError,
    compute_indicator_matrix,
)
from binance.market_data import (
    DEFAULT_SYMBOL,
    BinanceError,
    BinanceMarketData,
    Candle,
    validate_interval,
    validate_symbol,
)
from database.database import SignalRepository
from database.models import Signal

from .analysis import SignalAnalysis, SignalAnalysisError, analyze_signal
from .context import DEFAULT_MAX_CANDLES, AnalysisContextError
from .engine import SignalEngine, SignalOutcome
from .llm import LLMConfig, llm_config_from_env
from .multiagent import resolve_analysis_mode
from .state import SignalState

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from .telegram import TelegramNotifier

logger = logging.getLogger(__name__)

#: The AI analysis timeframe. Only closed candles on this interval are analyzed.
SCHEDULER_INTERVAL = "1h"

#: How many closed 1H candles the analysis window fetches. Must be enough for
#: the full indicator set (EMA200 needs 200 candles); +20 gives headroom.
DEFAULT_WINDOW_CANDLES = MIN_REQUIRED_CANDLES + 20

#: Detection fetch size: only the newest CLOSED candle is inspected, so a small
#: window covers kline-boundary edge cases (like the 1m monitor).
DEFAULT_DETECTION_LIMIT = 5

#: Default wall-clock delay between ``run_forever`` iterations. ``tick()`` is
#: non-blocking; real cadence is driven by candle-close timestamps, not sleep.
DEFAULT_POLL_INTERVAL = 30.0

#: Serializes the analyze+persist section so concurrent ticks for one candle
#: yield exactly one AI call within a process.
_AI_LOCK = threading.Lock()


def as_epoch_ms(now: datetime) -> int:
    """Convert a timezone-aware ``now`` to epoch milliseconds (UTC).

    A naive datetime is rejected: the scheduler never compares naive and
    aware timestamps.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware UTC (e.g. datetime.now(timezone.utc))")
    return int(now.timestamp() * 1000)


def iso_to_epoch_ms(value: str) -> int:
    """Parse an ISO-8601 UTC timestamp (ending ``Z``) to epoch milliseconds."""
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def as_iso(now: datetime) -> str:
    """Render a timezone-aware ``now`` as ISO-8601 UTC with millisecond precision."""
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SchedulerOutcome(str, Enum):
    """Structured outcome vocabulary of a single scheduler ``tick()``."""

    NO_NEW_CANDLE = "NO_NEW_CANDLE"
    ALREADY_PROCESSED = "ALREADY_PROCESSED"
    BLOCKED_ACTIVE_SIGNAL = "BLOCKED_ACTIVE_SIGNAL"
    WAIT = "WAIT"
    CREATED = "CREATED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SchedulerResult:
    """Result of one scheduler tick, without leaking raw implementation exceptions.

    ``candle_timestamp`` is the open time (epoch ms) of the closed 1H candle
    this tick considered; ``signal`` is the newly created ``PENDING_ENTRY``
    signal for ``CREATED`` and ``None`` otherwise.
    """

    outcome: SchedulerOutcome
    candle_timestamp: int | None = None
    signal: Signal | None = None
    message: str = ""
    errors: tuple[str, ...] = ()

    @classmethod
    def no_new_candle(cls, candle_ts: int) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.NO_NEW_CANDLE,
            candle_timestamp=candle_ts,
            message="the latest closed 1H candle is at or behind the last processed candle",
        )

    @classmethod
    def already_processed(cls, candle_ts: int) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.ALREADY_PROCESSED,
            candle_timestamp=candle_ts,
            message=f"candle {candle_ts} was already analyzed; no AI run for it again",
        )

    @classmethod
    def blocked(cls, signal: Signal, candle_ts: int) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL,
            candle_timestamp=candle_ts,
            signal=signal,
            message=(
                f"active signal {signal.id} ({signal.status}) exists; the AI is "
                "locked until the position is closed"
            ),
        )

    @classmethod
    def wait(cls, candle_ts: int) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.WAIT,
            candle_timestamp=candle_ts,
            message=f"AI returned WAIT for candle {candle_ts}; no signal created",
        )

    @classmethod
    def created(cls, candle_ts: int, signal: Signal) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.CREATED,
            candle_timestamp=candle_ts,
            signal=signal,
            message="signal created and persisted as PENDING_ENTRY (awaiting its Entry trigger)",
        )

    @classmethod
    def data_unavailable(cls, message: str, candle_ts: int | None = None) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.DATA_UNAVAILABLE,
            candle_timestamp=candle_ts,
            message=message,
            errors=(message,),
        )

    @classmethod
    def error(cls, message: str, candle_ts: int | None = None) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.ERROR,
            candle_timestamp=candle_ts,
            message=message,
            errors=(message,),
        )


class OneHourScheduler:
    """Decides when the AI may analyze a closed 1H candle, and drives the pipeline.

    ``market_data`` must expose Phase 1's ``fetch_closed_klines`` contract and
    is injected for deterministic tests. ``analyzer`` must accept ``(candles,
    indicators)`` positionally and return a Phase 5 ``SignalAnalysis``; the
    default builds one from the configured LLM. ``engine`` is the Phase 6
    validation/persistence boundary (never duplicated here). ``notifier`` is an
    optional Phase 11 ``TelegramNotifier``; when provided, a NEW PENDING_ENTRY
    signal is announced once, only after it has been persisted.
    """

    def __init__(
        self,
        repository: SignalRepository,
        *,
        market_data: BinanceMarketData | None = None,
        engine: SignalEngine | None = None,
        state: SignalState | None = None,
        analyzer: Callable[[list[Candle], Any], SignalAnalysis] | None = None,
        config: LLMConfig | None = None,
        symbol: str | Callable[[], str] = DEFAULT_SYMBOL,
        timeframe: str = SCHEDULER_INTERVAL,
        candle_limit: int | None = None,
        detection_limit: int = DEFAULT_DETECTION_LIMIT,
        max_candles: int = DEFAULT_MAX_CANDLES,
        notifier: TelegramNotifier | None = None,
    ) -> None:
        if candle_limit is None:
            candle_limit = DEFAULT_WINDOW_CANDLES
        if not isinstance(candle_limit, int) or candle_limit < MIN_REQUIRED_CANDLES:
            raise ValueError(
                f"candle_limit must be >= {MIN_REQUIRED_CANDLES} so the EMA200 "
                "indicator set warms up"
            )
        if not isinstance(detection_limit, int) or detection_limit <= 0:
            raise ValueError("detection_limit must be a positive integer")
        if not isinstance(max_candles, int) or max_candles <= 0:
            raise ValueError("max_candles must be a positive integer")

        self.repository = repository
        self.state = state if state is not None else SignalState(repository)
        self.engine = engine if engine is not None else SignalEngine(repository, self.state)
        self.market_data = BinanceMarketData() if market_data is None else market_data
        self.config = config if config is not None else llm_config_from_env()
        self.analyzer = analyzer if analyzer is not None else self._default_analyzer
        if callable(symbol):
            self._symbol_source = symbol
        else:
            self._symbol_source = validate_symbol(symbol)
        #: Resolves the analyzed symbol once per tick (set by ``_tick_locked``).
        self._tick_symbol: str | None = None
        self.timeframe = validate_interval(timeframe)
        self.candle_limit = candle_limit
        self.detection_limit = detection_limit
        self.max_candles = max_candles
        self.notifier = notifier

    @property
    def symbol(self) -> str:
        """The analysed symbol.

        When the constructor received a plain string this returns it exactly as
        before; a callable (the daemon's resolved, DB>env symbol) is invoked
        once per tick under the AI lock so a Settings change to a new symbol
        takes effect for the next eligible analysis.
        """
        source = self._tick_symbol
        if source is None:
            source = self._symbol_source
            if callable(source):
                source = source()
        return validate_symbol(source)

    def _default_analyzer(
        self, candles: list[Candle], indicators: Any
    ) -> SignalAnalysis:
        """Analyze via the shared Phase 5 entry point (builds its own context)."""
        mode = "multi" if resolve_analysis_mode() else "single"
        return analyze_signal(
            self.config,
            candles,
            indicators,
            symbol=self.symbol,
            timeframe=self.timeframe,
            max_candles=self.max_candles,
            mode=mode,
        )

    def tick(self, *, now: datetime | None = None) -> SchedulerResult:
        """Run one non-blocking scheduler pass.

        ``now`` is a timezone-aware UTC datetime (injectable for determinism);
        it defaults to the current UTC time. Returns a :class:`SchedulerResult`
        and never raises for expected market-data/LLM/database failures.
        """
        now = now if now is not None else datetime.now(timezone.utc)
        self._tick_symbol = None
        with _AI_LOCK:
            try:
                return self._tick_locked(as_epoch_ms(now), as_iso(now))
            finally:
                self._tick_symbol = None

    def _tick_locked(self, now_ms: int, now_iso: str) -> SchedulerResult:
        """The serialized tick body; the caller holds ``_AI_LOCK``."""
        # Resolve the analyzed symbol once for this whole tick so detection,
        # the analysis window, the marker and the signal all agree.
        self._tick_symbol = self.symbol
        detection = self._detect(now_ms)
        if isinstance(detection, SchedulerResult):
            return detection
        detection_ts = detection

        # The AI lock: while PENDING_ENTRY or OPEN exists, never fetch for
        # analysis and never call the LLM (checked again by the engine later).
        active = self.state.active_signal()
        if active is not None:
            logger.info(
                "[Scheduler] blocked: active signal %s (%s); AI must not run "
                "until the position is closed",
                active.id,
                active.status,
            )
            return SchedulerResult.blocked(active, detection_ts)

        try:
            window = self.market_data.fetch_closed_klines(
                self.symbol,
                self.timeframe,
                limit=self.candle_limit,
                now_ms=now_ms,
            )
        except BinanceError as exc:
            logger.error(
                "[Scheduler] data unavailable for %s %s: %s: %s",
                self.symbol,
                self.timeframe,
                exc.__class__.__name__,
                exc,
            )
            return SchedulerResult.data_unavailable(
                f"{exc.__class__.__name__}: {exc}", detection_ts
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error("[Scheduler] unexpected market-data failure: %s", exc)
            return SchedulerResult.error(
                f"unexpected market-data failure: {exc}", detection_ts
            )

        if len(window) < MIN_REQUIRED_CANDLES:
            logger.warning(
                "[Scheduler] only %d closed candles available; %d are required "
                "for the full indicator set",
                len(window),
                MIN_REQUIRED_CANDLES,
            )
            return SchedulerResult.data_unavailable(
                f"only {len(window)} closed {self.timeframe} candles are available; "
                f"at least {MIN_REQUIRED_CANDLES} are required for the full "
                "indicator set",
                detection_ts,
            )

        # The analysis candle is the newest closed candle actually handed to the
        # AI; if a newer candle closed between detection and this fetch, the
        # marker records THIS candle so processing stays consistent.
        analysis_ts = window[-1].timestamp

        try:
            indicators = compute_indicator_matrix(window)
        except IndicatorError as exc:
            logger.error(
                "[Scheduler] indicator calculation failed for candle %s: %s: %s",
                analysis_ts,
                exc.__class__.__name__,
                exc,
            )
            return SchedulerResult.data_unavailable(
                f"indicator calculation failed: {exc.__class__.__name__}: {exc}",
                detection_ts,
            )

        logger.info("[Scheduler] analyzing closed %s candle %s", self.timeframe, analysis_ts)
        try:
            analysis = self.analyzer(window, indicators)
        except AnalysisContextError as exc:
            logger.error(
                "[Scheduler] context invalid for candle %s: %s", analysis_ts, exc
            )
            return SchedulerResult.data_unavailable(
                f"analysis context invalid: {exc}", analysis_ts
            )
        except SignalAnalysisError as exc:
            logger.error("[Scheduler] AI analysis failed for candle %s: %s", analysis_ts, exc)
            return SchedulerResult.error(f"AI analysis failed: {exc}", analysis_ts)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error(
                "[Scheduler] AI analysis failed unexpectedly for candle %s: %s",
                analysis_ts,
                exc,
            )
            return SchedulerResult.error(
                f"AI analysis failed unexpectedly: {exc}", analysis_ts
            )
        logger.info("[Scheduler] AI analysis finished for candle %s", analysis_ts)

        try:
            engine_result = self.engine.process(analysis)
        except Exception as exc:
            logger.error(
                "[Scheduler] signal engine failed for candle %s: %s", analysis_ts, exc
            )
            return SchedulerResult.error(
                f"signal engine failed: {exc}", analysis_ts
            )

        if engine_result.outcome is SignalOutcome.CREATED:
            if engine_result.signal is None:
                logger.error(
                    "[Scheduler] engine reported CREATED for candle %s without a signal",
                    analysis_ts,
                )
                return SchedulerResult.error(
                    "engine reported CREATED without a signal", analysis_ts
                )
            marked = self._mark_processed(analysis_ts, now_iso)
            if not marked:
                # The signal itself is already persisted; the repository's
                # single-active invariant prevents a duplicate on the next
                # tick, so a marker-write failure must not report failure.
                logger.error(
                    "[Scheduler] signal %s created but failed to record candle %s "
                    "as processed; the single-active invariant prevents a duplicate",
                    engine_result.signal.id,
                    analysis_ts,
                )
            logger.info(
                "[Scheduler] signal %s created for candle %s (PENDING_ENTRY)",
                engine_result.signal.id,
                analysis_ts,
            )
            self._notify_created(engine_result.signal, analysis_ts)
            return SchedulerResult.created(analysis_ts, engine_result.signal)

        if engine_result.outcome is SignalOutcome.WAIT:
            marked = self._mark_processed(analysis_ts, now_iso)
            if not marked:
                # A WAIT produces no signal to fall back on: without the marker
                # the candle would be re-analyzed and the LLM called again.
                logger.error(
                    "[Scheduler] WAIT for candle %s but failed to record it as "
                    "processed; refusing to report success",
                    analysis_ts,
                )
                return SchedulerResult.error(
                    "AI returned WAIT but the candle could not be recorded as "
                    "processed; the analysis will be retried",
                    analysis_ts,
                )
            logger.info("[Scheduler] AI returned WAIT for candle %s", analysis_ts)
            return SchedulerResult.wait(analysis_ts)

        if engine_result.outcome is SignalOutcome.BLOCKED_OPEN_SIGNAL:
            # A concurrent creator (another process) won the race; the safe
            # re-read reports the blocking signal instead of inventing an error.
            blocking = self.state.active_signal()
            if blocking is not None:
                return SchedulerResult.blocked(blocking, analysis_ts)
            logger.error(
                "[Scheduler] race: create blocked for candle %s but no active "
                "signal is visible: %s",
                analysis_ts,
                engine_result.message,
            )
            return SchedulerResult.error(
                f"create blocked by another active signal (race): {engine_result.message}",
                analysis_ts,
            )

        logger.error("[Scheduler] analysis rejected/failed for candle %s: %s", analysis_ts, engine_result.message)
        return SchedulerResult.error(
            engine_result.message or "signal was not created", analysis_ts
        )

    def _detect(self, now_ms: int) -> int | SchedulerResult:
        """Return the newest closed candle's open time to proceed with, or a result.

        A proceed value means the candle is NEWER than the persisted marker (or
        no marker exists yet on a cold start). Equal ⇒ ``ALREADY_PROCESSED``;
        older ⇒ ``NO_NEW_CANDLE`` (data rewound / clock skew).
        """
        try:
            candles = self.market_data.fetch_closed_klines(
                self.symbol,
                self.timeframe,
                limit=self.detection_limit,
                now_ms=now_ms,
            )
        except BinanceError as exc:
            logger.error(
                "[Scheduler] detection failed for %s %s: %s: %s",
                self.symbol,
                self.timeframe,
                exc.__class__.__name__,
                exc,
            )
            return SchedulerResult.data_unavailable(
                f"{exc.__class__.__name__}: {exc}"
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error("[Scheduler] unexpected detection failure: %s", exc)
            return SchedulerResult.error(f"unexpected detection failure: {exc}")

        if not candles:
            return SchedulerResult.data_unavailable(
                f"no closed {self.timeframe} candle available for detection"
            )

        detection_ts = candles[-1].timestamp
        marker = self._last_processed_candle()
        if marker is not None:
            if detection_ts == marker:
                return SchedulerResult.already_processed(detection_ts)
            if detection_ts < marker:
                logger.warning(
                    "[Scheduler] market data behind the last processed candle "
                    "(market %s < marker %s); ignoring",
                    detection_ts,
                    marker,
                )
                return SchedulerResult.no_new_candle(detection_ts)
        return detection_ts

    def _last_processed_candle(self) -> int | None:
        """The persisted marker, or a best-effort seed from the latest signal.

        On a database that predates the scheduler (no marker row), the marker is
        seeded from the most recent signal's ``market_timestamp`` so a candle
        that already produced a signal is never re-analyzed after an upgrade.
        """
        marker = self._scheduler_state().last_processed_candle(self.symbol, self.timeframe)
        if marker is not None:
            return marker
        latest = self.repository.list_signals(symbol=self.symbol, limit=1)
        if not latest:
            return None
        market_timestamp = latest[0].market_timestamp
        if not market_timestamp:
            return None
        try:
            return iso_to_epoch_ms(market_timestamp)
        except ValueError:
            logger.warning(
                "[Scheduler] could not parse latest signal market_timestamp "
                "%r; starting from the current closed candle",
                market_timestamp,
            )
            return None

    def _mark_processed(self, candle_ts: int, processed_at: str) -> bool:
        """Persist ``candle_ts`` as processed; True on success."""
        try:
            self._scheduler_state().record_processed_candle(
                self.symbol, self.timeframe, candle_ts, processed_at
            )
        except Exception as exc:
            logger.error(
                "[Scheduler] failed to record candle %s as processed: %s", candle_ts, exc
            )
            return False
        return True

    def _notify_created(self, signal: Signal, candle_ts: int) -> None:
        """Announce a persisted PENDING_ENTRY signal once (never raises).

        Runs strictly after the signal row AND the processed-candle marker are
        committed, so a duplicate tick can never re-announce. A Telegram
        failure is reduced to a warning: the signal is already safely open.
        """
        if self.notifier is None:
            return
        try:
            self.notifier.notify_signal_created(signal, candle_ts=candle_ts)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning(
                "Telegram notification failed after signal %s was created: %s",
                signal.id,
                exc,
            )

    def _scheduler_state(self):
        """Lazily built store (kept as an instance field for reuse)."""
        if getattr(self, "_state_store", None) is None:
            from database.database import SchedulerStateRepository

            self._state_store = SchedulerStateRepository(self.repository.database)
        return self._state_store

    def run_forever(self, interval: float = DEFAULT_POLL_INTERVAL) -> None:
        """Blocking loop: ``tick()`` every ``interval`` seconds until interrupted.

        ``tick()`` is non-blocking; the cadence is anchored to candle-close
        timestamps, so this loop merely polls for a newly closed 1H candle.
        """
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError("interval must be a positive number of seconds")
        logger.info("[Scheduler] starting run_forever with %ss poll interval", interval)
        while True:
            result = self.tick()
            logger.info("[Scheduler] tick result: %s", result.outcome.value)
            time.sleep(interval)


__all__ = [
    "DEFAULT_DETECTION_LIMIT",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_WINDOW_CANDLES",
    "OneHourScheduler",
    "SCHEDULER_INTERVAL",
    "SchedulerOutcome",
    "SchedulerResult",
    "as_epoch_ms",
    "as_iso",
    "iso_to_epoch_ms",
]
