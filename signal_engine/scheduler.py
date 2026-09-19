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
        -> RATE_LIMITED (provider 429/capacity: candle skipped, no retry)
    Signal Engine validation + persist    (PENDING_ENTRY) or WAIT
        -> REJECTED (validation failed: candle recorded, no retry)
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
  ``DATA_UNAVAILABLE`` and an unexpected LLM/database failure is ``ERROR``;
  neither marks the candle processed, and a failure is never reported as
  WAIT. Rejected analyses and provider rate limits ARE recorded as processed
  (``REJECTED``/``RATE_LIMITED``) so the same candle never burns another LLM
  call on the next poll. After ``BTCUSDT_ERROR_BREAKER_N`` consecutive ERROR
  ticks the circuit breaker pauses LLM work for
  ``BTCUSDT_ERROR_BREAKER_COOLDOWN_S`` seconds instead of retrying every
  30s poll into a sick provider.
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

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from binance.htf import HTF_CANDLE_LIMIT, REGIME_NONE, classify_htf
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
from binance.positioning import fetch_positioning, render_note
from database.database import (
    CandleLogRepository,
    RuntimeStateRepository,
    SignalRepository,
)
from database.models import STATUS_PENDING_ENTRY, Signal

from .analysis import RateLimitedError, SignalAnalysis, SignalAnalysisError, analyze_signal
from .context import DEFAULT_MAX_CANDLES, AnalysisContextError
from .engine import SignalEngine, SignalOutcome
from .event_calendar import EventCalendar, EventCheck, format_vn
from .llm import LLMConfig, llm_config_from_env
from .multiagent import resolve_analysis_mode
from .state import SignalState

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from .telegram import TelegramNotifier

logger = logging.getLogger(__name__)

#: The AI analysis timeframe. Only closed candles on this interval are analyzed.
SCHEDULER_INTERVAL = "1h"

#: Epoch-ms length of one analysis candle (1h); used to attribute diagnostic
#: rows when detection cannot resolve a closed candle at all.
SCHEDULER_INTERVAL_MS = 3_600_000

#: Indicator columns stored in the Phase 16 candle-log snapshot (the analysis
#: candle's row of the deterministic indicator matrix).
_INDICATOR_SNAPSHOT_COLUMNS = (
    "close",
    "ema20",
    "ema50",
    "ema200",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr14",
    "adx14",
    "adx_plus_di",
    "adx_minus_di",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "volume_sma20",
    "volume_ratio",
)

#: How many closed 1H candles the analysis window fetches. Must be enough for
#: the full indicator set (EMA200 needs 200 candles); +20 gives headroom.
DEFAULT_WINDOW_CANDLES = MIN_REQUIRED_CANDLES + 20

#: Detection fetch size: only the newest CLOSED candle is inspected, so a small
#: window covers kline-boundary edge cases (like the 1m monitor).
DEFAULT_DETECTION_LIMIT = 5

#: Default wall-clock delay between ``run_forever`` iterations. ``tick()`` is
#: non-blocking; real cadence is driven by candle-close timestamps, not sleep.
DEFAULT_POLL_INTERVAL = 30.0

#: Env tunables for the Phase E circuit breaker. After this many consecutive
#: ERROR ticks the scheduler pauses LLM work for the cooldown instead of
#: retrying every poll (~120 calls/hour into a sick provider). Zero disables.
ENV_ERROR_BREAKER_N = "BTCUSDT_ERROR_BREAKER_N"
ENV_ERROR_BREAKER_COOLDOWN_S = "BTCUSDT_ERROR_BREAKER_COOLDOWN_S"
DEFAULT_ERROR_BREAKER_N = 5
DEFAULT_ERROR_BREAKER_COOLDOWN_S = 600.0

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


def _breaker_settings() -> tuple[int, float]:
    """Resolve the Phase E circuit-breaker knobs (per tick, no caching).

    Unset/empty/malformed values fall back to the defaults (fail-open towards
    the standard guarded behavior); explicit non-positive values disable the
    breaker entirely.
    """
    raw_count = (os.environ.get(ENV_ERROR_BREAKER_N) or "").strip()
    raw_cooldown = (os.environ.get(ENV_ERROR_BREAKER_COOLDOWN_S) or "").strip()
    try:
        count = int(raw_count) if raw_count else DEFAULT_ERROR_BREAKER_N
    except ValueError:
        count = DEFAULT_ERROR_BREAKER_N
    try:
        cooldown = (
            float(raw_cooldown) if raw_cooldown else DEFAULT_ERROR_BREAKER_COOLDOWN_S
        )
    except ValueError:
        cooldown = DEFAULT_ERROR_BREAKER_COOLDOWN_S
    if count <= 0 or cooldown <= 0:
        return (0, 0.0)
    return (count, cooldown)


class SchedulerOutcome(str, Enum):
    """Structured outcome vocabulary of a single scheduler ``tick()``."""

    NO_NEW_CANDLE = "NO_NEW_CANDLE"
    ALREADY_PROCESSED = "ALREADY_PROCESSED"
    BLOCKED_ACTIVE_SIGNAL = "BLOCKED_ACTIVE_SIGNAL"
    WAIT = "WAIT"
    CREATED = "CREATED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    ERROR = "ERROR"
    REJECTED = "REJECTED"
    RATE_LIMITED = "RATE_LIMITED"
    EVENT_BLACKOUT = "EVENT_BLACKOUT"


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

    @classmethod
    def rejected(cls, message: str, candle_ts: int | None = None) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.REJECTED,
            candle_timestamp=candle_ts,
            message=message,
            errors=(message,),
        )

    @classmethod
    def rate_limited(cls, message: str, candle_ts: int | None = None) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.RATE_LIMITED,
            candle_timestamp=candle_ts,
            message=message,
            errors=(message,),
        )

    @classmethod
    def event_blackout(cls, candle_ts: int | None, message: str) -> SchedulerResult:
        return cls(
            outcome=SchedulerOutcome.EVENT_BLACKOUT,
            candle_timestamp=candle_ts,
            message=message,
        )


class OneHourScheduler:
    """Decides when the AI may analyze a closed 1H candle, and drives the pipeline.

    ``market_data`` must expose Phase 1's ``fetch_closed_klines`` contract and
    is injected for deterministic tests. ``analyzer`` must accept ``(candles,
    indicators)`` positionally and return a Phase 5 ``SignalAnalysis``; the
    default builds one from the configured LLM. ``engine`` is the Phase 6
    validation/persistence boundary (never duplicated here). ``notifier`` is an
    optional Phase 11 ``TelegramNotifier``; when provided, a NEW PENDING_ENTRY
    signal is announced once, only after it has been persisted. When
    ``candle_log`` is provided (Phase 16), one diagnostic row is upserted per
    analysed 1H candle (WAIT/CREATED/REJECTED/RATE_LIMITED/locked/error
    outcomes), explaining why that candle did or did not yield a signal.
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
        candle_log: CandleLogRepository | None = None,
        event_calendar: EventCalendar | None = None,
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
        self.candle_log = candle_log
        #: Phase N macro-event calendar (None disables blackout + notices).
        #: Never performs I/O on its own; refreshed (throttled) per tick.
        self.event_calendar = event_calendar
        #: Phase N context note for the analysis about to run (None = clean).
        self._tick_event_note: str | None = None
        #: Phase P1 positioning note + funding rate for the analysis (None =
        #: feed unavailable; analysis proceeds without them).
        self._tick_positioning_note: str | None = None
        self._tick_funding_rate: float | None = None
        #: Phase P2 4H-bias note + regime for the analysis (None = unclear
        #: or unavailable; the veto only fires on a clear UP/DOWN regime).
        self._tick_htf_note: str | None = None
        self._tick_regime: str | None = None
        #: Consecutive ERROR-tick streak for the Phase E circuit breaker, plus
        #: the epoch-ms instant the current open state expires (None = closed).
        #: Mutated only under ``_AI_LOCK`` in ``tick()`` (single process).
        self._error_streak = 0
        self._breaker_open_until_ms: int | None = None

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
            event_note=self._tick_event_note,
            positioning=self._tick_positioning_note,
            funding_rate=self._tick_funding_rate,
            htf_note=self._tick_htf_note,
            regime=self._tick_regime,
        )

    def tick(self, *, now: datetime | None = None) -> SchedulerResult:
        """Run one non-blocking scheduler pass.

        ``now`` is a timezone-aware UTC datetime (injectable for determinism);
        it defaults to the current UTC time. Returns a :class:`SchedulerResult`
        and never raises for expected market-data/LLM/database failures.

        The Phase E circuit breaker pauses LLM work after
        ``BTCUSDT_ERROR_BREAKER_N`` consecutive ERROR ticks (skipped ticks
        report DATA_UNAVAILABLE without touching the provider); any other
        outcome resets the streak.
        """
        now = now if now is not None else datetime.now(timezone.utc)
        now_ms = as_epoch_ms(now)
        break_count, cooldown_s = _breaker_settings()
        self._tick_symbol = None
        with _AI_LOCK:
            try:
                if (
                    break_count > 0
                    and self._breaker_open_until_ms is not None
                ):
                    if now_ms < self._breaker_open_until_ms:
                        wait_s = (self._breaker_open_until_ms - now_ms) / 1000.0
                        return SchedulerResult.data_unavailable(
                            f"circuit breaker open after {self._error_streak} "
                            f"consecutive errors; next retry in {wait_s:.0f}s",
                            None,
                        )
                    self._breaker_open_until_ms = None
                    logger.info("[Scheduler] circuit breaker half-open: retrying")
                result = self._tick_locked(now_ms, as_iso(now))
                if break_count > 0:
                    if result.outcome is SchedulerOutcome.ERROR:
                        self._error_streak += 1
                        if self._error_streak >= break_count:
                            self._breaker_open_until_ms = now_ms + int(
                                cooldown_s * 1000
                            )
                            logger.warning(
                                "[Scheduler] circuit breaker OPEN after %d "
                                "consecutive errors; pausing LLM calls for %.0fs",
                                self._error_streak,
                                cooldown_s,
                            )
                    elif self._error_streak:
                        logger.info(
                            "[Scheduler] circuit breaker streak reset by %s",
                            result.outcome.value,
                        )
                        self._error_streak = 0
                        self._breaker_open_until_ms = None
                return result
            finally:
                self._tick_symbol = None

    def _tick_locked(self, now_ms: int, now_iso: str) -> SchedulerResult:
        """The serialized tick body; the caller holds ``_AI_LOCK``."""
        # Resolve the analyzed symbol once for this whole tick so detection,
        # the analysis window, the marker and the signal all agree.
        self._tick_symbol = self.symbol
        self._tick_event_note = None
        self._tick_positioning_note = None
        self._tick_funding_rate = None
        self._tick_htf_note = None
        self._tick_regime = None
        # Phase N event watch runs on EVERY poll (not just new candles) so
        # WARN/START/END notices and PENDING cancellation fire within about
        # a minute of the real instant. Cheap: calendar cache, no I/O storms.
        event_status = self._event_watch(now_ms)
        detection = self._detect(now_ms)
        if isinstance(detection, SchedulerResult):
            if detection.outcome in (
                SchedulerOutcome.DATA_UNAVAILABLE,
                SchedulerOutcome.ERROR,
            ):
                # Detection could not even resolve a closed candle, so there is
                # no analysis candle to attribute the row to; bucket it to the
                # hour window in which the failure happened. ALREADY_PROCESSED
                # and NO_NEW_CANDLE are normal control flow and stay unlogged.
                bucket_ms = int(now_ms) - (int(now_ms) % SCHEDULER_INTERVAL_MS)
                self._log_candle(
                    detection.outcome.value,
                    bucket_ms,
                    now_iso,
                    message=detection.message,
                )
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
            self._log_candle(
                SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL.value,
                detection_ts,
                now_iso,
                message=(
                    f"active signal {active.id} ({active.status}) exists; the AI is "
                    "locked until the position is closed"
                ),
            )
            return SchedulerResult.blocked(active, detection_ts)

        # Phase N blackout gate (by wall-clock now, not candle time): while a
        # scheduled release window is open, no new position may be created.
        # The candle is marked + logged so the window leaves a clean audit
        # trail and is never analyzed afterwards (like REJECTED/RATE_LIMITED).
        if event_status is not None and event_status.active is not None:
            blackout_event = event_status.active
            message = (
                f"event blackout: {blackout_event.title} "
                f"(AI resumes after {format_vn(blackout_event.blackout_end_ms)}); "
                "no new positions inside the release window"
            )
            logger.info(
                "[Scheduler] blackout gate for candle %s: %s",
                detection_ts,
                blackout_event.key,
            )
            marked = self._mark_processed(detection_ts, now_iso)
            if not marked:
                logger.error(
                    "[Scheduler] blackout for candle %s but failed to record it "
                    "as processed; the analysis will be retried",
                    detection_ts,
                )
                self._log_candle(
                    SchedulerOutcome.ERROR.value,
                    detection_ts,
                    now_iso,
                    message="event blackout but the failed marker write was not recorded",
                )
                return SchedulerResult.error(
                    "event blackout but the candle could not be recorded as "
                    "processed; the analysis will be retried",
                    detection_ts,
                )
            self._log_candle(
                SchedulerOutcome.EVENT_BLACKOUT.value,
                detection_ts,
                now_iso,
                message=message,
            )
            return SchedulerResult.event_blackout(detection_ts, message)

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
            self._log_candle(
                SchedulerOutcome.DATA_UNAVAILABLE.value,
                detection_ts,
                now_iso,
                message=f"{exc.__class__.__name__}: {exc}",
            )
            return SchedulerResult.data_unavailable(
                f"{exc.__class__.__name__}: {exc}", detection_ts
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error("[Scheduler] unexpected market-data failure: %s", exc)
            self._log_candle(
                SchedulerOutcome.ERROR.value,
                detection_ts,
                now_iso,
                message=f"unexpected market-data failure: {exc}",
            )
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
            self._log_candle(
                SchedulerOutcome.DATA_UNAVAILABLE.value,
                detection_ts,
                now_iso,
                message=(
                    f"only {len(window)} closed {self.timeframe} candles are available; "
                    f"at least {MIN_REQUIRED_CANDLES} are required for the full "
                    "indicator set"
                ),
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
            self._log_candle(
                SchedulerOutcome.DATA_UNAVAILABLE.value,
                analysis_ts,
                now_iso,
                message=f"indicator calculation failed: {exc.__class__.__name__}: {exc}",
            )
            return SchedulerResult.data_unavailable(
                f"indicator calculation failed: {exc.__class__.__name__}: {exc}",
                detection_ts,
            )

        logger.info("[Scheduler] analyzing closed %s candle %s", self.timeframe, analysis_ts)
        # Phase N: annotate the event candle / upcoming release for the AI.
        # Closed candles only, so this is look-ahead safe.
        if self.event_calendar is not None:
            try:
                self._tick_event_note = self.event_calendar.event_note(
                    window, now_ms
                )
            except Exception as exc:  # pragma: no cover - never break analysis
                logger.warning("[Scheduler] event note failed: %s", exc)
                self._tick_event_note = None
        # Phase P1: live positioning snapshot (3 tiny public GETs). Only
        # fetched when an analysis will actually run; any failure degrades
        # to no note and no funding guard (never blocks the analysis).
        client = getattr(self.market_data, "client", None)
        if client is not None:
            try:
                snapshot = fetch_positioning(client, self.symbol)
                self._tick_positioning_note = render_note(
                    snapshot.funding_rate,
                    snapshot.long_pct,
                    snapshot.short_pct,
                    snapshot.open_interest,
                )
                self._tick_funding_rate = snapshot.funding_rate
            except Exception as exc:  # pragma: no cover - never break analysis
                logger.warning("[Scheduler] positioning fetch failed: %s", exc)
        # Phase P2: 4H trend bias (one extra public fetch, same fail-soft
        # contract: unclear/unavailable never vetoes, analysis continues).
        fetch_klines = getattr(self.market_data, "fetch_closed_klines", None)
        if fetch_klines is not None:
            try:
                candles_4h = fetch_klines(
                    self.symbol, "4h", limit=HTF_CANDLE_LIMIT, now_ms=now_ms
                )
                bias = classify_htf(candles_4h)
                self._tick_htf_note = bias.note
                self._tick_regime = (
                    bias.regime if bias.regime != REGIME_NONE else None
                )
            except Exception as exc:  # pragma: no cover - never break analysis
                logger.warning("[Scheduler] 4H bias fetch failed: %s", exc)
        try:
            analysis = self.analyzer(window, indicators)
        except AnalysisContextError as exc:
            logger.error(
                "[Scheduler] context invalid for candle %s: %s", analysis_ts, exc
            )
            self._log_candle(
                SchedulerOutcome.DATA_UNAVAILABLE.value,
                analysis_ts,
                now_iso,
                message=f"analysis context invalid: {exc}",
                indicators=indicators,
            )
            return SchedulerResult.data_unavailable(
                f"analysis context invalid: {exc}", analysis_ts
            )
        except RateLimitedError as exc:
            # Provider rate limit (e.g. OpenRouter free-tier 429): retrying
            # on every 30s poll would deepen the limit, so the candle is
            # skipped and recorded as processed instead.
            marked = self._mark_processed(analysis_ts, now_iso)
            if not marked:
                logger.error(
                    "[Scheduler] provider rate limit for candle %s but failed "
                    "to record it as processed; the analysis will be retried",
                    analysis_ts,
                )
                self._log_candle(
                    SchedulerOutcome.ERROR.value,
                    analysis_ts,
                    now_iso,
                    message=f"provider rate limit but the failed marker write was not recorded: {exc}",
                    indicators=indicators,
                )
                return SchedulerResult.error(
                    "provider rate limit but the candle could not be recorded "
                    "as processed; the analysis will be retried",
                    analysis_ts,
                )
            logger.warning(
                "[Scheduler] provider rate limit for candle %s, skipping: %s",
                analysis_ts,
                exc,
            )
            self._log_candle(
                SchedulerOutcome.RATE_LIMITED.value,
                analysis_ts,
                now_iso,
                message=f"provider rate limit, candle skipped: {exc}",
                indicators=indicators,
            )
            return SchedulerResult.rate_limited(
                f"provider rate limit, candle skipped: {exc}", analysis_ts
            )
        except SignalAnalysisError as exc:
            logger.error("[Scheduler] AI analysis failed for candle %s: %s", analysis_ts, exc)
            self._log_candle(
                SchedulerOutcome.ERROR.value,
                analysis_ts,
                now_iso,
                message=f"AI analysis failed: {exc}",
                indicators=indicators,
            )
            return SchedulerResult.error(f"AI analysis failed: {exc}", analysis_ts)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.error(
                "[Scheduler] AI analysis failed unexpectedly for candle %s: %s",
                analysis_ts,
                exc,
            )
            self._log_candle(
                SchedulerOutcome.ERROR.value,
                analysis_ts,
                now_iso,
                message=f"AI analysis failed unexpectedly: {exc}",
                indicators=indicators,
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
            self._log_candle(
                SchedulerOutcome.ERROR.value,
                analysis_ts,
                now_iso,
                message=f"signal engine failed: {exc}",
                analysis=analysis,
                indicators=indicators,
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
                self._log_candle(
                    SchedulerOutcome.ERROR.value,
                    analysis_ts,
                    now_iso,
                    message="engine reported CREATED without a signal",
                    indicators=indicators,
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
            self._log_candle(
                SchedulerOutcome.CREATED.value,
                analysis_ts,
                now_iso,
                analysis=analysis,
                signal_id=engine_result.signal.id,
                indicators=indicators,
            )
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
                self._log_candle(
                    SchedulerOutcome.ERROR.value,
                    analysis_ts,
                    now_iso,
                    message="AI returned WAIT but the failed marker write was not recorded",
                    analysis=analysis,
                    indicators=indicators,
                )
                return SchedulerResult.error(
                    "AI returned WAIT but the candle could not be recorded as "
                    "processed; the analysis will be retried",
                    analysis_ts,
                )
            logger.info("[Scheduler] AI returned WAIT for candle %s", analysis_ts)
            self._log_candle(
                SchedulerOutcome.WAIT.value,
                analysis_ts,
                now_iso,
                analysis=analysis,
                indicators=indicators,
            )
            return SchedulerResult.wait(analysis_ts)

        if engine_result.outcome is SignalOutcome.BLOCKED_OPEN_SIGNAL:
            # A concurrent creator (another process) won the race; the safe
            # re-read reports the blocking signal instead of inventing an error.
            blocking = self.state.active_signal()
            if blocking is not None:
                self._log_candle(
                    SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL.value,
                    analysis_ts,
                    now_iso,
                    message=(
                        f"create blocked by another active signal {blocking.id} "
                        f"({blocking.status})"
                    ),
                    indicators=indicators,
                )
                return SchedulerResult.blocked(blocking, analysis_ts)
            logger.error(
                "[Scheduler] race: create blocked for candle %s but no active "
                "signal is visible: %s",
                analysis_ts,
                engine_result.message,
            )
            self._log_candle(
                SchedulerOutcome.ERROR.value,
                analysis_ts,
                now_iso,
                message=f"create blocked by another active signal (race): {engine_result.message}",
                indicators=indicators,
            )
            return SchedulerResult.error(
                f"create blocked by another active signal (race): {engine_result.message}",
                analysis_ts,
            )

        if engine_result.outcome is SignalOutcome.REJECTED:
            # Validation failed for this candle's analysis: re-analyzing the
            # same candle would burn another LLM call for the same verdict,
            # so record it as processed (Phase A token guard).
            marked = self._mark_processed(analysis_ts, now_iso)
            if not marked:
                logger.error(
                    "[Scheduler] analysis rejected for candle %s but failed to "
                    "record it as processed; the analysis will be retried",
                    analysis_ts,
                )
                self._log_candle(
                    SchedulerOutcome.ERROR.value,
                    analysis_ts,
                    now_iso,
                    message="analysis rejected but the failed marker write was not recorded",
                    analysis=analysis,
                    indicators=indicators,
                )
                return SchedulerResult.error(
                    "analysis rejected but the candle could not be recorded as "
                    "processed; the analysis will be retried",
                    analysis_ts,
                )
            logger.info(
                "[Scheduler] analysis rejected for candle %s: %s",
                analysis_ts,
                engine_result.message,
            )
            self._log_candle(
                SchedulerOutcome.REJECTED.value,
                analysis_ts,
                now_iso,
                message=engine_result.message or "signal was not created",
                analysis=analysis,
                indicators=indicators,
            )
            return SchedulerResult.rejected(
                engine_result.message or "signal was not created", analysis_ts
            )

        logger.error("[Scheduler] analysis rejected/failed for candle %s: %s", analysis_ts, engine_result.message)
        self._log_candle(
            SchedulerOutcome.ERROR.value,
            analysis_ts,
            now_iso,
            message=engine_result.message or "signal was not created",
            analysis=analysis,
            indicators=indicators,
        )
        return SchedulerResult.error(
            engine_result.message or "signal was not created", analysis_ts
        )

    def _log_candle(
        self,
        outcome: str,
        candle_ts: int,
        now_iso: str,
        *,
        analysis: SignalAnalysis | None = None,
        message: str = "",
        signal_id: int | None = None,
        indicators: Any = None,
    ) -> None:
        """Upsert the Phase 16 diagnostic row for one 1H candle.

        ``outcome`` mirrors the scheduler outcome; ``analysis`` supplies the
        decision/reasoning/prices/model/quota when the AI actually ran
        (CREATED/WAIT). Errors and locked ticks record only ``message`` with
        zero LLM calls. Never raises: a log failure must not block the
        trading pipeline (AGENTS.md 27).
        """
        if self.candle_log is None:
            return
        decision: str | None = None
        confidence: int | None = None
        entry = stop_loss = take_profit = None
        provider = model = temperature = reasoning = None
        close_price = None
        closed_at = None
        llm_calls = 0
        prompt_tokens = completion_tokens = total_tokens = None
        if analysis is not None:
            decision = analysis.decision
            confidence = None
            if analysis.confidence is not None:
                try:
                    confidence = int(round(float(analysis.confidence)))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    confidence = None
            entry = analysis.entry_price
            stop_loss = analysis.stop_loss
            take_profit = analysis.take_profit
            provider = analysis.provider
            model = analysis.model
            temperature = analysis.temperature
            reasoning = analysis.reasoning
            close_price = analysis.candle_close_price
            closed_at = analysis.closed_at
            llm_calls = analysis.llm_calls or 0
            prompt_tokens = analysis.prompt_tokens
            completion_tokens = analysis.completion_tokens
            total_tokens = analysis.total_tokens
        else:
            decision = "NONE"
        if closed_at is None:
            closed_at = self._candle_close_iso(candle_ts)
        try:
            self.candle_log.upsert(
                symbol=self.symbol,
                timeframe=self.timeframe,
                candle_timestamp_ms=int(candle_ts),
                closed_at=closed_at,
                recorded_at=now_iso,
                outcome=outcome,
                decision=decision,
                confidence=confidence,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                close_price=close_price,
                signal_id=signal_id,
                provider=provider,
                model=model,
                temperature=temperature,
                reasoning=reasoning,
                error_notes=message or None,
                indicators_json=self._indicator_snapshot_json(indicators),
                llm_calls=llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("[Scheduler] candle-log write failed: %s", exc)

    @staticmethod
    def _candle_close_iso(candle_ts: int) -> str:
        """ISO-8601 close time of the 1H candle given its epoch-ms open time."""
        opened = datetime.fromtimestamp(int(candle_ts) / 1000, tz=timezone.utc)
        closed = opened.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return closed.isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _indicator_snapshot_json(indicators: Any) -> str | None:
        """JSON snapshot of the analysis candle's indicator row, or None."""
        if indicators is None or getattr(indicators, "empty", True):
            return None
        try:
            row = indicators.iloc[-1]
            snapshot = {
                column: round(float(row[column]), 6)
                for column in _INDICATOR_SNAPSHOT_COLUMNS
                if column in indicators.columns and str(row[column]) != "nan"
            }
            return json.dumps(snapshot, sort_keys=True) if snapshot else None
        except Exception:  # pragma: no cover - defensive boundary
            return None

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

    def _runtime_state(self):
        """Lazily built key/value store for once-per-event notice flags."""
        if getattr(self, "_runtime_store", None) is None:
            self._runtime_store = RuntimeStateRepository(self.repository.database)
        return self._runtime_store

    def _event_flag(self, key: str, phase: str) -> bool:
        """True when a Telegram notice was already sent for this event+phase."""
        try:
            return (
                self._runtime_state().get(f"event.notified.{key}.{phase}") == "1"
            )
        except Exception as exc:
            logger.warning("[Scheduler] event flag read failed: %s", exc)
            return False

    def _set_event_flag(self, key: str, phase: str) -> None:
        try:
            self._runtime_state().set(f"event.notified.{key}.{phase}", "1")
        except Exception as exc:
            logger.warning("[Scheduler] event flag write failed: %s", exc)

    def _notify_event(self, text: str) -> None:
        """Send a Phase N Telegram notice (never raises)."""
        if self.notifier is None:
            return
        try:
            self.notifier.send_message(text)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("[Scheduler] event Telegram notice failed: %s", exc)

    @staticmethod
    def _event_label(key: str) -> str:
        """Human label for an event key (``fomc:2026-10-28`` -> ``FOMC 28/10``)."""
        try:
            series, day = key.split(":", 1)
            year, month, dom = day.split("-", 2)
            return f"{series.upper()} {int(dom):02d}/{int(month):02d}/{year}"
        except (ValueError, AttributeError):
            return key

    def _cancel_pending_for_event(self, event_key: str) -> int:
        """Cancel a PENDING_ENTRY signal at blackout start (0 or 1).

        Only a still-pending entry is ever cancelled here; an OPEN position
        keeps its TP/SL monitor untouched. Never raises.
        """
        try:
            active = self.state.active_signal()
        except Exception as exc:
            logger.warning("[Scheduler] blackout %s: active read failed: %s", event_key, exc)
            return 0
        if active is None or active.status != STATUS_PENDING_ENTRY:
            return 0
        try:
            self.state.cancel(active.id)
        except Exception as exc:
            logger.warning(
                "[Scheduler] blackout %s: failed to cancel PENDING signal %s: %s",
                event_key,
                active.id,
                exc,
            )
            return 0
        logger.info(
            "[Scheduler] blackout %s: cancelled PENDING_ENTRY signal %s",
            event_key,
            active.id,
        )
        return 1

    def _event_watch(self, now_ms: int) -> EventCheck | None:
        """Run the Phase N per-poll event watch (cheap, no market/LLM calls).

        Handles WARN/START/END Telegram transitions (each exactly once per
        event, persisted across restarts) and cancels a PENDING_ENTRY the
        minute a blackout starts. Returns the current calendar status for the
        blackout gate, or None when no calendar is configured. Never raises.
        """
        calendar = self.event_calendar
        if calendar is None:
            return None
        try:
            status = calendar.check(now_ms)
        except Exception as exc:  # pragma: no cover - calendar never raises
            logger.warning("[Scheduler] event watch failed: %s", exc)
            return None
        try:
            warn_ms = int(calendar.warn_hours * 3_600_000)
            if warn_ms > 0:
                for upcoming in status.upcoming:
                    delta_ms = upcoming.event_ms - now_ms
                    if (
                        0 < delta_ms <= warn_ms
                        and not self._event_flag(upcoming.key, "warn")
                    ):
                        self._notify_event(
                            f"⚠️ {upcoming.title} lúc {format_vn(upcoming.event_ms)} — "
                            f"từ {format_vn(upcoming.blackout_start_ms)} ngừng mở "
                            "lệnh mới, lệnh chờ sẽ tự hủy."
                        )
                        self._set_event_flag(upcoming.key, "warn")
                        logger.info(
                            "[Scheduler] event WARN sent for %s", upcoming.key
                        )
            active = status.active
            if active is not None and not self._event_flag(active.key, "start"):
                cancelled = self._cancel_pending_for_event(active.key)
                self._notify_event(
                    f"⏸ Blackout {active.title} bắt đầu — đã hủy {cancelled} "
                    f"lệnh PENDING, AI nghỉ đến {format_vn(active.blackout_end_ms)}."
                )
                self._set_event_flag(active.key, "start")
                logger.info("[Scheduler] event START sent for %s", active.key)
            # The previous-active key lives in the KV store (not RAM) so a
            # restart mid-blackout still emits END exactly once afterwards.
            try:
                last_key = self._runtime_state().get("event.last_active")
            except Exception as exc:
                logger.warning("[Scheduler] last-active read failed: %s", exc)
                last_key = None
            current_key = active.key if active is not None else None
            if (
                last_key
                and last_key != current_key
                and self._event_flag(last_key, "start")
                and not self._event_flag(last_key, "end")
            ):
                self._notify_event(
                    f"▶️ Hết blackout ({self._event_label(last_key)}) — AI "
                    "phân tích lại từ nến 1H tiếp theo."
                )
                self._set_event_flag(last_key, "end")
                logger.info("[Scheduler] event END sent for %s", last_key)
            try:
                stored = current_key if current_key is not None else ""
                if (last_key or "") != stored:
                    self._runtime_state().set("event.last_active", stored)
            except Exception as exc:
                logger.warning("[Scheduler] last-active write failed: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("[Scheduler] event watch transition failed: %s", exc)
        return status

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
