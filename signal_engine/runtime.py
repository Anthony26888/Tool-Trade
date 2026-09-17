"""Phase 14: the 24/7 production runtime + CLI renderers (BTCUSDT signal engine).

The runtime wires the independently tested components into one daemon:

- :class:`OneHourScheduler` decides WHEN the AI may analyze (once per closed 1H
  candle, only while no active signal exists). It keeps its own Phase 11
  notifier so a NEW ``PENDING_ENTRY`` signal is announced exactly once.
- :class:`SignalMonitor` (with ``notifier=None``) drives the active signal's
  lifecycle from 1m closed candles. The runtime observes ONLY its outcome and
  is the single place that (1) records the DEMO position/trade via
  :class:`DemoExecutor` — AFTER PnL is known — and (2) sends the Telegram
  opened/TP/SL/ambiguous notifications. This ordering guarantees TP/SL
  message include the net PnL and the new balance.
- :class:`RecoveryService` + :class:`DemoExecutor.reconcile` re-establish the
  state from SQLite on restart; reconciliation NEVER notifies, so a restart can
  never resend notifications or double-count a position/trade.
- :class:`RuntimeStateRepository` heartbeats scheduler/monitor activity into
  ``runtime_state`` purely for the CLI/health dashboard; trading logic never
  reads it.

Safety invariants (AGENTS.md sections 3, 12, 18, 19, 21)
--------------------------------------------------------
- While a signal is ``PENDING_ENTRY`` or ``OPEN`` the scheduler returns
  ``BLOCKED_ACTIVE_SIGNAL`` BEFORE any market-data fetch — no AI, no new
  signal, no second position.
- TP/SL monitoring stays on 1m closed candles and is fully independent from the
  1H AI analysis timeframe.
- The demo path writes only through the atomic, idempotent
  ``DemoRepository.record_trade``; the single-position/single-trade UNIQUE
  indexes make an accidental second open/close structurally impossible.
- No Binance order/cancel/modify endpoint is ever reached — the runtime only
  consumes public market data.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from binance.market_data import (
    DEFAULT_SYMBOL,
    BinanceMarketData,
    validate_interval,
    validate_symbol,
)
from database.database import (
    DEFAULT_DB_PATH,
    Database,
    RuntimeStateRepository,
    SignalRepository,
    iso_utc_now,
)
from database.models import STATUS_OPEN, STATUS_PENDING_ENTRY, Signal
from demo.executor import DemoExecutor
from demo.position import DemoAccountRecord, DemoTrade

from .analysis import analyze_signal
from .config import ConfigService
from .llm import llm_config_from_env
from .monitor import (
    AMBIGUITY_REASON_ENTRY,
    AMBIGUITY_REASON_EXIT,
    DEFAULT_CANDLE_LIMIT,
    MonitorOutcome,
    MonitorResult,
    SignalMonitor,
)
from .recovery import RecoveryOutcome, RecoveryResult, RecoveryService
from .scheduler import DEFAULT_POLL_INTERVAL, OneHourScheduler
from .telegram import (
    RefreshableTelegramNotifier,
    TelegramNotifier,
    format_amount,
    format_price,
    format_timestamp,
    telegram_notifier_from_env,
)

logger = logging.getLogger(__name__)

#: Phase 14 demo defaults (AGENTS.md section 13; configurable, never hard-coded
#: into trading logic). The pure ``demo/account.py`` defaults (1000/50/10/1)
#: are untouched.
DEFAULT_DEMO_INITIAL_BALANCE = Decimal("20")
DEFAULT_DEMO_MARGIN_PER_TRADE = Decimal("2")
DEFAULT_DEMO_LEVERAGE = 10
DEFAULT_DEMO_RISK_PERCENT = Decimal("1")
DEFAULT_DEMO_FEE_RATE = Decimal("0.0004")

#: Default wall-clock poll intervals for the two retryable subsystems.
DEFAULT_SCHEDULER_POLL = DEFAULT_POLL_INTERVAL
DEFAULT_MONITOR_POLL = 15.0

# Environment variables consumed here (documented in ``.env.example``).
ENV_SYMBOL = "BTCUSDT_SYMBOL"
ENV_TIMEFRAME = "BTCUSDT_TIMEFRAME"
ENV_DB_PATH = "BTCUSDT_DB_PATH"
ENV_RUNTIME_SCHEDULER_POLL = "BTCUSDT_RUNTIME_SCHEDULER_POLL"
ENV_RUNTIME_MONITOR_POLL = "BTCUSDT_RUNTIME_MONITOR_POLL"
ENV_DEMO_INITIAL_BALANCE = "BTCUSDT_DEMO_INITIAL_BALANCE"
ENV_DEMO_MARGIN_PER_TRADE = "BTCUSDT_DEMO_MARGIN_PER_TRADE"
ENV_DEMO_LEVERAGE = "BTCUSDT_DEMO_LEVERAGE"
ENV_DEMO_RISK_PERCENT = "BTCUSDT_DEMO_RISK_PERCENT"
ENV_DEMO_FEE_RATE = "BTCUSDT_DEMO_FEE_RATE"

#: RuntimeStateRepository heartbeat keys.
RUNTIME_KEY_STATE = "runtime.state"
RUNTIME_KEY_PID = "runtime.pid"
RUNTIME_KEY_STARTED_AT = "runtime.started_at"
RUNTIME_KEY_SCHEDULER_LAST_TICK = "scheduler.last_tick"
RUNTIME_KEY_MONITOR_LAST_POLL = "monitor.last_poll"
RUNTIME_KEY_LAST_ERROR = "runtime.last_error"
RUNTIME_KEY_LAST_ERROR_AT = "runtime.last_error_at"

#: ``state`` lifecycle values in runtime_state.
RUNTIME_STATE_RUNNING = "running"
RUNTIME_STATE_STOPPED = "stopped"

#: A subsystem is reported RUNNING while its heartbeat is no older than this
#: multiple of its poll interval (a crashed daemon stops heartbeating).
_HEALTH_STALE_MULTIPLIER = 3.0


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 millisecond string ending in ``Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# -- Configuration ------------------------------------------------------------


def _env_decimal(env: dict, name: str, default: Decimal) -> Decimal:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return Decimal(str(raw).strip())
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal number") from exc


def _env_int(env: dict, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(env: dict, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated production runtime configuration.

    Defaults match the Phase 14 spec (BTCUSDT, 1H AI analysis, 1m monitoring,
    demo account 20/2/10/1/0.04%). Everything is overridable through
    ``BTCUSDT_*`` environment variables; nothing financial is hard-coded inside
    trading logic.
    """

    symbol: str = DEFAULT_SYMBOL
    timeframe: str = "1h"
    db_path: str = DEFAULT_DB_PATH
    scheduler_poll: float = DEFAULT_SCHEDULER_POLL
    monitor_poll: float = DEFAULT_MONITOR_POLL
    demo: DemoConfigLike = field(default_factory=lambda: _default_demo_config())

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "symbol", validate_symbol(self.symbol))
            object.__setattr__(self, "timeframe", validate_interval(self.timeframe))
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(self.scheduler_poll, (int, float)) or self.scheduler_poll <= 0:
            raise ValueError("scheduler_poll must be a positive number of seconds")
        if not isinstance(self.monitor_poll, (int, float)) or self.monitor_poll <= 0:
            raise ValueError("monitor_poll must be a positive number of seconds")
        try:
            from demo.account import validate_config

            validate_config(self.demo)
        except Exception as exc:
            raise ValueError(f"invalid demo config: {exc}") from exc


DemoConfigLike = Any  # demo.account.DemoConfig (imported lazily to avoid import cycle)


def _default_demo_config():
    from demo.account import DemoConfig

    return DemoConfig(
        initial_balance=DEFAULT_DEMO_INITIAL_BALANCE,
        margin_per_trade=DEFAULT_DEMO_MARGIN_PER_TRADE,
        leverage=DEFAULT_DEMO_LEVERAGE,
        risk_percent=DEFAULT_DEMO_RISK_PERCENT,
        fee_rate=DEFAULT_DEMO_FEE_RATE,
    )


def demo_config_from_env(env: dict | None = None):
    """Build a :class:`demo.account.DemoConfig` from ``BTCUSDT_DEMO_*`` env vars."""
    from demo.account import DemoConfig

    env = dict(os.environ) if env is None else env
    return DemoConfig(
        initial_balance=_env_decimal(
            env, ENV_DEMO_INITIAL_BALANCE, DEFAULT_DEMO_INITIAL_BALANCE
        ),
        margin_per_trade=_env_decimal(
            env, ENV_DEMO_MARGIN_PER_TRADE, DEFAULT_DEMO_MARGIN_PER_TRADE
        ),
        leverage=_env_int(env, ENV_DEMO_LEVERAGE, DEFAULT_DEMO_LEVERAGE),
        risk_percent=_env_decimal(
            env, ENV_DEMO_RISK_PERCENT, DEFAULT_DEMO_RISK_PERCENT
        ),
        fee_rate=_env_decimal(env, ENV_DEMO_FEE_RATE, DEFAULT_DEMO_FEE_RATE),
    )


def runtime_config_from_env(env: dict | None = None) -> RuntimeConfig:
    """Build a validated :class:`RuntimeConfig` from ``BTCUSDT_*`` env vars."""
    env = dict(os.environ) if env is None else env
    return RuntimeConfig(
        symbol=str(env.get(ENV_SYMBOL, "") or DEFAULT_SYMBOL),
        timeframe=str(env.get(ENV_TIMEFRAME, "") or "1h"),
        db_path=str(env.get(ENV_DB_PATH, "") or DEFAULT_DB_PATH),
        scheduler_poll=_env_float(
            env, ENV_RUNTIME_SCHEDULER_POLL, DEFAULT_SCHEDULER_POLL
        ),
        monitor_poll=_env_float(env, ENV_RUNTIME_MONITOR_POLL, DEFAULT_MONITOR_POLL),
        demo=demo_config_from_env(env),
    )


# -- Runtime ------------------------------------------------------------------


class Runtime:
    """The production daemon: recover, schedule 1H analysis, monitor 1m TP/SL.

    Components are injectable so tests can substitute fake market data, fakes
    for the LLM analyzer, tracing notifiers, and a temporary database. The
    default construction is fully env-driven.
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        database: Database | None = None,
        repository: SignalRepository | None = None,
        scheduler: OneHourScheduler | None = None,
        monitor: SignalMonitor | None = None,
        executor: DemoExecutor | None = None,
        notifier: TelegramNotifier | None = None,
        state_store: RuntimeStateRepository | None = None,
        recovery: RecoveryService | None = None,
        market_data: BinanceMarketData | None = None,
        config_service: ConfigService | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.config = config if config is not None else runtime_config_from_env()
        self.database = database if database is not None else Database(self.config.db_path)
        self.database.initialize()
        self.config_service = (
            config_service
            if config_service is not None
            else ConfigService(self.database, env=os.environ)
        )
        # Stored Demo settings (database precedence) win over env for the
        # account created at startup; the ledger row is never touched here.
        stored_demo = self.config_service.demo_config_object()
        if stored_demo is not None:
            self.config = replace(self.config, demo=stored_demo)
        repo = repository if repository is not None else SignalRepository(self.database)
        market = market_data if market_data is not None else BinanceMarketData()
        self.notifier = (
            notifier
            if notifier is not None
            else RefreshableTelegramNotifier(self.config_service.resolve_telegram_notifier)
            if self.config_service is not None
            else telegram_notifier_from_env()
        )

        if scheduler is None:
            scheduler = OneHourScheduler(
                repo,
                market_data=market,
                symbol=self._resolving_symbol,
                timeframe=self.config.timeframe,
                config=self.llm_env_config(),
                analyzer=self._resolving_analyzer(),
                notifier=self.notifier,
            )
        if monitor is None:
            monitor = SignalMonitor(
                repo,
                market_data=market,
                candle_limit=DEFAULT_CANDLE_LIMIT,
                notifier=None,
            )
        self.repository = repo
        self.scheduler = scheduler
        self.monitor = monitor
        self.executor = (
            executor if executor is not None else DemoExecutor(self.database, config=self.config.demo)
        )
        self.state_store = (
            state_store if state_store is not None else RuntimeStateRepository(self.database)
        )
        self.recovery = recovery if recovery is not None else RecoveryService(repo)

        self._stop_event = threading.Event()
        self._sleep = sleep_fn
        self._started = False
        #: Signals whose OPEN notification was already emitted in this process
        #: (guards the race-loser re-read path from double-announcing).
        self._notified_opened: set[int] = set()

    # -- Startup recovery -------------------------------------------------------

    def llm_env_config(self):
        """Legacy env-resolved LLM config (attribute default for the scheduler)."""
        return llm_config_from_env()

    def _resolving_symbol(self) -> str:
        """The analysed symbol, resolved fresh per scheduler tick: DB > env > default.

        A Settings-page symbol change therefore takes effect for the next
        eligible analysis without a daemon restart (mirrors the LLM provider
        resolution). Falls back to the env-driven launch symbol on any failure.
        """
        try:
            return self.config_service.resolve_symbol()
        except Exception:  # pragma: no cover - best-effort fallback
            return self.config.symbol

    def _resolving_analyzer(self):
        """An analyzer that resolves the LLM config per analysis (DB > env).

        Phase 15: the AI provider settings are resolved fresh on every eligible
        closed-candle analysis, so a configuration change saved through the
        Settings page takes effect for the NEXT analysis — never mid-trade —
        without a daemon restart. Falls back to the legacy env-driven default
        when no database AI-provider setting exists.
        """

        def analyzer(candles, indicators):
            config = self.config_service.resolve_llm_config()
            mode = self.config_service.resolve_analysis_mode()
            return analyze_signal(
                config,
                candles,
                indicators,
                symbol=self._resolving_symbol(),
                timeframe=self.config.timeframe,
                mode=mode,
            )

        return analyzer

    def start(self) -> RecoveryResult:
        """Recover the runtime state from SQLite exactly once per process.

        Runs the read-only :class:`RecoveryService`, then reconciles the demo
        ledger against the signal ledger (repairing crash windows). Recovery
        NEVER notifies and NEVER calls the AI; a recovered ``PENDING_ENTRY`` or
        ``OPEN`` signal simply resumes monitoring.
        """
        # Phase 15: a config staged while a signal was active becomes live the
        # moment the system is idle (never applied to a running position).
        try:
            self.config_service.apply_pending_if_idle()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("[Runtime] pending config promotion failed: %s", exc)
        result = self.recovery.recover()
        if not self._started:
            self.executor.reconcile(active_signal=result.signal)
            self._started = True
            if result.outcome is RecoveryOutcome.RECOVERED_OPEN:
                logger.info(
                    "[Runtime] recovered signal %s (%s); AI is locked until it closes.",
                    result.signal.id,
                    result.signal.status,
                )
            self.state_store.set(RUNTIME_KEY_STATE, RUNTIME_STATE_RUNNING)
            self.state_store.set(RUNTIME_KEY_PID, str(os.getpid()))
            self.state_store.set(RUNTIME_KEY_STARTED_AT, iso_utc_now())
        return result

    def stop(self) -> None:
        """Graceful shutdown: stop the daemon loop and mark state stopped."""
        self._stop_event.set()
        try:
            self.state_store.set(RUNTIME_KEY_STATE, RUNTIME_STATE_STOPPED)
        except Exception as exc:  # pragma: no cover - best-effort shutdown bookkeeping
            logger.warning("[Runtime] could not record stopped state: %s", exc)

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # -- One pass ----------------------------------------------------------------

    def tick_scheduler(self) -> str:
        """Run one scheduler tick and heartbeat it; returns the outcome string."""
        # Phase 15: apply a pending AI-provider change as soon as the system is
        # idle (before the scheduler decides whether the AI may analyze).
        try:
            self.config_service.apply_pending_if_idle()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("[Runtime] pending config promotion failed: %s", exc)
        now = _now_iso()
        result = self.scheduler.tick()
        self.state_store.set(RUNTIME_KEY_SCHEDULER_LAST_TICK, now)
        if result.outcome == "ERROR":
            self._record_error(result.message)
        else:
            self._clear_error()
        return result.outcome.value

    def poll_monitor(self) -> str:
        """Run one monitor poll, heartbeat it, handle demo/notifications.

        Returns the monitor outcome string.
        """
        now = _now_iso()
        result = self.monitor.poll()
        self.state_store.set(RUNTIME_KEY_MONITOR_LAST_POLL, now)
        if result.outcome == "ERROR":
            self._record_error(result.message)
        self._handle_monitor_result(result)
        return result.outcome.value

    def poll(self) -> dict[str, str]:
        """Run one scheduler tick + one monitor poll.

        Returns a compact summary dict for logging/tests. Expected failures
        (market data, LLM, database) are recorded as heartbeats and errors; a
        single failing poll never stops the daemon.
        """
        return {
            "scheduler": self.tick_scheduler(),
            "monitor": self.poll_monitor(),
        }

    def run_once(self) -> dict[str, str]:
        """Recover, then run exactly one analysis pass + monitor poll, then stop.

        Used by ``python -m signal_engine --once`` (CI, cron, one-shot jobs).
        """
        summary: dict[str, str] = {}
        recovery = self.start()
        summary["recovery"] = recovery.outcome.value
        if recovery.outcome == "RECOVERY_ERROR":
            self._record_error(recovery.message)
            summary["recovery_error"] = recovery.message

        one_pass = self.poll()
        summary.update(one_pass)

        # Final reconcile repairs any crash window left between the poll's
        # transition and its demo write; it never notifies.
        reconcile = self.executor.reconcile(active_signal=self.repository.get_active_signal())
        summary["reconcile_positions"] = str(reconcile["positions_recovered"])
        summary["reconcile_trades"] = str(reconcile["trades_recovered"])

        self.state_store.set(RUNTIME_KEY_STATE, RUNTIME_STATE_STOPPED)
        return summary

    def run_forever(self) -> None:
        """Blocking daemon loop until :meth:`stop` (SIGINT/SIGTERM handler).

        The scheduler and monitor run on their own poll intervals; the loop
        sleeps in short steps so neither subsystem is starved.
        """
        self.start()
        logger.info(
            "[Runtime] daemon running for %s %s (scheduler every %ss, monitor every %ss)",
            self._resolving_symbol(),
            self.config.timeframe,
            self.config.scheduler_poll,
            self.config.monitor_poll,
        )
        next_scheduler = time.monotonic() + self.config.scheduler_poll
        next_monitor = time.monotonic() + self.config.monitor_poll
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now >= next_scheduler:
                    self.tick_scheduler()
                    next_scheduler = time.monotonic() + self.config.scheduler_poll
                if now >= next_monitor:
                    self.poll_monitor()
                    next_monitor = time.monotonic() + self.config.monitor_poll
                step = min(
                    max(0.01, next_scheduler - now),
                    max(0.01, next_monitor - now),
                )
                self._sleep(step)
        finally:
            self.stop()
            logger.info("[Runtime] daemon stopped")

    # -- Monitor outcome handling -------------------------------------------------

    def _handle_monitor_result(self, result: MonitorResult) -> None:
        """Drive demo accounting + user notifications from one monitor outcome.

        This is the ONLY place that records demo positions/trades and sends
        opened/TP/SL/ambiguous notifications for live transitions (recovery and
        reconcile never notify). Notifications run strictly after the demo
        write, so a TP/SL message includes the net PnL and the new balance.
        """
        signal = result.signal
        if result.outcome is MonitorOutcome.ENTRY_HIT:
            position = self.executor.open_position(signal)
            if position is not None and signal is not None:
                logger.info(
                    "[Runtime] demo position %s opened for signal %s", position.id, signal.id
                )
            self._clear_error()
            if signal is not None and signal.id not in self._notified_opened:
                self._notified_opened.add(signal.id)
                self._notify("notify_signal_opened", signal)
            return
        if result.outcome in (MonitorOutcome.TP_HIT, MonitorOutcome.SL_HIT):
            trade = self.executor.close_position(signal)
            if trade is not None:
                account = self.executor.account()
                method = "notify_signal_tp" if result.outcome is MonitorOutcome.TP_HIT else "notify_signal_sl"
                self._notify(
                    method,
                    signal,
                    pnl=trade.net_pnl,
                    balance=account.balance if account is not None else None,
                )
            return
        if result.outcome is MonitorOutcome.AMBIGUOUS:
            if signal is not None and signal.status == STATUS_PENDING_ENTRY:
                reason = AMBIGUITY_REASON_ENTRY
            else:
                reason = AMBIGUITY_REASON_EXIT
            self._notify("notify_ambiguous", signal, reason=reason)
            return

    def _notify(self, method_name: str, signal: Signal | None, **kwargs: Any) -> None:
        if signal is None:
            return
        method = getattr(self.notifier, method_name, None)
        if method is None:
            return
        try:
            result = method(signal, **kwargs)
            if result is not None and not result.ok:
                logger.warning(
                    "Telegram %s notification not sent for signal %s: %s",
                    method_name,
                    signal.id,
                    getattr(result, "error", "unknown"),
                )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning(
                "Telegram %s notification failed for signal %s: %s",
                method_name,
                signal.id,
                exc,
            )

    def _record_error(self, message: str) -> None:
        try:
            self.state_store.set(RUNTIME_KEY_LAST_ERROR, str(message))
            self.state_store.set(
                RUNTIME_KEY_LAST_ERROR_AT,
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("[Runtime] could not record error state: %s", exc)

    def _clear_error(self) -> None:
        """Drop a stale last-error entry after a successful pass/position open."""
        try:
            self.state_store.set(RUNTIME_KEY_LAST_ERROR, "")
            self.state_store.set(RUNTIME_KEY_LAST_ERROR_AT, "")
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("[Runtime] could not clear error state: %s", exc)

    # -- Health -------------------------------------------------------------------

    def health(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Live subsystem health from the ``runtime_state`` heartbeats.

        A subsystem is RUNNING while its heartbeat is recent relative to its
        poll interval; a crashed/cold daemon quickly reads STOPPED.
        """
        snapshot = self.state_store.snapshot()
        now = now if now is not None else datetime.now(timezone.utc)
        return health_from_snapshot(
            snapshot,
            now=now,
            scheduler_poll=self.config.scheduler_poll,
            monitor_poll=self.config.monitor_poll,
        )


def health_from_snapshot(
    snapshot: dict[str, str],
    *,
    now: datetime,
    scheduler_poll: float,
    monitor_poll: float,
) -> dict[str, Any]:
    """Compute the health dict from a runtime_state snapshot (pure, testable)."""

    def _subsystem(heartbeat_key: str, poll_seconds: float) -> tuple[str, str | None, str]:
        value = snapshot.get(heartbeat_key)
        running = "RUNNING" if _recent(now, value, poll_seconds) else "STOPPED"
        return running, value, snapshot.get(RUNTIME_KEY_LAST_ERROR)

    scheduler_running, scheduler_last, _ = _subsystem(
        RUNTIME_KEY_SCHEDULER_LAST_TICK, scheduler_poll
    )
    monitor_running, monitor_last, _ = _subsystem(
        RUNTIME_KEY_MONITOR_LAST_POLL, monitor_poll
    )
    return {
        "state": snapshot.get(RUNTIME_KEY_STATE, "unknown"),
        "pid": snapshot.get(RUNTIME_KEY_PID),
        "started_at": snapshot.get(RUNTIME_KEY_STARTED_AT),
        "scheduler": scheduler_running,
        "scheduler_last_tick": scheduler_last,
        "monitor": monitor_running,
        "monitor_last_poll": monitor_last,
        "last_error": snapshot.get(RUNTIME_KEY_LAST_ERROR),
        "last_error_at": snapshot.get(RUNTIME_KEY_LAST_ERROR_AT),
    }


def _recent(now: datetime, iso_timestamp: str | None, stale_after: float) -> bool:
    if not iso_timestamp:
        return False
    try:
        when = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if now is None or when.tzinfo is None or now.tzinfo is None:
        return False
    return (now - when).total_seconds() <= stale_after * _HEALTH_STALE_MULTIPLIER


# -- Renderers (pure, used by the CLI and covered by unit tests) ----------------


def render_status(health: dict[str, Any], active: Signal | None, *, db_path: str) -> str:
    """Daemon dashboard: health + the active signal summarized in one box."""
    lines = [
        "BTCUSDT Signal Engine — status",
        f"Database: {db_path}",
        f"Runtime state: {health['state']!r} (pid {health.get('pid') or '-'}, "
        f"started {health.get('started_at') or '-'})",
        f"Scheduler: {health['scheduler']} (last tick {health.get('scheduler_last_tick') or '-'})",
        f"Monitor:   {health['monitor']} (last poll {health.get('monitor_last_poll') or '-'})",
    ]
    if health.get("last_error"):
        lines.append(f"Last error: {health['last_error']}")
    lines.append("")
    if active is None:
        lines.append("Active signal: none")
    else:
        lines.append(f"Active signal: #{active.id} {active.direction} {active.symbol} — {render_signal_summary(active)}")
    return "\n".join(lines)


def render_signal_summary(signal: Signal) -> str:
    text = f"status {signal.status}"
    text += f", entry {format_price(signal.entry)}"
    text += f", SL {format_price(signal.stop_loss)}"
    text += f", TP {format_price(signal.take_profit)}"
    if signal.opened_at:
        text += f", opened {format_timestamp(signal.opened_at)}"
    if signal.closed_at:
        text += f", closed {format_timestamp(signal.closed_at)}"
    return text


def render_active(signal: Signal | None) -> str:
    if signal is None:
        return "No active signal (AI is unlocked for the next eligible 1H candle close)."
    heading = f"Active signal #{signal.id}: {signal.direction} {signal.symbol}"
    lines = [
        heading,
        f"Status: {signal.status}",
        f"Entry: {format_price(signal.entry)}",
        f"Stop loss: {format_price(signal.stop_loss)}",
        f"Take profit: {format_price(signal.take_profit)}",
        f"Created: {format_timestamp(signal.created_at)}",
    ]
    if signal.opened_at:
        lines.append(f"Opened: {format_timestamp(signal.opened_at)}")
    if signal.close_price is not None:
        lines.append(f"Close price: {format_price(signal.close_price)}")
    if signal.confidence is not None:
        lines.append(f"Confidence: {signal.confidence}%")
    if signal.model_name:
        lines.append(f"Model: {signal.model_name}")
    return "\n".join(lines)


def render_signals(signals: list[Signal]) -> str:
    if not signals:
        return "No signals recorded."
    lines = [f"Recent signals ({len(signals)}):", ""]
    for signal in signals:
        mark = "●" if signal.status in (STATUS_PENDING_ENTRY, STATUS_OPEN) else "○"
        lines.append(
            f"{mark} #{signal.id} {signal.direction:5s} {signal.status:12s} "
            f"{format_timestamp(signal.created_at)} "
            f"entry {format_price(signal.entry)}"
        )
    return "\n".join(lines)


def render_demo(account: DemoAccountRecord | None, trades: list[DemoTrade], stats) -> str:
    if account is None:
        return "No demo account yet (run the daemon once to create it)."
    lines = [
        "Demo account",
        f"Balance: {format_price(account.balance)} USDT",
        f"Equity:  {format_price(account.equity)} USDT",
        f"Peak equity: {format_price(account.peak_equity)} USDT",
        f"Initial balance: {format_price(account.initial_balance)} USDT",
        f"Margin per trade: {format_price(account.margin_per_trade)} USDT "
        f"(leverage {account.leverage}x, fee rate {100 * account.fee_rate:.4f}%)",
        "",
    ]
    if stats is not None:
        win_rate = f"{stats.win_rate:.2%}".replace(".00%", "%") if stats.win_rate is not None else "-"
        profit_factor = f"{stats.profit_factor:.2f}" if stats.profit_factor is not None else "-"
        lines += [
            "Statistics",
            f"Trades: {stats.total_trades} ({stats.wins} win / {stats.losses} loss)",
            f"Win rate: {win_rate}",
            f"Net PnL: {format_amount(stats.net_pnl)} USDT",
            f"Total fees: {format_amount(stats.total_fees)} USDT",
            f"Profit factor: {profit_factor}",
            f"Max drawdown: {format_amount(stats.max_drawdown)} USDT",
            "",
        ]
    if trades:
        lines.append("Recent trades:")
        for trade in trades:
            lines.append(
                f"  #{trade.id} {trade.side:5s} {format_amount(trade.net_pnl):>10s} USDT "
                f"{trade.result:4s} @ {format_timestamp(trade.closed_at)}"
            )
    return "\n".join(lines)


def render_health(health: dict[str, Any]) -> str:
    lines = [
        "BTCUSDT Signal Engine — health",
        f"Runtime state: {health['state']}",
        f"Scheduler: {health['scheduler']}",
        f"Monitor:   {health['monitor']}",
    ]
    if health.get("last_error"):
        lines.append(f"Last error: {health['last_error']}")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


# -- Module surface ---------------------------------------------------------------


__all__ = [
    "DEFAULT_DEMO_FEE_RATE",
    "DEFAULT_DEMO_INITIAL_BALANCE",
    "DEFAULT_DEMO_LEVERAGE",
    "DEFAULT_DEMO_MARGIN_PER_TRADE",
    "DEFAULT_DEMO_RISK_PERCENT",
    "DEFAULT_MONITOR_POLL",
    "DEFAULT_SCHEDULER_POLL",
    "ENV_DB_PATH",
    "ENV_DEMO_FEE_RATE",
    "ENV_DEMO_INITIAL_BALANCE",
    "ENV_DEMO_LEVERAGE",
    "ENV_DEMO_MARGIN_PER_TRADE",
    "ENV_DEMO_RISK_PERCENT",
    "ENV_RUNTIME_MONITOR_POLL",
    "ENV_RUNTIME_SCHEDULER_POLL",
    "ENV_SYMBOL",
    "ENV_TIMEFRAME",
    "RUNTIME_KEY_LAST_ERROR",
    "RUNTIME_KEY_LAST_ERROR_AT",
    "RUNTIME_KEY_MONITOR_LAST_POLL",
    "RUNTIME_KEY_PID",
    "RUNTIME_KEY_SCHEDULER_LAST_TICK",
    "RUNTIME_KEY_STARTED_AT",
    "RUNTIME_KEY_STATE",
    "RUNTIME_STATE_RUNNING",
    "RUNTIME_STATE_STOPPED",
    "Runtime",
    "RuntimeConfig",
    "demo_config_from_env",
    "health_from_snapshot",
    "render_active",
    "render_demo",
    "render_health",
    "render_signals",
    "render_signal_summary",
    "render_status",
    "runtime_config_from_env",
]
