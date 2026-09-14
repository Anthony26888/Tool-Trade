"""Deterministic, isolated backtest engine for the BTCUSDT strategy (Phase 12).

The engine replays the exact production lifecycle — AI decision, PENDING_ENTRY,
OPEN, TP/SL close — in pure memory against historical data. It never touches
SQLite, never calls Binance, never invokes the LLM, and never places orders.

No look-ahead guarantees
------------------------
- The decision for candle ``i`` uses only ``candles[:i+1]`` (closed candles up to
  and including ``i``); indicators are recomputed from that exact window.
- The decision time equals the candle open time plus the decision-timeframe
  length (``open_time + interval_ms``) — the moment the candle is closed. The
  candle's ``close_time`` field is never used as the decision instant.
- Entry/TP/SL confirmation uses closed 1m candles whose open time is ``>=`` the
  decision time; 1m candles inside the signal candle (before it closed) are
  never eligible.
- Confirmation shortcuts reuse the Phase 9 monitor vocabulary verbatim through
  ``evaluate_entry`` / ``evaluate_candle``, so a candle touching both TP and SL
  is AMBIGUOUS, nothing is closed, and monitoring continues (intrabar order is
  never guessed from OHLC).

Money model (Phase 8 reuse, not a fork)
---------------------------------------
Position notional is margin x leverage; quantity is notional / execution entry;
gross PnL is ``(exit - entry) * qty`` (LONG) or ``(entry - exit) * qty`` (SHORT);
entry/exit fees are charged separately; net PnL = gross - fees - funding. Slippage
moves the execution price away from the configured level but never affects touch
detection. Balance updates by net PnL only; equity equals balance (no
mark-to-market, matching Phase 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from binance.indicators import compute_indicator_matrix
from database.models import SignalValidationError
from demo.account import (
    DIRECTION_LONG,
    exit_fee as _exit_fee,
    gross_pnl,
    position_quantity,
    position_size,
)
from signal_engine.analysis import SignalAnalysis
from signal_engine.monitor import (
    MonitorOutcome,
    evaluate_candle,
    evaluate_entry,
)
from signal_engine.validator import validate_analysis

from .config import BacktestConfig, BacktestExecutionError, interval_to_ms
from .data import HistoricalData
from .provider import BacktestDecisionProvider

CLOSE_REASON_TP = "TP"
CLOSE_REASON_SL = "SL"


@dataclass(frozen=True)
class DecisionRecord:
    """What the provider saw and decided for one eligible candle."""

    candle_ts: int
    decision: str
    confidence: float
    candle_close_price: float
    provider: str
    model: str


@dataclass(frozen=True)
class BacktestTrade:
    """One executed and/or closed position, all prices as ``Decimal``."""

    index: int
    signal_candle_ts: int
    direction: str
    entry_level: Decimal
    stop_level: Decimal
    take_profit_level: Decimal
    entry_time_ms: int
    exit_time_ms: int | None
    entry_price: Decimal | None
    exit_price: Decimal | None
    quantity: Decimal | None
    position_size: Decimal
    entry_fee: Decimal | None
    exit_fee: Decimal | None
    funding_cost: Decimal | None
    gross_pnl: Decimal | None
    net_pnl: Decimal | None
    risk_amount: Decimal
    r_multiple: Decimal | None
    outcome: str | None
    was_ambiguous: bool
    holding_time_ms: int | None


@dataclass(frozen=True)
class PendingEvent:
    """A valid LONG/SHORT signal created, and its final resolution."""

    signal_candle_ts: int
    decision_time_ms: int
    direction: str
    entry_level: Decimal
    stop_level: Decimal
    take_profit_level: Decimal
    entered: bool
    entered_at_ms: int | None
    closed: bool
    closed_at_ms: int | None
    outcome: str | None
    trade_index: int | None


@dataclass(frozen=True)
class AmbiguityEvent:
    """A closed 1m candle whose intrabar order could not be resolved."""

    signal_candle_ts: int
    minute_ts: int
    kind: str  # "ENTRY" or "EXIT"
    direction: str


@dataclass(frozen=True)
class BlockedEvent:
    """A candle skipped because an active signal already existed."""

    candle_ts: int
    decision_time_ms: int
    reason: str
    active_direction: str | None
    active_signal_candle_ts: int | None


@dataclass(frozen=True)
class RejectedEvent:
    """A LONG/SHORT decision that failed signal validation."""

    candle_ts: int
    decision: str
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    """Complete, deterministic outcome of one backtest run."""

    config: BacktestConfig
    symbol: str
    timeframe: str
    execution_interval: str
    eligible: int
    analyzed: int
    wait_count: int
    long_decisions: int
    short_decisions: int
    rejected_count: int
    blocked_count: int
    long_created: int
    short_created: int
    pending_created: int
    entries_hit: int
    trades_completed: int
    open_at_end: int
    pending_at_end: int
    ambiguous_total: int
    entry_ambiguity_count: int
    exit_ambiguity_count: int
    trades: tuple[BacktestTrade, ...]
    pending_events: tuple[PendingEvent, ...]
    ambiguous_events: tuple[AmbiguityEvent, ...]
    blocked_events: tuple[BlockedEvent, ...]
    rejected_events: tuple[RejectedEvent, ...]
    decisions: tuple[DecisionRecord, ...]
    equity_curve: tuple[Decimal, ...]
    final_balance: Decimal


@dataclass
class _ActiveSignal:
    """Internal, mutable lifecycle object for the single active signal."""

    signal_candle_ts: int
    decision_time_ms: int
    direction: str
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    ndx: int
    entered: bool = False
    entry_time_ms: int | None = None
    exit_time_ms: int | None = None
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    quantity: Decimal | None = None
    position_size: Decimal = Decimal("0")
    entry_fee: Decimal | None = None
    exit_fee: Decimal | None = None
    funding_cost: Decimal | None = None
    gross_pnl: Decimal | None = None
    net_pnl: Decimal | None = None
    risk_amount: Decimal = Decimal("0")
    r_multiple: Decimal | None = None
    outcome: str | None = None
    was_ambiguous: bool = False


@dataclass(frozen=True)
class _Levels:
    """Minimal duck-typed fixture for the Phase 9 monitor evaluation functions."""

    direction: str
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal


def _slippage_factor(bps: Decimal, *, direction: str, entering: bool) -> Decimal:
    if bps == 0:
        return Decimal("1")
    slip = bps / Decimal(10000)
    if entering:
        return Decimal("1") + slip if direction == DIRECTION_LONG else Decimal("1") - slip
    return Decimal("1") - slip if direction == DIRECTION_LONG else Decimal("1") + slip


class BacktestEngine:
    """Runs one deterministic backtest over a :class:`HistoricalData`."""

    def __init__(
        self,
        config: BacktestConfig,
        data: HistoricalData,
        provider: BacktestDecisionProvider,
    ) -> None:
        self.config = config
        self.data = data
        if provider is None:
            raise BacktestExecutionError("a BacktestDecisionProvider is required")
        self.provider = provider
        if data.symbol != config.symbol:
            raise BacktestExecutionError(
                f"data symbol {data.symbol} does not match config symbol {config.symbol}"
            )
        if data.timeframe != config.timeframe:
            raise BacktestExecutionError(
                f"data timeframe {data.timeframe} does not match config timeframe "
                f"{config.timeframe}"
            )
        if data.execution_interval != config.execution_interval:
            raise BacktestExecutionError(
                f"data execution interval {data.execution_interval} does not match "
                f"config execution interval {config.execution_interval}"
            )
        if len(data.hour_candles) < config.min_candles:
            raise BacktestExecutionError(
                f"at least {config.min_candles} closed decision candles are required "
                f"for a backtest, got {len(data.hour_candles)}"
            )

    def run(self) -> BacktestResult:
        hours = list(self.data.hour_candles)
        minutes = list(self.data.minute_candles)
        config = self.config
        first_eligible = config.min_candles - 1
        total_eligible = len(hours) - first_eligible
        decision_period_ms = interval_to_ms(config.timeframe)

        balance = Decimal(config.initial_balance)
        equity_curve: list[Decimal] = [balance]
        minute_idx = 0
        minute_count = len(minutes)

        active: _ActiveSignal | None = None
        trades: list[BacktestTrade] = []
        pending_events: list[PendingEvent] = []
        ambiguous_events: list[AmbiguityEvent] = []
        blocked_events: list[BlockedEvent] = []
        rejected_events: list[RejectedEvent] = []
        decisions: list[DecisionRecord] = []

        long_decisions = 0
        short_decisions = 0
        wait_count = 0
        analyzed = 0
        blocked_count = 0
        rejected_count = 0
        created_counter = 0
        entries_hit = 0
        trades_completed = 0
        entry_ambiguity_count = 0
        exit_ambiguity_count = 0

        def monitor(minute_ts: int, high: Decimal, low: Decimal) -> None:
            nonlocal active, balance
            nonlocal entries_hit, trades_completed
            nonlocal entry_ambiguity_count, exit_ambiguity_count
            if active is None or minute_ts < active.decision_time_ms:
                return
            if not active.entered:
                outcome = evaluate_entry(_levels(active), high, low)
                if outcome is MonitorOutcome.ENTRY_HIT:
                    self._open_position(active, minute_ts)
                    entries_hit += 1
                elif outcome is MonitorOutcome.AMBIGUOUS:
                    entry_ambiguity_count += 1
                    active.was_ambiguous = True
                    ambiguous_events.append(
                        AmbiguityEvent(
                            signal_candle_ts=active.signal_candle_ts,
                            minute_ts=minute_ts,
                            kind="ENTRY",
                            direction=active.direction,
                        )
                    )
                return
            outcome = evaluate_candle(_levels(active), high, low)
            if outcome is MonitorOutcome.MONITORING:
                return
            if outcome is MonitorOutcome.AMBIGUOUS:
                exit_ambiguity_count += 1
                active.was_ambiguous = True
                ambiguous_events.append(
                    AmbiguityEvent(
                        signal_candle_ts=active.signal_candle_ts,
                        minute_ts=minute_ts,
                        kind="EXIT",
                        direction=active.direction,
                    )
                )
                return
            net = self._close_position(active, outcome, minute_ts)
            balance = balance + net
            equity_curve.append(balance)
            trades_completed += 1
            trades.append(self._finalize_trade(active))
            pending_events.append(self._pending_event(active))
            active = None

        for i in range(first_eligible, len(hours)):
            candle = hours[i]
            decision_time = candle.timestamp + decision_period_ms

            while (
                minute_idx < minute_count
                and minutes[minute_idx].timestamp < decision_time
            ):
                minute_candle = minutes[minute_idx]
                try:
                    monitor(
                        minute_candle.timestamp,
                        Decimal(str(minute_candle.high)),
                        Decimal(str(minute_candle.low)),
                    )
                except (ValueError, TypeError, ArithmeticError, InvalidOperation) as exc:
                    raise BacktestExecutionError(
                        f"invalid minute candle at {minute_candle.timestamp}: {exc}"
                    ) from exc
                minute_idx += 1

            if active is not None:
                blocked_count += 1
                blocked_events.append(
                    BlockedEvent(
                        candle_ts=candle.timestamp,
                        decision_time_ms=decision_time,
                        reason="an active signal exists; AI must not run",
                        active_direction=active.direction,
                        active_signal_candle_ts=active.signal_candle_ts,
                    )
                )
                continue

            window = hours[: i + 1]
            indicators = compute_indicator_matrix(window)
            try:
                analysis = self.provider.decide(
                    window,
                    indicators,
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    max_candles=config.max_candles,
                )
            except Exception as exc:
                raise BacktestExecutionError(
                    f"decision provider failed for candle {candle.timestamp}: {exc}"
                ) from exc
            if not isinstance(analysis, SignalAnalysis):
                raise BacktestExecutionError(
                    "decision provider must return a SignalAnalysis"
                )
            analyzed += 1
            decisions.append(
                DecisionRecord(
                    candle_ts=candle.timestamp,
                    decision=analysis.decision,
                    confidence=analysis.confidence,
                    candle_close_price=analysis.candle_close_price,
                    provider=self.provider.provider,
                    model=self.provider.model,
                )
            )
            if analysis.decision not in ("LONG", "SHORT", "WAIT"):
                raise BacktestExecutionError(
                    f"provider returned unknown decision {analysis.decision!r} "
                    f"for candle {candle.timestamp}"
                )
            if analysis.decision == "WAIT":
                wait_count += 1
                continue
            if analysis.decision == "LONG":
                long_decisions += 1
            else:
                short_decisions += 1
            try:
                candidate = validate_analysis(analysis)
            except SignalValidationError as exc:
                rejected_count += 1
                rejected_events.append(
                    RejectedEvent(
                        candle_ts=candle.timestamp,
                        decision=analysis.decision,
                        reason=str(exc),
                    )
                )
                continue

            active = _ActiveSignal(
                signal_candle_ts=candle.timestamp,
                decision_time_ms=decision_time,
                direction=candidate.decision,
                entry=candidate.entry,
                stop_loss=candidate.stop_loss,
                take_profit=candidate.take_profit,
                ndx=created_counter,
            )
            created_counter += 1

        # After the last decision candle, keep monitoring remaining minutes so a
        # signal created on (or waiting since) the last candle is still resolved.
        while minute_idx < minute_count:
            minute_candle = minutes[minute_idx]
            try:
                monitor(
                    minute_candle.timestamp,
                    Decimal(str(minute_candle.high)),
                    Decimal(str(minute_candle.low)),
                )
            except (ValueError, TypeError, ArithmeticError, InvalidOperation) as exc:
                raise BacktestExecutionError(
                    f"invalid minute candle at {minute_candle.timestamp}: {exc}"
                ) from exc
            minute_idx += 1

        pending_created = created_counter
        if active is not None:
            if active.entered:
                trades.append(self._finalize_trade(active))
                pending_events.append(self._pending_event(active))
                open_at_end = 1
                pending_at_end = 0
            else:
                pending_events.append(self._pending_event(active))
                open_at_end = 0
                pending_at_end = 1
        else:
            open_at_end = 0
            pending_at_end = 0

        long_created = 0
        short_created = 0
        for event in pending_events:
            if event.direction == DIRECTION_LONG:
                long_created += 1
            else:
                short_created += 1

        return BacktestResult(
            config=config,
            symbol=config.symbol,
            timeframe=config.timeframe,
            execution_interval=config.execution_interval,
            eligible=total_eligible,
            analyzed=analyzed,
            wait_count=wait_count,
            long_decisions=long_decisions,
            short_decisions=short_decisions,
            rejected_count=rejected_count,
            blocked_count=blocked_count,
            long_created=long_created,
            short_created=short_created,
            pending_created=pending_created,
            entries_hit=entries_hit,
            trades_completed=trades_completed,
            open_at_end=open_at_end,
            pending_at_end=pending_at_end,
            ambiguous_total=entry_ambiguity_count + exit_ambiguity_count,
            entry_ambiguity_count=entry_ambiguity_count,
            exit_ambiguity_count=exit_ambiguity_count,
            trades=tuple(trades),
            pending_events=tuple(pending_events),
            ambiguous_events=tuple(ambiguous_events),
            blocked_events=tuple(blocked_events),
            rejected_events=tuple(rejected_events),
            decisions=tuple(decisions),
            equity_curve=tuple(equity_curve),
            final_balance=balance,
        )

    def _open_position(self, active: _ActiveSignal, minute_ts: int) -> None:
        config = self.config
        notional = position_size(config.margin_per_trade, config.leverage)
        factor = _slippage_factor(
            config.slippage_bps, direction=active.direction, entering=True
        )
        entry_exec = active.entry * factor
        quantity = position_quantity(notional, entry_exec)
        active.entered = True
        active.entry_time_ms = minute_ts
        active.entry_price = entry_exec
        active.quantity = quantity
        active.position_size = notional
        active.entry_fee = notional * config.fee_rate
        active.risk_amount = abs(active.entry - active.stop_loss) * quantity

    def _close_position(
        self, active: _ActiveSignal, outcome: MonitorOutcome, minute_ts: int
    ) -> Decimal:
        """Fill the configured exit level with slippage and compute net PnL.

        Returns the net PnL so the caller can update the balance exactly once.
        """
        config = self.config
        if outcome is MonitorOutcome.TP_HIT:
            exit_level = active.take_profit
            reason = CLOSE_REASON_TP
        else:
            exit_level = active.stop_loss
            reason = CLOSE_REASON_SL
        factor = _slippage_factor(
            config.slippage_bps, direction=active.direction, entering=False
        )
        exit_exec = exit_level * factor
        gross = gross_pnl(
            active.direction, active.entry_price, exit_exec, active.quantity
        )
        fee_exit = _exit_fee(exit_exec, active.quantity, config.fee_rate)
        funding = self._funding_cost(active, minute_ts)
        net = gross - active.entry_fee - fee_exit - funding
        risk = active.risk_amount
        active.exit_time_ms = minute_ts
        active.exit_price = exit_exec
        active.exit_fee = fee_exit
        active.funding_cost = funding
        active.gross_pnl = gross
        active.net_pnl = net
        active.outcome = reason
        active.r_multiple = net / risk if risk > 0 else None
        return net

    def _funding_cost(self, active: _ActiveSignal, exit_time_ms: int) -> Decimal:
        rates = self.config.funding_rates
        if not rates:
            return Decimal("0")
        total = Decimal("0")
        start = active.entry_time_ms
        end = exit_time_ms
        for timestamp, rate in rates.items():
            if start <= timestamp <= end:
                total += rate * active.position_size
        return total

    def _pending_event(self, active: _ActiveSignal) -> PendingEvent:
        return PendingEvent(
            signal_candle_ts=active.signal_candle_ts,
            decision_time_ms=active.decision_time_ms,
            direction=active.direction,
            entry_level=active.entry,
            stop_level=active.stop_loss,
            take_profit_level=active.take_profit,
            entered=active.entered,
            entered_at_ms=active.entry_time_ms,
            closed=active.outcome is not None,
            closed_at_ms=active.exit_time_ms,
            outcome=active.outcome,
            trade_index=active.ndx,
        )

    def _finalize_trade(self, active: _ActiveSignal) -> BacktestTrade:
        return BacktestTrade(
            index=active.ndx,
            signal_candle_ts=active.signal_candle_ts,
            direction=active.direction,
            entry_level=active.entry,
            stop_level=active.stop_loss,
            take_profit_level=active.take_profit,
            entry_time_ms=active.entry_time_ms,
            exit_time_ms=active.exit_time_ms,
            entry_price=active.entry_price,
            exit_price=active.exit_price,
            quantity=active.quantity,
            position_size=active.position_size,
            entry_fee=active.entry_fee,
            exit_fee=active.exit_fee,
            funding_cost=active.funding_cost,
            gross_pnl=active.gross_pnl,
            net_pnl=active.net_pnl,
            risk_amount=active.risk_amount,
            r_multiple=active.r_multiple,
            outcome=active.outcome,
            was_ambiguous=active.was_ambiguous,
            holding_time_ms=(active.exit_time_ms - active.entry_time_ms)
            if active.exit_time_ms is not None
            else None,
        )


def _levels(active: _ActiveSignal) -> _Levels:
    return _Levels(
        direction=active.direction,
        entry=active.entry,
        stop_loss=active.stop_loss,
        take_profit=active.take_profit,
    )


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "BlockedEvent",
    "CLOSE_REASON_SL",
    "CLOSE_REASON_TP",
    "AmbiguityEvent",
    "DecisionRecord",
    "PendingEvent",
    "RejectedEvent",
]
