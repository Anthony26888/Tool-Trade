"""Shared builders for Phase 12 backtest tests (not collected by pytest)."""

from __future__ import annotations

from decimal import Decimal

from backtest.config import BacktestConfig
from backtest.data import HistoricalData
from backtest.engine import (
    AmbiguityEvent,
    BacktestResult,
    BacktestTrade,
    BlockedEvent,
    DecisionRecord,
    PendingEvent,
    RejectedEvent,
)
from binance.market_data import Candle
from tests.signal_engine_test_helpers import make_candles

HOUR_MS = 3_600_000
MINUTE_MS = 60_000
DEFAULT_START_MS = 1_720_000_000_000


def backtest_config(**overrides) -> BacktestConfig:
    kwargs = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "execution_interval": "1m",
        "initial_balance": "1000",
        "margin_per_trade": "50",
        "leverage": 10,
        "fee_rate": "0.0004",
        "slippage_bps": "0",
        "min_candles": 200,
        "max_candles": 40,
    }
    kwargs.update(overrides)
    for money_field in ("initial_balance", "margin_per_trade", "fee_rate", "slippage_bps"):
        if not isinstance(kwargs[money_field], Decimal):
            kwargs[money_field] = Decimal(str(kwargs[money_field]))
    return BacktestConfig(**kwargs)


def make_hours(count: int = 220, *, base: float = 60000.0, trend: float = 0.5) -> list[Candle]:
    return make_candles(count, start_ms=DEFAULT_START_MS, base=base, trend=trend)


def minute(ts: int, open_, high, low, close, volume: float = 1.0) -> Candle:
    return Candle(ts, float(open_), float(high), float(low), float(close), volume, ts + MINUTE_MS - 1, True)


def covering_minutes(
    hours: list[Candle], *, high_pad: float = 2.0, low_pad: float = 2.0, volume: float = 1.0
) -> list[Candle]:
    """One flat 1m candle per minute across every hour block."""
    minutes: list[Candle] = []
    for hour in hours:
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            price = float(hour.close)
            minutes.append(minute(ts, price, price + high_pad, price - low_pad, price, volume))
    return minutes


def trade_minutes(hours: list[Candle], *, volume: float = 1.0) -> list[Candle]:
    """Minutes designed for fast deterministic fills/TP/SL.

    Within each hour block: minute 0 jumps (clean entry fill), minute 20 has a
    high spike (LONG TP / SHORT SL), minute 40 has a low spike (SHORT TP /
    LONG SL). Other minutes are flat.
    """
    minutes: list[Candle] = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if offset == 0:
                high, low = base + 40.0, base - 40.0
            elif offset == 20:
                high, low = base + 200.0, base - 2.0
            elif offset == 40:
                high, low = base + 2.0, base - 200.0
            else:
                high, low = base + 2.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base, volume))
    return minutes


def dataset(
    hours: list[Candle] | None = None,
    minutes: list[Candle] | None = None,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    execution_interval: str = "1m",
) -> HistoricalData:
    if hours is None:
        hours = make_hours()
    if minutes is None:
        minutes = covering_minutes(hours)
    return HistoricalData(
        symbol=symbol,
        timeframe=timeframe,
        execution_interval=execution_interval,
        hour_candles=tuple(hours),
        minute_candles=tuple(minutes),
    )


def long_decision(close: float) -> dict:
    return {
        "decision": "LONG",
        "confidence": 90.0,
        "reasoning": "backtest helper",
        "entry_price": close - 5.0,
        "stop_loss": close - 50.0,
        "take_profit": close + 60.0,
    }


def short_decision(close: float) -> dict:
    return {
        "decision": "SHORT",
        "confidence": 90.0,
        "reasoning": "backtest helper",
        "entry_price": close + 5.0,
        "stop_loss": close + 50.0,
        "take_profit": close - 60.0,
    }


def long_replay(hours: list[Candle], indices: list[int]) -> dict[int, dict]:
    return {
        hours[index].timestamp: long_decision(float(hours[index].close))
        for index in indices
    }


def make_trade(
    index: int = 0,
    *,
    signal_candle_ts: int = 0,
    direction: str = "LONG",
    entry_level: str = "60000",
    stop_level: str = "59950",
    take_profit_level: str = "60100",
    entry_time_ms: int = 0,
    exit_time_ms: int | None = None,
    entry_price: str | None = None,
    exit_price: str | None = None,
    quantity: str = "0.00833333",
    position_size: str = "500",
    entry_fee: str | None = None,
    exit_fee: str | None = None,
    funding_cost: str = "0",
    gross_pnl: str | None = None,
    net_pnl: str | None = None,
    risk_amount: str = "10",
    r_multiple: str | None = None,
    outcome: str | None = None,
    was_ambiguous: bool = False,
    holding_time_ms: int | None = None,
) -> BacktestTrade:
    def dec(value: str | None) -> Decimal | None:
        return Decimal(value) if value is not None else None

    return BacktestTrade(
        index=index,
        signal_candle_ts=signal_candle_ts,
        direction=direction,
        entry_level=dec(entry_level),
        stop_level=dec(stop_level),
        take_profit_level=dec(take_profit_level),
        entry_time_ms=entry_time_ms,
        exit_time_ms=exit_time_ms,
        entry_price=dec(entry_price),
        exit_price=dec(exit_price),
        quantity=dec(quantity),
        position_size=dec(position_size),
        entry_fee=dec(entry_fee),
        exit_fee=dec(exit_fee),
        funding_cost=dec(funding_cost),
        gross_pnl=dec(gross_pnl),
        net_pnl=dec(net_pnl),
        risk_amount=dec(risk_amount),
        r_multiple=dec(r_multiple),
        outcome=outcome,
        was_ambiguous=was_ambiguous,
        holding_time_ms=holding_time_ms,
    )


def build_result(
    *,
    trades: tuple[BacktestTrade, ...] = (),
    pending_events: tuple[PendingEvent, ...] = (),
    ambiguous_events: tuple[AmbiguityEvent, ...] = (),
    blocked_events: tuple[BlockedEvent, ...] = (),
    rejected_events: tuple[RejectedEvent, ...] = (),
    decisions: tuple[DecisionRecord, ...] = (),
    equity_curve: tuple[Decimal, ...] | None = None,
    final_balance: Decimal | None = None,
    initial_balance: str = "1000",
    eligible: int = 0,
    analyzed: int = 0,
    wait_count: int = 0,
    long_decisions: int = 0,
    short_decisions: int = 0,
    rejected_count: int = 0,
    blocked_count: int = 0,
    pending_created: int = 0,
    entries_hit: int = 0,
    trades_completed: int = 0,
    open_at_end: int = 0,
    pending_at_end: int = 0,
    ambiguous_total: int = 0,
    entry_ambiguity_count: int = 0,
    exit_ambiguity_count: int = 0,
    long_created: int = 0,
    short_created: int = 0,
) -> BacktestResult:
    start = Decimal(initial_balance)
    if equity_curve is None:
        equity_curve = (start,)
    if final_balance is None:
        final_balance = start
    return BacktestResult(
        config=backtest_config(initial_balance=initial_balance),
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        eligible=eligible,
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
        ambiguous_total=ambiguous_total,
        entry_ambiguity_count=entry_ambiguity_count,
        exit_ambiguity_count=exit_ambiguity_count,
        trades=trades,
        pending_events=pending_events,
        ambiguous_events=ambiguous_events,
        blocked_events=blocked_events,
        rejected_events=rejected_events,
        decisions=decisions,
        equity_curve=equity_curve,
        final_balance=final_balance,
    )
