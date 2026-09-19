"""Phase 12: backtest engine lifecycle, money model, and counts."""

from __future__ import annotations

from decimal import Decimal

import pytest

from backtest.config import BacktestExecutionError
from backtest.engine import BacktestEngine
from backtest.provider import (
    FunctionDecisionProvider,
    ReplayDecisionProvider,
    WaitDecisionProvider,
)
from tests.backtest_helpers import (
    MINUTE_MS,
    backtest_config,
    covering_minutes,
    dataset,
    long_decision,
    make_hours,
    minute,
    short_decision,
    trade_minutes,
)


def _replay(hours, decisions_by_index, decision_fn=long_decision):
    return {hours[index].timestamp: decision_fn(float(hours[index].close)) for index in decisions_by_index}


def test_insufficient_data_raises():
    hours = make_hours(50)
    data = dataset(hours, covering_minutes(hours))
    with pytest.raises(BacktestExecutionError):
        BacktestEngine(backtest_config(), data, WaitDecisionProvider())


def test_data_symbol_timeframe_must_match_config():
    hours = make_hours(220)
    data = dataset(hours, symbol="ETHUSDT")
    with pytest.raises(BacktestExecutionError):
        BacktestEngine(backtest_config(), data, WaitDecisionProvider())


def test_wait_provider_creates_no_position():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    result = BacktestEngine(backtest_config(), data, WaitDecisionProvider()).run()
    assert result.eligible == len(hours) - 200 + 1
    assert result.analyzed == result.eligible
    assert result.wait_count == result.eligible
    assert result.long_decisions == 0
    assert result.short_decisions == 0
    assert result.pending_created == 0
    assert result.entries_hit == 0
    assert result.trades_completed == 0
    assert result.blocked_count == 0
    assert result.trades == ()
    assert result.final_balance == Decimal("1000")
    assert all(record.decision == "WAIT" for record in result.decisions)


def test_single_long_enters_and_hits_tp():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199]))
    ).run()
    assert result.pending_created == 1
    assert result.entries_hit == 1
    assert result.trades_completed == 1
    assert result.long_created == 1
    assert result.short_created == 0
    trade = result.trades[0]
    assert trade.outcome == "TP"
    assert trade.direction == "LONG"
    assert trade.entry_time_ms is not None and trade.exit_time_ms is not None
    assert trade.exit_time_ms == hours[200].timestamp + 20 * MINUTE_MS
    assert trade.net_pnl > 0
    assert result.final_balance > Decimal("1000")


def test_single_long_hits_sl():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if hour.timestamp == hours[200].timestamp:
                # No high spike: TP must never be touched.
                # Minute 40 has a LOW spike that trips the SL.
                if offset == 0:
                    high, low = base + 40.0, base - 40.0
                elif offset == 40:
                    high, low = base + 2.0, base - 400.0
                else:
                    high, low = base + 2.0, base - 2.0
            else:
                high, low = base + 40.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base))
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199]))
    ).run()
    assert result.trades_completed == 1
    trade = result.trades[0]
    assert trade.outcome == "SL"
    assert trade.exit_time_ms == hours[200].timestamp + 40 * MINUTE_MS
    assert trade.net_pnl < 0


def test_single_short_enters_and_hits_tp():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if offset == 0:
                high, low = base + 40.0, base - 40.0
            elif offset == 40:
                high, low = base + 2.0, base - 200.0
            else:
                high, low = base + 2.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base))
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199], short_decision))
    ).run()
    assert result.trades_completed == 1
    trade = result.trades[0]
    assert trade.outcome == "TP"
    assert trade.direction == "SHORT"
    assert trade.exit_time_ms == hours[200].timestamp + 40 * MINUTE_MS
    assert trade.net_pnl > 0


def test_single_short_hits_sl():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if offset == 0:
                high, low = base + 40.0, base - 40.0
            elif offset == 20:
                high, low = base + 400.0, base - 2.0
            else:
                high, low = base + 2.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base))
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199], short_decision))
    ).run()
    assert result.trades_completed == 1
    assert result.trades[0].outcome == "SL"
    assert result.trades[0].net_pnl < 0


def test_second_signal_blocked_while_active():
    hours = make_hours()
    data = dataset(hours, covering_minutes(hours))
    provider = ReplayDecisionProvider(_replay(hours, [199]))
    result = BacktestEngine(backtest_config(), data, provider).run()
    assert result.pending_created == 1
    assert result.entries_hit == 1
    assert result.blocked_count >= 1
    blocked = result.blocked_events[0]
    assert blocked.reason.startswith("an active signal")
    assert blocked.active_direction == "LONG"


def test_valid_long_blocked_never_analyzed_twice():
    hours = make_hours()
    data = dataset(hours, covering_minutes(hours))
    provider = ReplayDecisionProvider(_replay(hours, [199, 200, 201]))
    result = BacktestEngine(backtest_config(), data, provider).run()
    # The first decision is analyzed; the candle was entered at the next hour
    # start and stays OPEN, so candles 200+ are blocked, never analyzed.
    assert result.analyzed == 1
    assert result.blocked_count == result.eligible - result.analyzed
    assert result.trades_completed == 0


def test_invalid_long_is_rejected_not_opened():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    bad = {
        "decision": "LONG",
        "confidence": 80.0,
        "reasoning": "bad ordering",
        "entry_price": float(hours[199].close) + 5,
        "stop_loss": float(hours[199].close) - 50,
        "take_profit": float(hours[199].close) - 10,  # entry > take_profit -> invalid
    }
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider({hours[199].timestamp: bad})
    ).run()
    assert result.rejected_count == 1
    assert result.pending_created == 0
    assert result.trades == ()
    assert result.rejected_events[0].decision == "LONG"


def test_unknown_decision_raises():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    provider = ReplayDecisionProvider({hours[199].timestamp: {"decision": "HOLD"}})
    with pytest.raises(BacktestExecutionError):
        BacktestEngine(backtest_config(), data, provider).run()


def test_non_signal_analysis_result_raises():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    provider = FunctionDecisionProvider(lambda *args, **kwargs: 42)
    with pytest.raises(BacktestExecutionError):
        BacktestEngine(backtest_config(), data, provider).run()


def test_entry_monitoring_never_uses_signal_candle_minutes():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if offset == 10 and hour.timestamp == hours[199].timestamp:
                # Inside the signal candle: touches entry, must be ignored.
                high, low = base + 1000.0, base - 1000.0
            else:
                high, low = base + 2.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base))
    # Entry is placed ABOVE the reachable post-decision high so it can never fill.
    # TP keeps RR >= 1.2 (Phase B guardrail): TP distance 100 / SL distance 80.
    entry = float(hours[199].close) + 30
    decision = {
        "decision": "LONG",
        "confidence": 90.0,
        "reasoning": "gate",
        "entry_price": entry,
        "stop_loss": float(hours[199].close) - 50,
        "take_profit": float(hours[199].close) + 130,
    }
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(),
        data,
        ReplayDecisionProvider({hours[199].timestamp: decision}),
    ).run()
    # The in-candle minute touched entry, but it is below the decision time:
    # it must never fill, so the signal stays pending forever.
    assert result.entries_hit == 0
    assert result.pending_at_end == 1
    assert result.pending_created == 1
    assert result.blocked_count == result.eligible - 1


def test_pending_never_entered_blocks_later_candles():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            high = base + 150.0  # high enough to fill a normal entry
            low = base - 150.0  # and to trip TP/SL -> ambiguous on every candle
            minutes.append(minute(ts, base, high, low, base))
    data = dataset(hours, minutes)
    provider = ReplayDecisionProvider(_replay(hours, [199]))
    result = BacktestEngine(backtest_config(), data, provider).run()
    # Every minute candle touches entry AND an exit level: ambiguous, no fill.
    assert result.entries_hit == 0
    assert result.pending_at_end == 1
    assert result.entry_ambiguity_count >= 1
    assert all(event.kind == "ENTRY" for event in result.ambiguous_events)


def test_exit_ambiguity_keeps_monitoring_until_unambiguous():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if offset == 0:
                high, low = base + 40.0, base - 40.0
            elif offset == 20:
                high, low = base + 200.0, base - 200.0  # TP AND SL together
            elif offset == 40:
                high, low = base + 200.0, base - 2.0  # TP
            else:
                high, low = base + 2.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base))
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199]))
    ).run()
    assert result.entries_hit == 1
    assert result.exit_ambiguity_count == 1
    assert result.trades_completed == 1
    assert result.trades[0].outcome == "TP"
    assert result.trades[0].was_ambiguous is True
    assert result.ambiguous_total == 1


def test_entry_ambiguity_stays_pending():
    hours = make_hours()
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if offset == 5 and hour.timestamp > hours[199].timestamp:
                # One minute touches entry AND TP AND SL: order unknowable.
                high, low = base + 200.0, base - 200.0
            else:
                # Flat candles never reach the (fixed) entry level; the entry
                # stays strictly below base but far above any reachable high.
                high, low = base - 200.0, base - 400.0
            minutes.append(minute(ts, base, high, low, base))
    decision = long_decision(float(hours[199].close))
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(),
        data,
        ReplayDecisionProvider({hours[199].timestamp: decision}),
    ).run()
    assert result.entry_ambiguity_count >= 1
    assert result.entries_hit == 0
    assert result.pending_at_end == 1
    assert all(event.kind == "ENTRY" for event in result.ambiguous_events)


def test_open_position_at_end_of_data():
    hours = make_hours()
    data = dataset(hours, covering_minutes(hours))
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199]))
    ).run()
    assert result.open_at_end == 1
    assert result.trades_completed == 0
    trade = result.trades[0]
    assert trade.outcome is None
    assert trade.exit_time_ms is None
    assert trade.net_pnl is None
    assert result.final_balance == Decimal("1000")


def test_immutable_entry_tp_sl():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    decision = long_decision(float(hours[199].close))
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider({hours[199].timestamp: decision})
    ).run()
    trade = result.trades[0]
    assert trade.entry_level == Decimal(str(decision["entry_price"]))
    assert trade.stop_level == Decimal(str(decision["stop_loss"]))
    assert trade.take_profit_level == Decimal(str(decision["take_profit"]))
    event = result.pending_events[0]
    assert event.entry_level == trade.entry_level


def test_slippage_moves_execution_not_detection():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    decision = long_decision(float(hours[199].close))
    provider = ReplayDecisionProvider({hours[199].timestamp: decision})
    plain = BacktestEngine(backtest_config(), data, provider).run()
    slipped = BacktestEngine(
        backtest_config(slippage_bps="100"), data, provider
    ).run()
    plain_trade = plain.trades[0]
    slipped_trade = slipped.trades[0]
    # Level detection is unchanged; execution price absorbs the slippage.
    assert slipped_trade.entry_level == plain_trade.entry_level
    assert slipped_trade.entry_price > slipped_trade.entry_level
    assert slipped_trade.exit_price < slipped_trade.take_profit_level
    assert plain_trade.entry_price == plain_trade.entry_level
    assert plain_trade.exit_price == plain_trade.take_profit_level
    assert slipped_trade.net_pnl < plain_trade.net_pnl


def test_entry_fill_uses_margin_leverage_and_fee():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    result = BacktestEngine(
        backtest_config(), data, ReplayDecisionProvider(_replay(hours, [199]))
    ).run()
    trade = result.trades[0]
    expected_notional = Decimal("50") * Decimal(10)
    assert trade.position_size == expected_notional
    assert trade.entry_price == trade.entry_level  # no slippage
    assert trade.entry_fee == expected_notional * Decimal("0.0004")
    assert abs(trade.entry_price * trade.quantity - expected_notional) < Decimal("0.0001")
    expected_gross = (trade.exit_price - trade.entry_price) * trade.quantity
    assert trade.gross_pnl == expected_gross
    assert trade.net_pnl == trade.gross_pnl - trade.entry_fee - trade.exit_fee
    assert trade.quantity == expected_notional / trade.entry_price


def test_funding_rates_apply_only_while_open():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    base_config = backtest_config()
    mid_ts = hours[200].timestamp + 10 * MINUTE_MS
    rates = {mid_ts: Decimal("0.01")}
    provider = ReplayDecisionProvider(_replay(hours, [199]))
    plain = BacktestEngine(base_config, data, provider).run()
    funded = BacktestEngine(backtest_config(funding_rates=rates), data, provider).run()
    assert funded.trades[0].funding_cost == Decimal("5")
    assert funded.trades[0].net_pnl == plain.trades[0].net_pnl - Decimal("5")
    # Rates before entry or after exit never apply.
    before = {hours[100].timestamp: Decimal("0.01")}
    after = {hours[210].timestamp + 30 * MINUTE_MS: Decimal("0.01")}
    assert (
        BacktestEngine(backtest_config(funding_rates={**rates, **before, **after}), data, provider)
        .run()
        .trades[0]
        .funding_cost
        == Decimal("5")
    )


def test_deterministic_runs_identical():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    provider = ReplayDecisionProvider(_replay(hours, [199, 203, 207]))
    first = BacktestEngine(backtest_config(), data, provider).run()
    second = BacktestEngine(backtest_config(), data, provider).run()
    assert first.trades == second.trades
    assert first.decisions == second.decisions
    assert first.equity_curve == second.equity_curve
    assert first.final_balance == second.final_balance
    assert first.pending_events == second.pending_events
    assert first.ambiguous_events == second.ambiguous_events


def test_waits_between_signals_are_not_blocked():
    hours = make_hours()
    data = dataset(hours, trade_minutes(hours))
    provider = ReplayDecisionProvider(_replay(hours, [199, 203, 207]))
    result = BacktestEngine(backtest_config(), data, provider).run()
    assert result.pending_created == 3
    assert result.trades_completed == 3
    # WAIT candles between signals are analyzed, not blocked.
    assert result.wait_count > 0
    assert result.blocked_count == 0
    assert result.analyzed == result.eligible
