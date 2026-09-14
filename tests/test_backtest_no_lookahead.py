"""Phase 12: no-look-ahead guarantees (AGENTS.md section 10).

These tests prove the engine NEVER lets a decision or an execution use
information that would not exist at that point in time:

- window slicing: the provider for candle ``i`` sees exactly ``candles[:i+1]``
  and indicators recomputed from that window;
- future-candle isolation: mutating candles AFTER a boundary must produce
  identical decisions, identical indicator inputs, and identical trades that
  closed before the boundary;
- minute gating: entry execution never consumes a 1m candle whose open time
  falls before the decision time.
"""

from __future__ import annotations

from backtest.engine import BacktestEngine
from backtest.provider import FunctionDecisionProvider, ReplayDecisionProvider
from binance.indicators import compute_indicator_matrix
from binance.market_data import Candle
from tests.backtest_helpers import (
    MINUTE_MS,
    backtest_config,
    dataset,
    long_decision,
    make_hours,
    minute,
    trade_minutes,
)

BOUNDARY_INDEX = 220


def _rule_factory(observations):
    def rule(candles, indicators, *, symbol=None, timeframe=None, max_candles=None):
        last = indicators.iloc[-1]
        ts = int(last["timestamp"])
        close = float(last["close"])
        ema20 = float(last["ema20"])
        index = len(candles) - 1
        observations.append((ts, len(candles), round(close, 6), round(ema20, 6)))
        if index % 2 == 0 and close > ema20:
            return long_decision(close)
        return "WAIT"

    return rule


def _mutate_future(hours, minutes):
    """Return a copy of the dataset with every hour candle >= the boundary changed."""
    boundary_ts = hours[BOUNDARY_INDEX].timestamp
    mutated = []
    for candle in hours:
        if candle.timestamp < boundary_ts:
            mutated.append(candle)
            continue
        position = len(mutated)
        price = 30000.0 + position * 10.0
        mutated.append(
            Candle(
                timestamp=candle.timestamp,
                open=price - 5,
                high=price + 10,
                low=price - 10,
                close=price,
                volume=candle.volume,
                close_time=candle.close_time,
                is_closed=True,
            )
        )
    return dataset(mutated, trade_minutes(mutated))


def test_decisions_use_only_past_and_current_candle():
    hours = make_hours(240)
    data = dataset(hours, trade_minutes(hours))

    def recording_rule(candles, indicators, *, symbol=None, timeframe=None, max_candles=None):
        index = len(candles) - 1
        last = indicators.iloc[-1]
        assert len(candles) == index + 1
        assert last["timestamp"] == hours[index].timestamp
        assert candles[-1].timestamp == hours[index].timestamp
        return "WAIT"

    provider = FunctionDecisionProvider(recording_rule)
    result = BacktestEngine(backtest_config(), data, provider).run()
    assert result.analyzed == result.eligible
    assert result.decisions[0].candle_ts == hours[199].timestamp
    assert result.decisions[-1].candle_ts == hours[239].timestamp


def test_indicators_passed_are_recomputed_from_window_only():
    hours = make_hours(240)
    observations = []
    provider = FunctionDecisionProvider(_rule_factory(observations))
    BacktestEngine(backtest_config(), dataset(hours, trade_minutes(hours)), provider).run()
    for index, (ts, window_len, close, ema20) in enumerate(observations):
        expected_index = 199 + index
        assert ts == hours[expected_index].timestamp
        assert window_len == expected_index + 1
        assert close == round(float(hours[expected_index].close), 6)
        expected_ema = round(
            float(compute_indicator_matrix(hours[: expected_index + 1]).iloc[-1]["ema20"]),
            6,
        )
        assert ema20 == expected_ema


def test_future_candle_mutation_does_not_change_past_decisions():
    hours = make_hours(240)
    data = dataset(hours, trade_minutes(hours))
    boundary = hours[BOUNDARY_INDEX].timestamp

    observations_a, observations_b = [], []

    def rule_a(candles, indicators, *, symbol=None, timeframe=None, max_candles=None):
        last = indicators.iloc[-1]
        ts = int(last["timestamp"])
        observations_a.append(
            (ts, len(candles), round(float(last["close"]), 6), round(float(last["ema20"]), 6))
        )
        index = len(candles) - 1
        if index % 2 == 0 and ts < boundary:
            return long_decision(float(last["close"]))
        return "WAIT"

    def rule_b(candles, indicators, *, symbol=None, timeframe=None, max_candles=None):
        last = indicators.iloc[-1]
        ts = int(last["timestamp"])
        observations_b.append(
            (ts, len(candles), round(float(last["close"]), 6), round(float(last["ema20"]), 6))
        )
        index = len(candles) - 1
        if index % 2 == 0 and ts < boundary:
            return long_decision(float(last["close"]))
        return "WAIT"

    result_a = BacktestEngine(backtest_config(), data, FunctionDecisionProvider(rule_a)).run()
    data_b = _mutate_future(hours, trade_minutes(hours))
    result_b = BacktestEngine(backtest_config(), data_b, FunctionDecisionProvider(rule_b)).run()

    pre_a = [record for record in observations_a if record[0] < boundary]
    pre_b = [record for record in observations_b if record[0] < boundary]
    assert pre_a == pre_b

    decisions_a = {d.candle_ts: d for d in result_a.decisions}
    decisions_b = {d.candle_ts: d for d in result_b.decisions}
    for candle_ts in [hours[i].timestamp for i in range(199, BOUNDARY_INDEX)]:
        assert decisions_a[candle_ts].decision == decisions_b[candle_ts].decision
        assert decisions_a[candle_ts].confidence == decisions_b[candle_ts].confidence

    def pre_boundary_trades(result):
        return [
            (
                trade.signal_candle_ts,
                trade.outcome,
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.gross_pnl,
                trade.net_pnl,
            )
            for trade in result.trades
            if trade.exit_time_ms is not None and trade.exit_time_ms < boundary
        ]

    assert len(pre_boundary_trades(result_a)) > 0
    assert pre_boundary_trades(result_a) == pre_boundary_trades(result_b)

    def pre_events(result):
        return [
            (event.candle_ts, getattr(event, "reason", getattr(event, "kind", None)))
            for event in (
                list(result.blocked_events)
                + list(result.rejected_events)
                + list(result.ambiguous_events)
            )
            if event.candle_ts < boundary
        ]

    assert pre_events(result_a) == pre_events(result_b)


def test_minute_gate_excludes_signal_candle_minutes():
    hours = make_hours(220)
    minutes = []
    for hour in hours:
        base = float(hour.close)
        for offset in range(60):
            ts = hour.timestamp + offset * MINUTE_MS
            if hour.timestamp == hours[199].timestamp and offset == 20:
                # In-candle spike that would fill entry AND trip TP/SL if used.
                high, low = base + 5000.0, base - 5000.0
            elif hour.timestamp == hours[200].timestamp and offset == 20:
                high, low = base + 200.0, base - 2.0  # post-decision TP spike
            else:
                high, low = base + 2.0, base - 2.0
            minutes.append(minute(ts, base, high, low, base))
    decision = long_decision(float(hours[199].close))  # entry close - 5
    data = dataset(hours, minutes)
    result = BacktestEngine(
        backtest_config(),
        data,
        ReplayDecisionProvider({hours[199].timestamp: decision}),
    ).run()
    # The huge in-candle spike is unusable: it opens before the decision time.
    # The entry fills on the first post-decision minute (hour 200, flat +/-2),
    # then TP trips at hour 200 minute 20.
    assert result.entries_hit == 1
    assert result.trades_completed == 1
    trade = result.trades[0]
    assert trade.entry_time_ms == hours[200].timestamp
    assert trade.outcome == "TP"
    assert trade.exit_time_ms == hours[200].timestamp + 20 * MINUTE_MS


def test_all_executed_trades_stem_from_pre_boundary_decisions():
    hours = make_hours(240)
    data = dataset(hours, trade_minutes(hours))
    boundary = hours[BOUNDARY_INDEX].timestamp

    def rule(candles, indicators, *, symbol=None, timeframe=None, max_candles=None):
        last = indicators.iloc[-1]
        ts = int(last["timestamp"])
        index = len(candles) - 1
        if index % 3 == 0 and ts < boundary:
            return long_decision(float(last["close"]))
        return "WAIT"

    provider = FunctionDecisionProvider(rule)
    result = BacktestEngine(backtest_config(), data, provider).run()
    assert len(result.trades) > 0
    assert all(trade.signal_candle_ts < boundary for trade in result.trades)
    for trade in result.trades:
        decision_hour = next(h for h in hours if h.timestamp == trade.signal_candle_ts)
        assert trade.entry_time_ms >= decision_hour.timestamp + MINUTE_MS * 60
