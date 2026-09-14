"""Phase 12: decision providers and coerce_decision."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.config import BacktestExecutionError
from backtest.provider import (
    FunctionDecisionProvider,
    ReplayDecisionProvider,
    WaitDecisionProvider,
    coerce_decision,
)
from binance.market_data import Candle
from signal_engine.analysis import SignalAnalysis, SignalDecision, SignalDecisionModel
from tests.backtest_helpers import long_decision, make_hours

CANDLE = Candle(
    1_728_000_000_000,
    60000.0,
    60060.0,
    59950.0,
    60055.0,
    100.0,
    1_728_003_600_000 - 1,
    True,
)


def test_coerce_string_decisions():
    for raw, expected in (("LONG", "LONG"), ("short", "SHORT"), ("WAIT", "WAIT")):
        analysis = coerce_decision(raw, candle=CANDLE, symbol="BTCUSDT", timeframe="1h")
        assert analysis.decision == expected
        assert analysis.candle_close_price == CANDLE.close


def test_coerce_invalid_string_rejected():
    with pytest.raises(BacktestExecutionError):
        coerce_decision("HOLD", candle=CANDLE, symbol="BTCUSDT", timeframe="1h")
    with pytest.raises(BacktestExecutionError):
        coerce_decision("", candle=CANDLE, symbol="BTCUSDT", timeframe="1h")


def test_coerce_dict_and_model():
    payload = {"decision": "SHORT", "confidence": 70.0, "reasoning": "r"}
    analysis = coerce_decision(payload, candle=CANDLE, symbol="BTCUSDT", timeframe="1h")
    assert isinstance(analysis, SignalAnalysis)
    assert analysis.decision == "SHORT"
    model = SignalDecisionModel(
        decision=SignalDecision.LONG,
        confidence=55.0,
        reasoning="model path",
        entry_price=60000.0,
        stop_loss=59900.0,
        take_profit=60200.0,
    )
    from_signal = coerce_decision(model, candle=CANDLE, symbol="BTCUSDT", timeframe="1h")
    assert from_signal.entry_price == 60000.0


def test_coerce_analysis_keeps_decision_fields():
    source = SignalAnalysis(
        decision="SHORT",
        confidence=77.5,
        reasoning="carried over",
        entry_price=60060.0,
        stop_loss=60120.0,
        take_profit=59900.0,
        provider="x",
        model="y",
        symbol="BTCUSDT",
        timeframe="1h",
        analysis_timestamp="t",
        market_timestamp="m",
        closed_at="c",
        candle_close_price=60055.0,
    )
    coerced = coerce_decision(source, candle=CANDLE, symbol="BTCUSDT", timeframe="1h")
    assert coerced.decision == "SHORT"
    assert coerced.confidence == 77.5
    assert coerced.reasoning == "carried over"
    assert coerced.entry_price == 60060.0
    # Metadata is restamped from the current candle, not the stale source.
    expected_market = datetime.fromtimestamp(
        CANDLE.timestamp / 1000.0, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert coerced.market_timestamp == expected_market
    assert coerced.market_timestamp != "m"


def test_coerce_rejects_unrecognized_payload():
    with pytest.raises(BacktestExecutionError):
        coerce_decision(42, candle=CANDLE, symbol="BTCUSDT", timeframe="1h")


def test_wait_provider_never_decides_long():
    provider = WaitDecisionProvider()
    hours = make_hours(210)
    analysis = provider.decide(hours, None)
    assert isinstance(analysis, SignalAnalysis)
    assert analysis.decision == "WAIT"


def test_function_provider_wraps_callable():
    calls = []

    def rule(candles, indicators, *, symbol, timeframe, max_candles):
        calls.append((len(candles), symbol, timeframe, max_candles))
        return "LONG"

    provider = FunctionDecisionProvider(rule)
    hours = make_hours(205)
    analysis = provider.decide(list(hours), None, symbol="BTCUSDT", timeframe="1h")
    assert analysis.decision == "LONG"
    assert calls == [(205, "BTCUSDT", "1h", 40)]


def test_replay_provider_missing_key_waits():
    close = 60000.0
    provider = ReplayDecisionProvider({CANDLE.timestamp: long_decision(close)})
    candles = [CANDLE]
    assert provider.decide(candles, None).decision == "LONG"
    other = Candle(1730000000, 60000.0, 60000.0, 60000.0, 60000.0, 1.0, 1730003600 - 1, True)
    assert provider.decide([other], None).decision == "WAIT"


def test_replay_provider_invalid_payload_raises():
    provider = ReplayDecisionProvider({CANDLE.timestamp: {"decision": "HOLD"}})
    with pytest.raises(BacktestExecutionError):
        provider.decide([CANDLE], None)
