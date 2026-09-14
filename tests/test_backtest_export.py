"""Phase 12: deterministic backtest export (CSV + JSON).

Exports are pure file writers: no database, no network, only the path the
caller provides. Values round-trip through serialization unchanged.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal

import pytest

from backtest.config import BacktestExportError
from backtest.engine import PendingEvent
from backtest.export import (
    export_result_json,
    export_trades_csv,
    result_to_dict,
    trade_to_record,
)
from backtest.metrics import BacktestStatistics
from tests.backtest_helpers import backtest_config, build_result, make_trade


def _closed_trade():
    return make_trade(
        0,
        signal_candle_ts=1_720_000_000_000,
        direction="LONG",
        entry_level="60000",
        stop_level="59950",
        take_profit_level="60100",
        entry_time_ms=1_720_003_600_000,
        exit_time_ms=1_720_004_800_000,
        entry_price="60000.5",
        exit_price="60120",
        quantity="0.00833333",
        position_size="500",
        entry_fee="0.2",
        exit_fee="0.21",
        funding_cost="0",
        gross_pnl="0.997",
        net_pnl="0.587",
        risk_amount="10",
        r_multiple="0.0997",
        outcome="TP",
        holding_time_ms=1_200_000,
    )


def _open_trade():
    return make_trade(
        0,
        signal_candle_ts=1_720_000_000_000,
        direction="SHORT",
        entry_level="60000",
        stop_level="60050",
        take_profit_level="59900",
        entry_time_ms=1_720_003_600_000,
        entry_price="59999.9",
        quantity="0.00833333",
        position_size="500",
        entry_fee="0.2",
        funding_cost="0",
        risk_amount="10",
        outcome=None,
    )


def _result(trade):
    closed = trade.outcome == "TP"
    return build_result(
        trades=(trade,),
        pending_events=(
            PendingEvent(
                signal_candle_ts=1_720_000_000_000,
                decision_time_ms=1_720_003_600_000,
                direction=trade.direction,
                entry_level=trade.entry_level,
                stop_level=trade.stop_level,
                take_profit_level=trade.take_profit_level,
                entered=True,
                entered_at_ms=trade.entry_time_ms,
                closed=closed,
                closed_at_ms=trade.exit_time_ms,
                outcome=trade.outcome,
                trade_index=0,
            ),
        ),
        pending_created=1,
        entries_hit=1,
        trades_completed=1 if closed else 0,
        final_balance=Decimal("1000.587") if closed else Decimal("1000"),
        equity_curve=(Decimal("1000"), Decimal("1000.587")) if closed else (Decimal("1000"),),
        initial_balance="1000",
        eligible=10,
        analyzed=4,
    )


def test_trade_to_record_json_safe():
    record = trade_to_record(_closed_trade())
    assert record["net_pnl"] == "0.587"
    assert record["entry_price"] == "60000.5"
    assert record["was_ambiguous"] is False
    assert record["outcome"] == "TP"
    json.dumps(record)  # must not raise


def test_export_trades_csv_round_trip(tmp_path):
    result = _result(_closed_trade())
    path = export_trades_csv(tmp_path / "trades.csv", result)
    assert path == str(tmp_path / "trades.csv")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["index"] == "0"
    assert row["direction"] == "LONG"
    assert row["net_pnl"] == "0.587"
    assert row["outcome"] == "TP"
    assert row["signal_candle_ts_ms"] == "1720000000000"


def test_export_result_json_with_statistics(tmp_path):
    result = _result(_closed_trade())
    statistics = BacktestStatistics.compute(backtest_config(), result)
    path = export_result_json(tmp_path / "result.json", result, statistics)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["symbol"] == "BTCUSDT"
    assert payload["counts"]["pending_created"] == 1
    assert payload["counts"]["trades_completed"] == 1
    assert payload["final_balance"] == "1000.587"
    assert payload["equity_curve"] == ["1000", "1000.587"]
    assert payload["trades"][0]["net_pnl"] == "0.587"
    assert payload["statistics"]["win_rate"] == "100"
    rebound = BacktestStatistics.from_dict(payload["statistics"])
    assert rebound.net_pnl == statistics.net_pnl


def test_result_to_dict_counts_and_events():
    payload = result_to_dict(_result(_closed_trade()))
    assert payload["counts"]["eligible"] == 10
    assert payload["counts"]["entries_hit"] == 1
    assert payload["pending_events"][0]["outcome"] == "TP"
    assert payload["trades"][0]["outcome"] == "TP"
    assert payload["decisions"] == []
    assert payload["ambiguous_events"] == []


def test_export_open_trade_serializes_nones(tmp_path):
    result = _result(_open_trade())
    path = export_trades_csv(tmp_path / "open.csv", result)
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["outcome"] == ""
    assert rows[0]["net_pnl"] == ""
    assert rows[0]["exit_price"] == ""
    json_path = export_result_json(tmp_path / "open.json", result)
    with open(json_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["trades"][0]["gross_pnl"] is None
    assert payload["counts"]["trades_completed"] == 0


def test_export_empty_trades_result(tmp_path):
    result = build_result(trades=(), final_balance=Decimal("1000"), equity_curve=(Decimal("1000"),))
    path = export_trades_csv(tmp_path / "empty.csv", result)
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1  # header only
    assert "net_pnl" in rows[0]


def test_export_rejects_non_result(tmp_path):
    with pytest.raises(BacktestExportError):
        export_trades_csv(tmp_path / "bad.csv", "not a result")
    with pytest.raises(BacktestExportError):
        export_result_json(tmp_path / "bad.json", None)
