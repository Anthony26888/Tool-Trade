"""Phase 12: end-to-end CLI smoke tests for ``python -m backtest``.

The CLI reads an on-disk historical dataset + decisions file, runs the engine,
writes summary output, and exports JSON + CSV. All files live under tmp_path.
"""

from __future__ import annotations

import csv
import json

from backtest.__main__ import main
from backtest.data import HistoricalData, save_historical_data_json
from backtest.metrics import BacktestStatistics
from tests.backtest_helpers import long_decision, make_hours, trade_minutes


def _fixture(tmp_path):
    hours = make_hours(240)
    minutes = trade_minutes(hours)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(minutes),
    )
    data_path = tmp_path / "data.json"
    save_historical_data_json(str(data_path), data)
    decisions = {hours[i].timestamp: long_decision(float(hours[i].close)) for i in (199, 204, 209)}
    decisions_path = tmp_path / "decisions.json"
    with open(decisions_path, "w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in decisions.items()}, handle)
    return str(data_path), str(decisions_path)


def test_cli_end_to_end(tmp_path, capsys):
    data_path, decisions_path = _fixture(tmp_path)
    out = tmp_path / "out"
    code = main(
        [
            "--data",
            data_path,
            "--decisions",
            decisions_path,
            "--out",
            str(out),
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "Eligible        41" in captured
    assert "Analyzed        41" in captured
    assert "Wrote" in captured

    result_path = out / "backtest_result.json"
    trades_path = out / "backtest_trades.csv"
    assert result_path.exists()
    assert trades_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["counts"]["pending_created"] == 3
    assert payload["counts"]["trades_completed"] == 3
    assert payload["counts"]["eligible"] == 41
    assert len(payload["trades"]) == 3
    stats = BacktestStatistics.from_dict(payload["statistics"])
    assert stats.wins == 3
    with open(trades_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["outcome"] == "TP" for row in rows)


def test_cli_returns_error_code_on_bad_data(tmp_path, capsys):
    data_path, decisions_path = _fixture(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("", encoding="utf-8")
    code = main(["--data", str(bad), "--decisions", decisions_path, "--out", str(tmp_path / "o2")])
    assert code == 2
    assert "backtest failed" in capsys.readouterr().err
    assert not (tmp_path / "o2" / "backtest_result.json").exists()
