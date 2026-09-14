"""Phase 12: backtest statistics (AGENTS.md section 24, metrics block).

Statistics are derived from the engine result deterministically; PnL numbers
used here are synthetic, factory-built trades (not engine-produced) so each
metric is asserted against an independently hand-computed value.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from backtest.config import BacktestMetricError
from backtest.metrics import BacktestStatistics
from tests.backtest_helpers import (
    backtest_config,
    build_result,
    make_trade,
)

Q = Decimal("0.1")


def _build_stats(
    *,
    equity=(Decimal("1000"), Decimal("1010.0"), Decimal("1007.9"), Decimal("1037.8")),
    final_balance=None,
    trades=None,
    initial_balance="1000",
):
    if trades is None:
        trades = [
            make_trade(
                0,
                direction="LONG",
                entry_level="60000",
                stop_level="59950",
                take_profit_level="60100",
                entry_price="59999",
                exit_price="60100",
                quantity=str(Q),
                entry_fee="0.05",
                exit_fee="0.05",
                gross_pnl="10.1",
                net_pnl="10.0",
                r_multiple="1.0",
                holding_time_ms=120,
                outcome="TP",
            ),
            make_trade(
                1,
                direction="SHORT",
                entry_level="60500",
                stop_level="60550",
                take_profit_level="60400",
                entry_price="60500",
                exit_price="60520",
                quantity=str(Q),
                entry_fee="0.05",
                exit_fee="0.05",
                gross_pnl="-2.0",
                net_pnl="-2.1",
                r_multiple="-0.2",
                holding_time_ms=60,
                outcome="SL",
            ),
            make_trade(
                2,
                direction="SHORT",
                entry_level="60500",
                stop_level="60550",
                take_profit_level="60400",
                entry_price="60500",
                exit_price="60200",
                quantity=str(Q),
                entry_fee="0.05",
                exit_fee="0.05",
                gross_pnl="30.0",
                net_pnl="29.9",
                r_multiple="3.0",
                holding_time_ms=90,
                outcome="TP",
            ),
            make_trade(3, direction="LONG", outcome=None),  # open at end
        ]
    result = build_result(
        trades=tuple(trades),
        equity_curve=tuple(equity),
        final_balance=Decimal(str(final_balance)) if final_balance is not None else equity[-1],
        initial_balance=initial_balance,
        trades_completed=sum(t.outcome in ("TP", "SL") for t in trades),
    )
    return BacktestStatistics.compute(backtest_config(initial_balance=initial_balance), result)


def test_completed_trade_counts_and_pnl():
    stats = _build_stats()
    assert stats.trades_completed == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.gross_profit == Decimal("40.1")
    assert stats.gross_loss == Decimal("2.0")
    assert stats.gross_pnl == Decimal("38.1")
    assert stats.net_pnl == Decimal("37.8")
    assert stats.total_fees == Decimal("0.3")
    assert stats.total_funding == Decimal("0")
    assert stats.final_balance == Decimal("1037.8")


def test_rates_and_averages():
    stats = _build_stats()
    assert float(stats.win_rate) == pytest.approx(66.666666, abs=1e-5)
    assert float(stats.long_win_rate) == pytest.approx(100.0)
    assert float(stats.short_win_rate) == pytest.approx(50.0)
    assert float(stats.profit_factor) == pytest.approx(20.05)
    assert float(stats.expectancy) == pytest.approx(12.6)
    assert float(stats.average_pnl) == pytest.approx(12.6)
    assert float(stats.average_win) == pytest.approx(20.05)
    assert float(stats.average_loss) == pytest.approx(2.0)
    assert float(stats.average_r) == pytest.approx(1.2666667, abs=1e-5)
    assert stats.average_holding_time_ms == 90  # (120+60+90)//3
    assert stats.max_holding_time_ms == 120


def test_direction_breakdown():
    stats = _build_stats()
    assert stats.long_wins == 1
    assert stats.long_losses == 0
    assert stats.short_wins == 1
    assert stats.short_losses == 1
    assert stats.long_trades == 1  # only completed LONG counts
    assert stats.short_trades == 2
    assert stats.long_net_pnl == Decimal("10.0")
    assert stats.short_net_pnl == Decimal("27.8")


def test_drawdown_return_and_streaks():
    stats = _build_stats()
    assert stats.peak_balance == Decimal("1037.8")
    assert stats.max_drawdown == Decimal("2.1")
    assert float(stats.max_drawdown_pct) == pytest.approx(2.1 / 10.10, rel=1e-6)
    assert float(stats.total_return_pct) == pytest.approx(3.78)
    assert stats.consecutive_wins == 1  # [win, loss, win]
    assert stats.consecutive_losses == 1


def test_slippage_informational_cost():
    stats = _build_stats()
    # Level vs executed gross, per outcome:
    #   t1 LONG TP: (60100-60000)*0.1=10.0  vs 10.1              -> 0.1
    #   t2 SHORT SL: (60500-60550)*0.1=-5.0 vs -2.0              -> 3.0
    #   t3 SHORT TP: (60500-60400)*0.1=10.0 vs 30.0              -> 20.0
    assert float(stats.slippage_cost) == pytest.approx(23.1)
    assert stats.net_pnl == Decimal("37.8")  # slippage is NOT double-counted


def test_profit_factor_none_when_no_losses():
    trades = [
        make_trade(
            0,
            direction="LONG",
            entry_level="60000",
            take_profit_level="60100",
            entry_price="60000",
            exit_price="60100",
            quantity=str(Q),
            entry_fee="0.05",
            exit_fee="0.05",
            gross_pnl="10.0",
            net_pnl="9.9",
            r_multiple="1.0",
            holding_time_ms=60,
            outcome="TP",
        )
    ]
    stats = _build_stats(trades=trades, equity=(Decimal("1000"), Decimal("1009.9")))
    assert stats.wins == 1
    assert stats.losses == 0
    assert float(stats.win_rate) == pytest.approx(100.0)
    assert stats.profit_factor is None
    assert stats.average_loss is None
    assert stats.total_fees == Decimal("0.1")


def test_no_completed_trades():
    stats = _build_stats(
        trades=[make_trade(0, direction="LONG", outcome=None)],
        equity=(Decimal("1000"),),
    )
    assert stats.trades_completed == 0
    assert stats.wins == 0
    assert stats.losses == 0
    assert stats.gross_pnl == Decimal("0")
    assert stats.net_pnl == Decimal("0")
    assert stats.profit_factor is None
    assert stats.expectancy is None
    assert stats.average_pnl is None
    assert stats.average_r is None
    assert stats.win_rate is None
    assert stats.max_drawdown == Decimal("0")


def test_missing_pnl_raises():
    trades = [
        make_trade(0, direction="LONG", outcome="TP", gross_pnl=None, net_pnl=None)
    ]
    with pytest.raises(BacktestMetricError):
        _build_stats(trades=trades)


def test_empty_equity_raises():
    result = build_result(equity_curve=())
    with pytest.raises(BacktestMetricError):
        BacktestStatistics.compute(backtest_config(), result)


def test_requires_config_and_result():
    with pytest.raises(BacktestMetricError):
        BacktestStatistics.compute(None, None)


def test_to_dict_from_dict_round_trip():
    stats = _build_stats()
    rebuilt = BacktestStatistics.from_dict(stats.to_dict())
    assert rebuilt.to_dict() == stats.to_dict()
    assert rebuilt.win_rate == stats.win_rate
    assert rebuilt.net_pnl == Decimal("37.8")


def test_to_dict_is_json_safe():
    payload = _build_stats().to_dict()
    assert isinstance(payload["net_pnl"], str)
    assert payload["gross_loss"] == "2.0"
    assert payload["profit_factor"] == "20.05"
    assert isinstance(payload["eligible"], int)
