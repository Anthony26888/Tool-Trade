"""Unit tests for the TP/SL outcome preview (demo/scenario.py).

The preview must mirror the exact formulas ``DemoExecutor.close_position``
persists (gross PnL, entry+exit fee, net PnL, balance update, peak update,
ROI against the margin), so the dashboard numbers always match the ledger.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import pytest

from demo.position import DemoAccountRecord, DemoPosition
from demo.scenario import tpsl_outcomes, unrealized_pnl


def _account(balance: float = 1000.0, peak: float = 1000.0) -> DemoAccountRecord:
    return DemoAccountRecord(
        id=1,
        name="demo",
        initial_balance=Decimal("1000"),
        balance=Decimal(str(balance)),
        equity=Decimal(str(balance)),
        margin_per_trade=Decimal("10"),
        leverage=10,
        risk_percent=Decimal("1"),
        fee_rate=Decimal("0.001"),
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        peak_equity=Decimal(str(peak)),
    )


def _position(side: str, entry: str, tp: str, sl: str) -> DemoPosition:
    return DemoPosition(
        id=1,
        account_id=1,
        signal_id=1,
        symbol="BTCUSDT",
        side=side,
        entry_price=Decimal(entry),
        quantity=Decimal("1"),
        position_size=Decimal("100"),
        margin=Decimal("10"),
        leverage=10,
        stop_loss=Decimal(sl),
        take_profit=Decimal(tp),
        unrealized_pnl=None,
        status="OPEN",
        opened_at="2026-01-01T00:00:00Z",
        closed_at=None,
    )


@pytest.mark.unit
class TestTpslOutcomes(unittest.TestCase):
    def setUp(self) -> None:
        self.account = _account()
        self.long = _position("LONG", "100.0", "110.0", "90.0")
        self.short = _position("SHORT", "100.0", "90.0", "110.0")

    def test_long_tp_is_a_win(self):
        out = tpsl_outcomes(self.account, self.long)
        tp = out["take_profit"]
        self.assertEqual(tp["exit_price"], Decimal("110.0"))
        # gross=(110-100)*1=10 ; fee=100*0.001 + 110*1*0.001=0.21
        self.assertEqual(tp["gross_pnl"], Decimal("10"))
        self.assertEqual(tp["fee"], Decimal("0.21"))
        self.assertEqual(tp["net_pnl"], Decimal("9.79"))
        self.assertEqual(tp["result"], "WIN")
        self.assertEqual(tp["pnl_percent"], Decimal("97.9"))
        self.assertEqual(tp["projected_balance"], Decimal("1009.79"))
        self.assertEqual(tp["projected_peak"], Decimal("1009.79"))

    def test_long_sl_is_a_loss_and_keeps_peak(self):
        out = tpsl_outcomes(self.account, self.long)
        sl = out["stop_loss"]
        self.assertEqual(sl["exit_price"], Decimal("90.0"))
        # gross=(90-100)*1=-10 ; fee=100*0.001 + 90*1*0.001=0.19
        self.assertEqual(sl["gross_pnl"], Decimal("-10"))
        self.assertEqual(sl["fee"], Decimal("0.19"))
        self.assertEqual(sl["net_pnl"], Decimal("-10.19"))
        self.assertEqual(sl["result"], "LOSS")
        self.assertEqual(sl["pnl_percent"], Decimal("-101.9"))
        self.assertEqual(sl["projected_balance"], Decimal("989.81"))
        # peak equity never drops below the current peak.
        self.assertEqual(sl["projected_peak"], Decimal("1000"))

    def test_short_tp_is_a_win(self):
        out = tpsl_outcomes(self.account, self.short)
        tp = out["take_profit"]
        self.assertEqual(tp["gross_pnl"], Decimal("10"))
        # fee = 100*0.001 + 90*1*0.001 (exit at TP 90) = 0.19
        self.assertEqual(tp["fee"], Decimal("0.19"))
        self.assertEqual(tp["net_pnl"], Decimal("9.81"))
        self.assertEqual(tp["result"], "WIN")
        self.assertEqual(tp["projected_balance"], Decimal("1009.81"))

    def test_short_sl_is_a_loss(self):
        out = tpsl_outcomes(self.account, self.short)
        sl = out["stop_loss"]
        self.assertEqual(sl["gross_pnl"], Decimal("-10"))
        # fee = 100*0.001 + 110*1*0.001 (exit at SL 110) = 0.21
        self.assertEqual(sl["fee"], Decimal("0.21"))
        self.assertEqual(sl["net_pnl"], Decimal("-10.21"))
        self.assertEqual(sl["result"], "LOSS")
        self.assertEqual(sl["projected_balance"], Decimal("989.79"))

    def test_secondary_peak_extends_to_new_high(self):
        account = _account(balance=990.0, peak=1000.0)
        out = tpsl_outcomes(account, self.long)
        tp = out["take_profit"]
        # balance 990 + 9.79 = 999.79 -> below peak 1000.
        self.assertEqual(tp["projected_balance"], Decimal("999.79"))
        self.assertEqual(tp["projected_peak"], Decimal("1000"))
        account2 = _account(balance=1010.0, peak=1005.0)
        out2 = tpsl_outcomes(account2, self.long)
        tp2 = out2["take_profit"]
        # balance 1010 + 9.79 = 1019.79 -> new peak.
        self.assertEqual(tp2["projected_peak"], Decimal("1019.79"))

    def test_zero_margin_produces_zero_pnl_percent(self):
        position = _position("LONG", "100.0", "110.0", "90.0")
        position = position.__class__(
            **{**position.__dict__, "margin": Decimal("0")}
        )
        out = tpsl_outcomes(self.account, position)
        self.assertEqual(out["take_profit"]["pnl_percent"], Decimal("0"))


@pytest.mark.unit
class TestUnrealizedPnl(unittest.TestCase):
    """Live PnL row: gross $ against entry plus ROI on margin (with leverage)."""

    def setUp(self) -> None:
        self.long = _position("LONG", "100.0", "110.0", "90.0")
        self.short = _position("SHORT", "100.0", "90.0", "110.0")

    def test_long_profit_at_higher_price(self):
        out = unrealized_pnl(self.long, Decimal("105"))
        self.assertEqual(out["gross_pnl"], Decimal("5"))
        # margin 10 -> 5/10*100 = 50%
        self.assertEqual(out["pnl_percent"], Decimal("50"))
        self.assertEqual(out["result"], "WIN")

    def test_long_loss_at_lower_price(self):
        out = unrealized_pnl(self.long, Decimal("95"))
        self.assertEqual(out["gross_pnl"], Decimal("-5"))
        self.assertEqual(out["pnl_percent"], Decimal("-50"))
        self.assertEqual(out["result"], "LOSS")

    def test_short_profit_at_lower_price(self):
        out = unrealized_pnl(self.short, Decimal("95"))
        self.assertEqual(out["gross_pnl"], Decimal("5"))
        self.assertEqual(out["pnl_percent"], Decimal("50"))
        self.assertEqual(out["result"], "WIN")

    def test_short_loss_at_higher_price(self):
        out = unrealized_pnl(self.short, Decimal("105"))
        self.assertEqual(out["gross_pnl"], Decimal("-5"))
        self.assertEqual(out["pnl_percent"], Decimal("-50"))
        self.assertEqual(out["result"], "LOSS")

    def test_flat_when_price_equals_entry(self):
        out = unrealized_pnl(self.long, Decimal("100"))
        self.assertEqual(out["gross_pnl"], Decimal("0"))
        self.assertEqual(out["pnl_percent"], Decimal("0"))
        self.assertEqual(out["result"], "FLAT")

    def test_pnl_percent_reflects_leverage(self):
        # qty=1, margin=10 -> position_size 100 = 10x. A 1% price move (1 USDT)
        # yields 10% ROI on margin.
        out = unrealized_pnl(self.long, Decimal("101"))
        self.assertEqual(out["gross_pnl"], Decimal("1"))
        self.assertEqual(out["pnl_percent"], Decimal("10"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
