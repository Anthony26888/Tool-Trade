"""Phase 14 unit tests for the DEMO account integration (demo/ package).

Covers the persistence repositories plus the DEMO executor: opening/closing
positions from signal lifecycles, Phase 8 formula reuse, idempotency, atomic
trade recording, restart reconciliation, and ledger-derived statistics.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import pytest

from database.database import (
    DemoDuplicateError,
    DemoRepository,
)
from database.models import STATUS_OPEN, STATUS_SL_HIT, STATUS_TP_HIT
from demo.account import (
    DemoConfig,
    balance_after_close,
    gross_pnl,
    net_pnl,
    position_quantity,
    position_size,
    total_fee,
    update_peak_equity,
)
from demo.executor import DemoExecutor
from demo.position import DemoAccountRecord
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis

MARGIN = Decimal("50")
LEVERAGE = 10
FEE = Decimal("0.0004")


def demo_config(**overrides) -> DemoConfig:
    defaults = {"margin_per_trade": MARGIN, "leverage": LEVERAGE, "fee_rate": FEE}
    defaults.update(overrides)
    return DemoConfig(**defaults)


class DemoExecutorTestCase(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.repo = DemoRepository(self.harness.db)
        self.executor = DemoExecutor(self.harness.db, config=demo_config())

    def _create_signal(self, decision="LONG", **overrides):
        from signal_engine import GuardrailConfig

        # Guardrails relaxed on purpose: demo tests verify PnL math on
        # symmetric levels, not validator policy.
        relaxed = GuardrailConfig(
            min_confidence=0,
            min_risk_reward=Decimal("0"),
            fee_rate=Decimal("0"),
        )
        result = self.harness.engine.process(
            make_analysis(decision, **overrides), guardrails=relaxed
        )
        self.assertEqual(result.outcome.value, "CREATED")
        return result.signal

    def _open(self, signal):
        return self.harness.repository.transition_signal(signal.id, STATUS_OPEN)

    def _close(self, signal, status, *, close_price):
        return self.harness.repository.transition_signal(
            signal.id,
            status,
            close_price=close_price,
            close_reason="TP" if status == STATUS_TP_HIT else "SL",
        )

    def _open_signal(self, decision="LONG", **overrides):
        """A persisted PENDING_ENTRY signal promoted to OPEN."""
        signal = self._create_signal(decision, **overrides)
        return self._open(signal)

    def _close_signal(self, signal, status=STATUS_SL_HIT):
        close_price = signal.stop_loss if status == STATUS_SL_HIT else signal.take_profit
        return self._close(signal, status, close_price=close_price)


# ── Account persistence ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestAccountPersistence(DemoExecutorTestCase):
    def test_ensure_account_creates_once_and_never_resets(self):
        account = self.executor.ensure_account()
        self.assertIsInstance(account, DemoAccountRecord)
        self.assertEqual(account.balance, Decimal("1000"))
        self.assertEqual(account.initial_balance, Decimal("1000"))
        self.assertEqual(account.peak_equity, Decimal("1000"))
        self.assertEqual(account.margin_per_trade, MARGIN)
        self.assertEqual(account.leverage, LEVERAGE)
        self.assertEqual(account.fee_rate, FEE)

        again = self.executor.ensure_account()
        self.assertEqual(again.id, account.id)
        self.assertEqual(self.repo.count_trades(), 0)

    def test_ensure_account_preserves_balance_on_restart(self):
        account = self.executor.ensure_account()
        self.repo.update_balance(
            account.id,
            balance="876.5",
            equity="876.5",
            peak_equity=account.peak_equity,
        )
        restarted = DemoExecutor(
            self.harness.db, config=demo_config(margin_per_trade=7, leverage=3)
        ).ensure_account()
        self.assertEqual(restarted.balance, Decimal("876.5"))
        self.assertNotEqual(restarted.margin_per_trade, Decimal("7"))

    def test_reset_account_full_fresh_start(self):
        signal = self._create_signal("LONG")
        signal_id = signal.id
        signal = self._open(signal)
        self.executor.open_position(signal)
        signal = self._close_signal(signal, STATUS_TP_HIT)
        self.executor.close_position(signal)
        account = self.executor.ensure_account()
        self.assertEqual(self.repo.count_trades(), 1)
        self.assertEqual(len(self.repo.list_positions()), 1)
        self.assertNotEqual(account.balance, Decimal("1000"))

        row = self.repo.reset_account(account.id, initial_balance="2000")
        self.assertEqual(row["initial_balance"], "2000")
        self.assertEqual(row["balance"], "2000")
        self.assertEqual(row["equity"], "2000")
        self.assertEqual(row["peak_equity"], "2000")
        self.assertEqual(self.repo.count_trades(), 0)
        self.assertEqual(self.repo.list_positions(), [])
        # the immutable signal ledger survives the demo reset
        restored = self.harness.repository.get_signal(signal_id)
        self.assertEqual(restored.id, signal_id)


# ── Position opening ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPositionOpen(DemoExecutorTestCase):
    def test_open_position_requires_an_open_signal(self):
        pending = self._create_signal("LONG")
        self.assertIsNone(self.executor.open_position(pending))
        self.assertEqual(self.repo.list_positions(), [])

    def test_open_long_position_uses_phase8_formulas(self):
        signal = self._open(self._create_signal("LONG"))
        position = self.executor.open_position(signal)
        self.assertIsNotNone(position)
        self.assertEqual(position.side, "LONG")
        account = self.executor.account()
        self.assertEqual(position.position_size, position_size(MARGIN, LEVERAGE))
        self.assertEqual(position.quantity, position_quantity(position.position_size, signal.entry))
        self.assertEqual(position.entry_price, signal.entry)
        self.assertEqual(position.stop_loss, signal.stop_loss)
        self.assertEqual(position.take_profit, signal.take_profit)
        self.assertEqual(position.margin, account.margin_per_trade)
        self.assertEqual(position.account_id, account.id)
        self.assertEqual(position.signal_id, signal.id)

    def test_open_position_idempotent(self):
        signal = self._open(self._create_signal("LONG"))
        first = self.executor.open_position(signal)
        second = self.executor.open_position(signal)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.repo.list_positions(status="OPEN")), 1)

    def test_open_applies_risk_cap_on_wide_stop(self):
        # SL 5000 away: risk qty 10/5000 = 0.002 beats margin qty 500/61000.
        signal = self._open(
            self._create_signal(
                "LONG",
                entry_price=61000.0,
                stop_loss=56000.0,
                take_profit=67000.0,
            )
        )
        position = self.executor.open_position(signal)
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, Decimal("0.002"))
        self.assertEqual(position.position_size, Decimal("0.002") * Decimal("61000.0"))

    def test_open_refused_when_balance_below_margin(self):
        account = self.executor.ensure_account()
        self.repo.update_balance(account.id, balance="10", equity="10", peak_equity="10")
        signal = self._open(self._create_signal("LONG"))
        self.assertIsNone(self.executor.open_position(signal))
        self.assertEqual(self.repo.list_positions(), [])

    def test_open_applies_slippage_when_configured(self):
        from demo.executor import DemoExecutor

        executor = DemoExecutor(
            self.harness.db, config=demo_config(slippage_bps=Decimal("2"))
        )
        signal = self._open(self._create_signal("LONG"))
        position = executor.open_position(signal)
        self.assertIsNotNone(position)
        # LONG pays up 2 bps on entry: 61000 * 1.0002.
        self.assertEqual(position.entry_price, Decimal("61012.2"))
        self.assertEqual(
            position.quantity,
            Decimal("500") / Decimal("61012.2"),
        )


# ── Position close / trade recording ─────────────────────────────────────────


@pytest.mark.unit
class TestPositionClose(DemoExecutorTestCase):
    def test_close_requires_terminal_signal(self):
        signal = self._open(self._create_signal("LONG"))
        self.assertIsNone(self.executor.close_position(signal))
        self.assertEqual(self.repo.count_trades(), 0)

    def test_close_without_position_invents_nothing(self):
        signal = self._open_signal("LONG")
        signal = self._close_signal(signal)
        self.assertIsNone(self.executor.close_position(signal))
        self.assertEqual(self.repo.count_trades(), 0)

    def test_close_records_trade_with_phase8_pnl(self):
        signal = self._open_signal("LONG")
        self.executor.open_position(signal)
        signal = self._close_signal(signal, STATUS_TP_HIT)
        trade = self.executor.close_position(signal)
        self.assertIsNotNone(trade)
        account = self.executor.account()
        notional = position_size(account.margin_per_trade, account.leverage)
        qty = position_quantity(notional, signal.entry)
        fees = total_fee(notional, signal.take_profit, qty, account.fee_rate)
        expected_gross = gross_pnl("LONG", signal.entry, signal.take_profit, qty)
        expected_net = net_pnl(
            "LONG", signal.entry, signal.take_profit, qty, notional, account.fee_rate
        )
        self.assertEqual(trade.gross_pnl, expected_gross)
        self.assertEqual(trade.fee, fees)
        self.assertEqual(trade.net_pnl, expected_net)
        self.assertEqual(trade.entry_price, signal.entry)
        self.assertEqual(trade.exit_price, signal.take_profit)
        self.assertEqual(trade.result, "WIN" if expected_net > 0 else "LOSS")
        self.assertEqual(
            trade.pnl_percent, (expected_net / account.margin_per_trade) * 100
        )

    def test_close_updates_balance_with_net_pnl(self):
        signal = self._open_signal("LONG")
        self.executor.open_position(signal)
        starting = self.executor.account().balance
        signal = self._close_signal(signal, STATUS_TP_HIT)
        trade = self.executor.close_position(signal)
        expected = balance_after_close(
            starting,
            "LONG",
            signal.entry,
            signal.take_profit,
            trade.quantity,
            trade.position_size,
            self.executor.account().fee_rate,
        )
        self.assertEqual(self.executor.account().balance, expected)
        self.assertEqual(self.executor.account().equity, expected)
        self.assertEqual(
            self.executor.account().peak_equity,
            update_peak_equity(starting, expected),
        )

    def test_close_position_idempotent(self):
        signal = self._open_signal("LONG")
        self.executor.open_position(signal)
        signal = self._close_signal(signal, STATUS_TP_HIT)
        first = self.executor.close_position(signal)
        second = self.executor.close_position(signal)
        self.assertEqual(first.id, second.id)
        self.assertEqual(self.repo.count_trades(), 1)

    def test_double_close_structural_guard(self):
        signal = self._open_signal("LONG")
        position = self.executor.open_position(signal)
        signal = self._close_signal(signal, STATUS_TP_HIT)
        self.executor.close_position(signal)
        account = self.executor.account()
        with self.assertRaises(DemoDuplicateError):
            self.repo.record_trade(
                position_id=position.id,
                signal_id=signal.id,
                account_id=account.id,
                side="LONG",
                entry_price="61000",
                exit_price="64000",
                quantity="1",
                margin="50",
                position_size="500",
                leverage=10,
                gross_pnl="1",
                fee="0.4",
                net_pnl="0.6",
                pnl_percent="1.2",
                result="WIN",
            )

    def test_short_close_uses_short_pnl(self):
        signal = self._open_signal(
            "SHORT", entry_price=59000.0, stop_loss=60000.0, take_profit=58000.0
        )
        self.executor.open_position(signal)
        signal = self._close_signal(signal, STATUS_SL_HIT)
        trade = self.executor.close_position(signal)
        account = self.executor.account()
        qty = trade.quantity
        self.assertEqual(
            trade.net_pnl,
            net_pnl("SHORT", trade.entry_price, trade.exit_price, qty, trade.position_size, account.fee_rate),
        )
        self.assertEqual(trade.exit_price, Decimal("60000.0"))


# ── Restart reconciliation ───────────────────────────────────────────────────


@pytest.mark.unit
class TestReconcile(DemoExecutorTestCase):
    def test_reconcile_opens_missing_position(self):
        signal = self._open(self._create_signal("LONG"))
        self.assertEqual(self.repo.list_positions(), [])
        report = self.executor.reconcile(active_signal=signal)
        self.assertEqual(report["positions_recovered"], 1)
        self.assertEqual(len(self.repo.list_positions(status="OPEN")), 1)

    def test_reconcile_does_not_insert_for_terminal_without_position(self):
        signal = self._open_signal("LONG")
        signal = self._close_signal(signal)
        report = self.executor.reconcile(active_signal=None)
        self.assertEqual(report["trades_recovered"], 0)
        self.assertEqual(report["positions_recovered"], 0)
        self.assertEqual(self.repo.count_trades(), 0)
        self.assertEqual(self.repo.get_trade_for_signal(signal.id), None)

    def test_reconcile_closes_position_whose_signal_is_terminal(self):
        signal = self._open_signal("LONG")
        position = self.executor.open_position(signal)
        self.assertIsNotNone(position)
        signal = self._close_signal(signal)
        report = self.executor.reconcile(active_signal=None)
        self.assertEqual(report["trades_recovered"], 1)
        trade = self.repo.get_trade_for_signal(signal.id)
        self.assertIsNotNone(trade)
        closed = self.repo.list_positions(status="CLOSED")
        self.assertEqual([p["id"] for p in closed], [position.id])

    def test_reconcile_idempotent(self):
        signal = self._open(self._create_signal("LONG"))
        self.executor.reconcile(active_signal=signal)
        second = self.executor.reconcile(active_signal=signal)
        self.assertEqual(second["positions_recovered"], 1)
        self.assertEqual(len(self.repo.list_positions(status="OPEN")), 1)


@pytest.mark.unit
class TestStatistics(DemoExecutorTestCase):
    def test_statistics_from_ledger(self):
        # A TP win then an SL loss (SL still below entry in both cases).
        cases = (
            (61000.0, 60000.0, 64000.0, STATUS_TP_HIT),
            (61800.0, 61400.0, 64000.0, STATUS_SL_HIT),
        )
        for entry, sl, tp, close_status in cases:
            signal = self._open_signal("LONG", entry_price=entry, stop_loss=sl, take_profit=tp)
            self.executor.open_position(signal)
            signal = self._close_signal(signal, close_status)
            self.executor.close_position(signal)

        stats = self.executor.statistics()
        self.assertIsNotNone(stats)
        self.assertEqual(stats.total_trades, 2)
        self.assertEqual(stats.wins, 1)
        self.assertEqual(stats.losses, 1)
        self.assertEqual(stats.win_rate, Decimal("0.5"))
        # The ledger statistics are measured from the account's own final balance.
        self.assertLess(
            abs(stats.final_balance - self.executor.account().balance),
            Decimal("1e-12"),
        )
        self.assertLess(
            abs(stats.net_pnl - (self.executor.account().balance - Decimal("1000"))),
            Decimal("1e-12"),
        )
        self.assertGreater(stats.max_drawdown, Decimal("0"))
        self.assertIsNotNone(stats.profit_factor)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
