"""Demo execution — the Phase 14 integration between signals and the DEMO
account (never a real order).

``DemoExecutor`` is driven by signal lifecycle events:

- a signal promoted to ``OPEN`` (entry touched)         -> ``open_position``
- a signal closed as ``TP_HIT`` / ``SL_HIT``            -> ``close_position``

It reuses the Phase 8 pure formulas (:mod:`demo.account`) for every number:
position notional = margin x leverage, quantity = notional / entry,
net PnL = gross - entry fee - exit fee, balance = balance + net PnL, peak
equity = max(peak, current). No formula is re-implemented here.

Idempotency and restart safety
------------------------------
- All reads/writes go through :class:`DemoRepository`; a second ``open_position``
  for the same signal returns the existing position and a second
  ``close_position`` returns the existing trade. The UNIQUE indexes on
  ``demo_positions(signal_id)`` and ``demo_trades(signal_id)`` make a double
  open/close structurally impossible.
- ``reconcile`` repairs the crash windows that can exist between the signal
  transition and the demo writes: an OPEN signal missing its position gets one,
  and any position whose signal reached TP/SL without a trade gets closed.
  Reconcile never invents history — a terminal signal without a position is
  never turned into a trade.

The executor never places a Binance order, never uses a trading API key, and
simulates execution only from the signal's closed-candle levels.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from database.database import DemoRepository
from database.models import (
    STATUS_OPEN,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    Signal,
)

from .account import (
    DemoConfig,
    balance_after_close,
    gross_pnl,
    net_pnl,
    position_quantity,
    position_size,
    total_fee,
    update_peak_equity,
    validate_config,
)
from .position import DemoAccountRecord, DemoPosition, DemoTrade
from .statistics import DemoStatistics, demo_statistics

logger = logging.getLogger(__name__)

CLOSE_REASON_TP = "TP"
CLOSE_REASON_SL = "SL"


class DemoExecutor:
    """Open/close demo positions and record trades tied to signals."""

    def __init__(
        self,
        database,
        *,
        config: DemoConfig | None = None,
        account_name: str = "demo",
        repository: DemoRepository | None = None,
    ) -> None:
        self.database = database
        self.config = config if config is not None else DemoConfig()
        validate_config(self.config)
        self.account_name = account_name
        self.repository = repository if repository is not None else DemoRepository(database)

    # -- Account --------------------------------------------------------------

    def ensure_account(self) -> DemoAccountRecord:
        """Return the demo account, creating it at the configured values once.

        Creating NEVER resets an existing account row, so a restart preserves
        the balance/equity/peak (AGENTS.md sections 17 and 23).
        """
        row = self.repository.ensure_account(
            name=self.account_name,
            initial_balance=self.config.initial_balance,
            margin_per_trade=self.config.margin_per_trade,
            leverage=self.config.leverage,
            risk_percent=self.config.risk_percent,
            fee_rate=self.config.fee_rate,
        )
        return DemoAccountRecord.from_row(row)

    def account(self) -> DemoAccountRecord | None:
        """The current account, or None before the demo account is created."""
        row = self.repository.get_account(self.account_name)
        return DemoAccountRecord.from_row(row) if row is not None else None

    # -- Position lifecycle ----------------------------------------------------

    def open_position(self, signal: Signal | None) -> DemoPosition | None:
        """Persist the OPEN demo position for a signal that reached OPEN.

        Idempotent: an existing position for the signal is returned unchanged.
        Returns ``None`` when the signal is not OPEN (nothing to open).
        """
        if signal is None or signal.status != STATUS_OPEN:
            return None
        existing = self.repository.get_position_for_signal(signal.id)
        if existing is not None:
            return DemoPosition.from_row(existing)
        account = self.ensure_account()
        notional = position_size(account.margin_per_trade, account.leverage)
        quantity = position_quantity(notional, signal.entry)
        row = self.repository.create_position(
            account_id=account.id,
            signal_id=signal.id,
            symbol=signal.symbol,
            side=signal.direction,
            entry_price=signal.entry,
            quantity=quantity,
            position_size=notional,
            margin=account.margin_per_trade,
            leverage=account.leverage,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        logger.info("[Demo] position opened for signal %s (%s %s)", signal.id, signal.direction, signal.symbol)
        return DemoPosition.from_row(row)

    def close_position(
        self, signal: Signal | None, *, closed_at: str | None = None
    ) -> DemoTrade | None:
        """Record the closed demo trade + balance update for a TP/SL signal.

        Idempotent per signal (returns the existing trade on a repeat). Returns
        ``None`` when the signal is not terminal or has no opened position — a
        trade is never invented without a position.
        """
        if signal is None or signal.status not in (STATUS_TP_HIT, STATUS_SL_HIT):
            return None
        existing_trade = self.repository.get_trade_for_signal(signal.id)
        if existing_trade is not None:
            return DemoTrade.from_row(existing_trade)
        position_row = self.repository.get_position_for_signal(signal.id)
        if position_row is None:
            # Crash window (position insert lost) or a pre-Phase-14 terminal
            # signal: do not fabricate a trade.
            logger.warning(
                "[Demo] signal %s closed but has no demo position; no trade recorded",
                signal.id,
            )
            return None

        position = DemoPosition.from_row(position_row)
        account = self.ensure_account()
        exit_price = (
            signal.close_price
            if signal.close_price is not None
            else position.take_profit if signal.status == STATUS_TP_HIT else position.stop_loss
        )
        fee_rate = account.fee_rate
        gross = gross_pnl(
            signal.direction, position.entry_price, exit_price, position.quantity
        )
        fees = total_fee(position.position_size, exit_price, position.quantity, fee_rate)
        net = net_pnl(
            signal.direction,
            position.entry_price,
            exit_price,
            position.quantity,
            position.position_size,
            fee_rate,
        )
        next_balance = balance_after_close(
            account.balance,
            signal.direction,
            position.entry_price,
            exit_price,
            position.quantity,
            position.position_size,
            fee_rate,
        )
        next_peak = update_peak_equity(account.peak_equity, next_balance)
        result = "WIN" if net > 0 else "LOSS"
        pnl_percent = (net / position.margin) * 100 if position.margin else Decimal("0")

        row = self.repository.record_trade(
            position_id=position.id,
            signal_id=signal.id,
            account_id=account.id,
            side=signal.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            margin=position.margin,
            position_size=position.position_size,
            leverage=position.leverage,
            gross_pnl=gross,
            fee=fees,
            net_pnl=net,
            pnl_percent=pnl_percent,
            result=result,
            closed_at=closed_at,
            next_balance=next_balance,
            next_equity=next_balance,
            next_peak_equity=next_peak,
        )
        logger.info(
            "[Demo] trade for signal %s recorded: %s net_pnl=%s balance=%s",
            signal.id, result, net, next_balance,
        )
        return DemoTrade.from_row(row)

    # -- Restart reconciliation -------------------------------------------------

    def reconcile(self, active_signal: Signal | None = None) -> dict[str, int]:
        """Repair demo state to match the signal ledger after a restart.

        Returns ``{"positions_recovered": n, "trades_recovered": m}``. This is
        the ONLY demo write path used on startup; it never notifies Telegram and
        never re-sends anything.
        """
        self.ensure_account()
        recovered_positions = 0
        recovered_trades = 0

        if active_signal is not None and active_signal.status == STATUS_OPEN:
            position = self.open_position(active_signal)
            if position is not None:
                recovered_positions = recovered_positions + 1

        for position_row in self.repository.list_positions():
            position = DemoPosition.from_row(position_row)
            try:
                signal = _signal_for(self, position.signal_id)
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.warning("[Demo] reconcile skipped signal %s: %s", position.signal_id, exc)
                continue
            if signal is None or signal.status not in (STATUS_TP_HIT, STATUS_SL_HIT):
                continue
            if self.repository.get_trade_for_signal(position.signal_id) is not None:
                continue
            if self.close_position(signal) is not None:
                recovered_trades = recovered_trades + 1

        if recovered_positions or recovered_trades:
            logger.info(
                "[Demo] reconcile recovered %d position(s) and %d trade(s)",
                recovered_positions, recovered_trades,
            )
        return {"positions_recovered": recovered_positions, "trades_recovered": recovered_trades}

    # -- Statistics -------------------------------------------------------------

    def statistics(self) -> DemoStatistics | None:
        """Win/loss statistics and drawdown from the persisted ledger."""
        account = self.account()
        if account is None:
            return None
        trades = [
            DemoTrade.from_row(row) for row in self.repository.list_trades(account.id)
        ]
        return demo_statistics(account, trades)


def _signal_for(executor: DemoExecutor, signal_id: int) -> Signal | None:
    """Fetch a signal from the executor's database (fresh connection)."""
    from database.database import SignalRepository

    return SignalRepository(executor.database).get_signal(signal_id)


__all__ = ["DemoExecutor", "CLOSE_REASON_TP", "CLOSE_REASON_SL"]
