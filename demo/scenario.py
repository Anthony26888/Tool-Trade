"""Demo position outcome preview for OPEN positions.

This module computes, from the persisted :class:`DemoAccountRecord` and the
opened :class:`DemoPosition`, the exact result the executor would record for
each possible exit level (TP or SL). It deliberately reuses the pure Phase 8
formulas (:mod:`demo.account`) relied on by ``DemoExecutor.close_position``,
so the preview numbers always match the ledger that is written when the price
actually reaches a level.

The preview is read-only display data: it never writes a trade and never
affects trading state, balances, or TP/SL monitoring.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .account import (
    balance_after_close,
    gross_pnl,
    net_pnl,
    total_fee,
    update_peak_equity,
)
from .position import DemoAccountRecord, DemoPosition

WIN = "WIN"
LOSS = "LOSS"


def _outcome(
    account: DemoAccountRecord,
    position: DemoPosition,
    exit_price: Decimal,
) -> dict[str, Any]:
    gross = gross_pnl(position.side, position.entry_price, exit_price, position.quantity)
    fees = total_fee(position.position_size, exit_price, position.quantity, account.fee_rate)
    net = net_pnl(
        position.side,
        position.entry_price,
        exit_price,
        position.quantity,
        position.position_size,
        account.fee_rate,
    )
    next_balance = balance_after_close(
        account.balance,
        position.side,
        position.entry_price,
        exit_price,
        position.quantity,
        position.position_size,
        account.fee_rate,
    )
    next_peak = update_peak_equity(account.peak_equity, next_balance)
    pnl_percent = (net / position.margin) * 100 if position.margin else Decimal("0")
    return {
        "exit_price": exit_price,
        "gross_pnl": gross,
        "fee": fees,
        "net_pnl": net,
        "pnl_percent": pnl_percent,
        "result": WIN if net > 0 else LOSS,
        "projected_balance": next_balance,
        "projected_peak": next_peak,
    }


def tpsl_outcomes(
    account: DemoAccountRecord,
    position: DemoPosition,
) -> dict[str, Any]:
    """Projected TP and SL results for an OPEN demo position.

    Returns a dict with the ``take_profit`` / ``stop_loss`` keys, each holding
    the exact numbers ``DemoExecutor.close_position`` would persist if that
    exit level were hit:
    ``exit_price``, ``gross_pnl``, ``fee``, ``net_pnl``, ``pnl_percent``
    (ROI against the position margin), ``result`` and the projected
    ``balance``/``peak``.
    """
    return {
        "take_profit": _outcome(account, position, position.take_profit),
        "stop_loss": _outcome(account, position, position.stop_loss),
    }


def unrealized_pnl(position: DemoPosition, current_price: Any) -> dict[str, Any]:
    """Live PnL a position would realize if closed at ``current_price``.

    Display-only data from the live market price: ``gross_pnl`` in USDT and
    ``pnl_percent`` — ROI against the position margin, which already reflects
    the configured leverage (quantity = margin x leverage / entry). No fees
    are charged since nothing has been closed yet. Always matches the pure
    ``demo.account.gross_pnl`` formula used by the executor.
    """
    gross = gross_pnl(position.side, position.entry_price, current_price, position.quantity)
    pnl_percent = (gross / position.margin) * 100 if position.margin else Decimal("0")
    result = WIN if gross > 0 else (LOSS if gross < 0 else "FLAT")
    return {
        "gross_pnl": gross,
        "pnl_percent": pnl_percent,
        "result": result,
    }


__all__ = ["tpsl_outcomes", "unrealized_pnl"]
