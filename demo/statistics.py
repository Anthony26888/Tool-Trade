"""Demo statistics derived purely from the Persisted Ledger (Phase 14).

All values are ``Decimal`` and computed via the Phase 8 formula functions and
the ledger of closed ``demo_trades`` — the database, never memory, is the
source of truth. Max drawdown is computed from the balance path
(``initial_balance`` followed by each trade's net PnL), so it never uses
invented intraday prices.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from .account import current_drawdown
from .position import DemoAccountRecord, DemoTrade

ZERO = Decimal("0")


@dataclass(frozen=True)
class DemoStatistics:
    """Win/loss statistics over a completed demo trade ledger."""

    total_trades: int
    wins: int
    losses: int
    win_rate: Decimal | None
    net_pnl: Decimal
    gross_pnl: Decimal
    total_fees: Decimal
    profit_factor: Decimal | None
    average_r: Decimal | None
    max_drawdown: Decimal
    final_balance: Decimal

    def as_dict(self) -> dict:
        """Plain-dict view for CLI/status rendering."""
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "net_pnl": self.net_pnl,
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "profit_factor": self.profit_factor,
            "average_r": self.average_r,
            "max_drawdown": self.max_drawdown,
            "final_balance": self.final_balance,
        }


def _profit_factor(wins: Decimal, losses: Decimal) -> Decimal | None:
    """Gross win / gross loss; None when there is no loss to compare."""
    if losses <= 0:
        # No losing R: an infinite profit factor is reported as
        # None so the CLI can render it meaningfully ("no losses").
        return None
    if wins <= 0:
        return ZERO
    return wins / losses


def _average_r(net_pnls: list[Decimal], margins: list[Decimal]) -> Decimal | None:
    """Mean net PnL in R multiples (PnL / margin per trade)."""
    if not net_pnls:
        return None
    return sum(net_pnls, ZERO) / Decimal(len(net_pnls))


def _max_drawdown_from_path(path: list[Decimal]) -> Decimal:
    """Maximum peak-to-trough drawdown across a running-balance path."""
    if not path:
        return ZERO
    peak = path[0]
    worst = ZERO
    for balance in path:
        if balance > peak:
            peak = balance
        drawdown = current_drawdown(peak, balance)
        if drawdown > worst:
            worst = drawdown
    return worst


def demo_statistics(
    account: DemoAccountRecord,
    trades: Iterable[DemoTrade],
    *,
    initial_balance: Decimal | None = None,
) -> DemoStatistics:
    """Compute statistics for one account over its closed trades.

    ``initial_balance`` defaults to the account's stored initial balance.
    Max drawdown is measured on the balance path ``initial_balance + net_pnl``
    (fees are already inside net PnL), so it never guesses intrabar equity.
    """
    trade_list = list(trades)
    start = initial_balance if initial_balance is not None else account.initial_balance

    wins = [t.net_pnl for t in trade_list if t.result == "WIN"]
    losses = [t.net_pnl for t in trade_list if t.result == "LOSS"]
    gross = sum((t.gross_pnl for t in trade_list), ZERO)
    fees = sum((t.fee for t in trade_list), ZERO)
    net = sum((t.net_pnl for t in trade_list), ZERO)

    path = [start]
    running = start
    for trade in trade_list:
        running = running + trade.net_pnl
        path.append(running)

    return DemoStatistics(
        total_trades=len(trade_list),
        wins=len(wins),
        losses=len(losses),
        win_rate=(Decimal(len(wins)) / Decimal(len(trade_list))) if trade_list else None,
        net_pnl=net,
        gross_pnl=gross,
        total_fees=fees,
        profit_factor=_profit_factor(sum(wins, ZERO), abs(sum(losses, ZERO))),
        average_r=_average_r(
            [t.net_pnl for t in trade_list], [t.margin for t in trade_list]
        ),
        max_drawdown=_max_drawdown_from_path(path),
        final_balance=running,
    )


__all__ = ["DemoStatistics", "demo_statistics"]
