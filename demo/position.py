"""Demo domain records (Phase 14).

``sqlite3.Row`` values are converted into immutable ``Decimal``-backed
dataclasses so trading/statistics code never touches binary floats for money.
The dataclasses carry exactly the Phase 3 ``demo_*`` table columns; nothing
here computes strategy — the Phase 8 pure formulas in :mod:`demo.account` and
the Phase 14 executor/statistics do that.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .account import DIRECTION_LONG, DIRECTION_SHORT


def direction_long(side: str) -> bool:
    """True when the side is LONG."""
    return side == DIRECTION_LONG


def direction_short(side: str) -> bool:
    """True when the side is SHORT."""
    return side == DIRECTION_SHORT


def _dec(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _dec_opt(value: Any, field: str) -> Decimal | None:
    return _dec(value, field)


@dataclass(frozen=True)
class DemoAccountRecord:
    """The persisted demo account state (balances always ``Decimal``)."""

    id: int
    name: str
    initial_balance: Decimal
    balance: Decimal
    equity: Decimal
    margin_per_trade: Decimal
    leverage: int
    risk_percent: Decimal
    fee_rate: Decimal
    created_at: str
    updated_at: str | None
    peak_equity: Decimal

    @classmethod
    def from_row(cls, row) -> DemoAccountRecord:
        """Build from a ``sqlite3.Row`` (a ``peak_equity`` default of the
        initial balance keeps genuinely old rows valid)."""
        peak = _dec(row["peak_equity"], "peak_equity")
        return cls(
            id=int(row["id"]),
            name=row["name"],
            initial_balance=Decimal(str(row["initial_balance"])),
            balance=Decimal(str(row["balance"])),
            equity=Decimal(str(row["equity"])) if row["equity"] is not None else Decimal(str(row["balance"])),
            margin_per_trade=Decimal(str(row["margin_per_trade"])),
            leverage=int(row["leverage"]),
            risk_percent=Decimal(str(row["risk_percent"])),
            fee_rate=Decimal(str(row["fee_rate"])),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            peak_equity=peak if peak is not None else Decimal(str(row["balance"])),
        )


@dataclass(frozen=True)
class DemoPosition:
    """A demo position opened when a signal reaches OPEN (never a real order)."""

    id: int
    account_id: int
    signal_id: int
    symbol: str
    side: str
    entry_price: Decimal
    quantity: Decimal
    position_size: Decimal
    margin: Decimal
    leverage: int
    stop_loss: Decimal
    take_profit: Decimal
    unrealized_pnl: Decimal | None
    status: str
    opened_at: str
    closed_at: str | None

    @classmethod
    def from_row(cls, row) -> DemoPosition:
        return cls(
            id=int(row["id"]),
            account_id=int(row["account_id"]),
            signal_id=int(row["signal_id"]),
            symbol=row["symbol"],
            side=row["side"],
            entry_price=Decimal(str(row["entry_price"])),
            quantity=Decimal(str(row["quantity"])),
            position_size=Decimal(str(row["position_size"])),
            margin=Decimal(str(row["margin"])),
            leverage=int(row["leverage"]),
            stop_loss=Decimal(str(row["stop_loss"])),
            take_profit=Decimal(str(row["take_profit"])),
            unrealized_pnl=_dec_opt(row["unrealized_pnl"], "unrealized_pnl"),
            status=row["status"],
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
        )


@dataclass(frozen=True)
class DemoTrade:
    """A completed demo trade recorded on TP/SL (never a real order)."""

    id: int
    account_id: int
    position_id: int
    signal_id: int
    side: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    margin: Decimal
    position_size: Decimal
    leverage: int
    gross_pnl: Decimal
    fee: Decimal
    net_pnl: Decimal
    pnl_percent: Decimal
    result: str
    opened_at: str
    closed_at: str
    #: Symbol inherited from the closed position (None when the position row
    #: is gone or the query did not join it). Display only, never math input.
    symbol: str | None = None

    @classmethod
    def from_row(cls, row) -> DemoTrade:
        keys = row.keys() if hasattr(row, "keys") else ()
        return cls(
            id=int(row["id"]),
            account_id=int(row["account_id"]),
            position_id=int(row["position_id"]),
            signal_id=int(row["signal_id"]),
            side=row["side"],
            entry_price=Decimal(str(row["entry_price"])),
            exit_price=Decimal(str(row["exit_price"])),
            quantity=Decimal(str(row["quantity"])),
            margin=Decimal(str(row["margin"])),
            position_size=Decimal(str(row["position_size"])),
            leverage=int(row["leverage"]),
            gross_pnl=Decimal(str(row["gross_pnl"])),
            fee=Decimal(str(row["fee"])),
            net_pnl=Decimal(str(row["net_pnl"])),
            pnl_percent=Decimal(str(row["pnl_percent"])),
            result=row["result"],
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            symbol=row["position_symbol"] if "position_symbol" in keys else None,
        )


__all__ = [
    "DemoAccountRecord",
    "DemoPosition",
    "DemoTrade",
    "direction_long",
    "direction_short",
]
