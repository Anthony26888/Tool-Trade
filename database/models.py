"""Signal domain model and persistence constants (Phase 3).

Financial values never rely on binary floating point: money/TEXT columns
store exact ``Decimal`` strings (see ``database.DECIMAL_COLUMNS``). All
timestamps are ISO-8601 UTC strings ending in ``Z``; local machine time is
never written to the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Direction = Literal["LONG", "SHORT"]
SignalStatus = Literal["PENDING_ENTRY", "OPEN", "TP_HIT", "SL_HIT", "CANCELLED"]

DIRECTION_LONG: Direction = "LONG"
DIRECTION_SHORT: Direction = "SHORT"
DIRECTIONS: frozenset[Direction] = frozenset({DIRECTION_LONG, DIRECTION_SHORT})

STATUS_PENDING_ENTRY = "PENDING_ENTRY"
STATUS_OPEN = "OPEN"
STATUS_TP_HIT = "TP_HIT"
STATUS_SL_HIT = "SL_HIT"
STATUS_CANCELLED = "CANCELLED"
STATUSES: frozenset[str] = frozenset(
    {
        STATUS_PENDING_ENTRY,
        STATUS_OPEN,
        STATUS_TP_HIT,
        STATUS_SL_HIT,
        STATUS_CANCELLED,
    }
)
#: Statuses that occupy the single-active slot: no AI, no new signal, while any
#: active signal exists. ``PENDING_ENTRY`` awaits its Entry trigger; ``OPEN`` is
#: the confirmed position awaiting TP/SL.
ACTIVE_STATUSES: frozenset[str] = frozenset({STATUS_PENDING_ENTRY, STATUS_OPEN})
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_TP_HIT, STATUS_SL_HIT, STATUS_CANCELLED}
)

#: Lifecycle: PENDING_ENTRY -> OPEN -> TP_HIT / SL_HIT / CANCELLED.
#: A pending signal NEVER transitions directly to TP_HIT or SL_HIT: the entry
#: must first be touched, then the normal exit monitoring applies.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_PENDING_ENTRY: frozenset({STATUS_OPEN, STATUS_CANCELLED}),
    STATUS_OPEN: TERMINAL_STATUSES,
}
ALLOWED_TRANSITIONS.update({status: frozenset() for status in TERMINAL_STATUSES})

PRICE_ORDERING: dict[str, str] = {
    DIRECTION_LONG: "stop_loss < entry < take_profit",
    DIRECTION_SHORT: "take_profit < entry < stop_loss",
}


class SignalError(Exception):
    """Base class for signal persistence errors."""


class SignalValidationError(ValueError, SignalError):
    """A signal field or price ordering is invalid."""


class SignalNotFoundError(SignalError):
    """No signal exists with the requested id."""


class SignalExistsError(SignalError):
    """Another active signal (PENDING_ENTRY or OPEN) already exists."""


class InvalidTransitionError(SignalError):
    """The requested status transition is not allowed."""


@dataclass(frozen=True)
class Signal:
    """A persisted BTCUSDT trading signal.

    ``entry``, ``stop_loss``, and ``take_profit`` are immutable once the signal
    is created. Money fields are ``Decimal`` backed by exact TEXT storage.
    ``status`` follows the lifecycle ``PENDING_ENTRY -> OPEN -> TP_HIT / SL_HIT / CANCELLED``;
    ``opened_at`` is NULL while the signal awaits its Entry trigger.
    """

    id: int
    symbol: str
    timeframe: str
    direction: Direction
    status: SignalStatus
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    created_at: str
    opened_at: str | None = None
    closed_at: str | None = None
    close_price: Decimal | None = None
    close_reason: str | None = None
    confidence: int | None = None
    risk_reward: Decimal | None = None
    rationale: str | None = None
    provider: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    strategy_name: str | None = None
    strategy_version: str | None = None
    result: str | None = None
    # Phase 6 audit metadata: which candle the AI analyzed and when.
    analysis_timestamp: str | None = None
    market_timestamp: str | None = None
    candle_close_price: Decimal | None = None

    @classmethod
    def from_row(cls, row) -> Signal:
        """Build a :class:`Signal` from a ``sqlite3.Row``."""
        return cls(
            id=int(row["id"]),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            direction=row["direction"],
            status=row["status"],
            entry=Decimal(row["entry"]),
            stop_loss=Decimal(row["stop_loss"]),
            take_profit=Decimal(row["take_profit"]),
            created_at=row["created_at"],
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            close_price=Decimal(row["close_price"]) if row["close_price"] is not None else None,
            close_reason=row["close_reason"],
            confidence=row["confidence"],
            risk_reward=Decimal(row["risk_reward"]) if row["risk_reward"] is not None else None,
            rationale=row["rationale"],
            provider=row["provider"],
            model_name=row["model_name"],
            temperature=float(row["temperature"]) if row["temperature"] is not None else None,
            strategy_name=row["strategy_name"],
            strategy_version=row["strategy_version"],
            result=row["result"],
            analysis_timestamp=row["analysis_timestamp"],
            market_timestamp=row["market_timestamp"],
            candle_close_price=Decimal(row["candle_close_price"])
            if row["candle_close_price"] is not None
            else None,
        )
