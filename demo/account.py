"""Demo account configuration and pure accounting math (Phase 8).

Phase 8 scope
-------------
This module is pure arithmetic over :class:`decimal.Decimal` values. It has no
database, no Binance, and no LLM imports, and it never places orders. The
``demo`` package is intentionally a namespace package in Phase 8 (there is no
``demo/__init__.py``), so this module stays importable in isolation.
:mod:`demo.position`, :mod:`demo.executor`, and :mod:`demo.statistics` arrive
in later phases and are only ever *forward-referenced* from here, never
imported at module load.

Accounting model (Phase 8)
--------------------------
- ``balance`` is the available balance.
- Opening a position reserves margin but does not move the balance; the
  balance only changes at close time.
- ``equity`` equals ``balance`` in Phase 8: open-position mark-to-market is
  not modelled here, so equity never uses invented prices.
- ``net_pnl`` is gross PnL minus the entry fee and the exit fee. The balance
  is updated with net PnL (AGENTS.md section 15).
- Money is always ``Decimal``; binary floats are never used for money.

Configuration defaults (AGENTS.md section 13)
---------------------------------------------
initial_balance = 1000 USDT
margin_per_trade = 50 USDT
leverage = 10x
risk_percent = 1 (i.e. 1%)
fee_rate = 0.0004 (i.e. 0.04%, stored as a fraction)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

DEFAULT_INITIAL_BALANCE = Decimal("1000")
DEFAULT_MARGIN_PER_TRADE = Decimal("50")
DEFAULT_LEVERAGE = 10
DEFAULT_RISK_PERCENT = Decimal("1")
DEFAULT_FEE_RATE = Decimal("0.0004")

DIRECTION_LONG = "LONG"
DIRECTION_SHORT = "SHORT"
DIRECTIONS = frozenset({DIRECTION_LONG, DIRECTION_SHORT})


class DemoAccountError(Exception):
    """Base class for demo account errors."""


class DemoConfigError(DemoAccountError, ValueError):
    """A demo account configuration value is invalid."""


class DemoCalculationError(DemoAccountError, ValueError):
    """A positional or financial input to an accounting function is invalid."""


def _to_decimal_api(value: Any, field: str) -> Decimal:
    """Exact conversion of a numeric input, raising ``DemoCalculationError``."""
    if isinstance(value, bool):
        raise DemoCalculationError(f"{field} must be a number")
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise DemoCalculationError(f"{field} must be a decimal number") from exc
    if not decimal_value.is_finite():
        raise DemoCalculationError(f"{field} must be finite")
    return decimal_value


def _require_positive(value: Any, field: str) -> Decimal:
    decimal_value = _to_decimal_api(value, field)
    if decimal_value <= 0:
        raise DemoCalculationError(f"{field} must be positive")
    return decimal_value


def _require_nonnegative(value: Any, field: str) -> Decimal:
    decimal_value = _to_decimal_api(value, field)
    if decimal_value < 0:
        raise DemoCalculationError(f"{field} must be non-negative")
    return decimal_value


def _validate_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise DemoCalculationError(f"direction must be one of {sorted(DIRECTIONS)}")
    return direction


def _require_positive_leverage(leverage: Any) -> int:
    if isinstance(leverage, bool):
        raise DemoConfigError("leverage must be an integer")
    if not isinstance(leverage, int):
        raise DemoConfigError("leverage must be an integer")
    if leverage < 1:
        raise DemoConfigError("leverage must be at least 1")
    return leverage


@dataclass(frozen=True)
class DemoConfig:
    """Validated demo account configuration.

    ``fee_rate`` is stored as a fraction (0.04% is ``Decimal("0.0004")``).
    ``risk_percent`` is a percent number (1 means 1% of the balance).
    """

    initial_balance: Decimal = DEFAULT_INITIAL_BALANCE
    margin_per_trade: Decimal = DEFAULT_MARGIN_PER_TRADE
    leverage: int = DEFAULT_LEVERAGE
    risk_percent: Decimal = DEFAULT_RISK_PERCENT
    fee_rate: Decimal = DEFAULT_FEE_RATE

    def __post_init__(self) -> None:
        validate_config(self)

    @classmethod
    def with_percent_fee(
        cls,
        *,
        initial_balance: Any = DEFAULT_INITIAL_BALANCE,
        margin_per_trade: Any = DEFAULT_MARGIN_PER_TRADE,
        leverage: int = DEFAULT_LEVERAGE,
        risk_percent: Any = DEFAULT_RISK_PERCENT,
        fee_rate_percent: Any = Decimal("0.04"),
    ) -> DemoConfig:
        """Build a config from a fee given as a percent (0.04 -> 0.04%)."""
        return cls(
            initial_balance=initial_balance,
            margin_per_trade=margin_per_trade,
            leverage=leverage,
            risk_percent=risk_percent,
            fee_rate=fee_rate_from_percent(fee_rate_percent),
        )


def validate_config(config: DemoConfig) -> None:
    """Validate every demo account configuration field.

    Raises:
        DemoConfigError: for any invalid field value.
    """
    if not isinstance(config, DemoConfig):
        raise DemoConfigError("config must be a DemoConfig")
    try:
        _require_positive(config.initial_balance, "initial_balance")
    except DemoCalculationError as exc:
        raise DemoConfigError(str(exc)) from exc
    try:
        _require_positive(config.margin_per_trade, "margin_per_trade")
    except DemoCalculationError as exc:
        raise DemoConfigError(str(exc)) from exc
    _require_positive_leverage(config.leverage)
    try:
        _require_positive(config.risk_percent, "risk_percent")
    except DemoCalculationError as exc:
        raise DemoConfigError(str(exc)) from exc
    try:
        _require_nonnegative(config.fee_rate, "fee_rate")
    except DemoCalculationError as exc:
        raise DemoConfigError(str(exc)) from exc
    if config.fee_rate >= 1:
        raise DemoConfigError("fee_rate must be a fraction below 1")


@dataclass(frozen=True)
class DemoAccount:
    """An account's persisted state plus on-demand equity.

    ``balance`` is the available balance. ``equity`` equals ``balance`` while
    no OPEN position exists; mark-to-market of reserved margin is a later-phase
    concern, so equity never invents prices in Phase 8.
    """

    id: int
    name: str
    initial_balance: Decimal
    balance: Decimal
    margin_per_trade: Decimal
    leverage: int
    risk_percent: Decimal
    fee_rate: Decimal
    created_at: str
    updated_at: str | None = None

    @classmethod
    def from_config(
        cls,
        config: DemoConfig,
        *,
        account_id: int = 0,
        name: str = "demo",
        created_at: str = "",
    ) -> DemoAccount:
        """New account at its initial balance for a validated config."""
        validate_config(config)
        return cls(
            id=account_id,
            name=name,
            initial_balance=config.initial_balance,
            balance=config.initial_balance,
            margin_per_trade=config.margin_per_trade,
            leverage=config.leverage,
            risk_percent=config.risk_percent,
            fee_rate=config.fee_rate,
            created_at=created_at,
        )

    def equity(self) -> Decimal:
        """Phase 8 equity: equal to the available balance."""
        return self.balance


def fee_rate_from_percent(percent: Any) -> Decimal:
    """Convert a fee percent (0.04 means 0.04%) into a fraction."""
    percent_value = _to_decimal_api(percent, "fee_rate_percent")
    return percent_value / Decimal(100)


def fee_rate_to_percent(fee_rate: Any) -> Decimal:
    """Convert a fractional fee rate back into a percent number."""
    rate = _require_nonnegative(fee_rate, "fee_rate")
    return rate * Decimal(100)


def risk_amount(balance: Any, risk_percent: Any) -> Decimal:
    """Maximum risk in USDT for a balance at a percent (1 -> 1%)."""
    balance_value = _require_positive(balance, "balance")
    percent = _require_positive(risk_percent, "risk_percent")
    return balance_value * percent / Decimal(100)


def position_size(margin_per_trade: Any, leverage: Any) -> Decimal:
    """Position notional when using the fixed margin: margin x leverage."""
    margin = _require_positive(margin_per_trade, "margin_per_trade")
    multiplier = _require_positive_leverage(leverage)
    return margin * multiplier


def position_quantity(position_notional: Any, entry_price: Any) -> Decimal:
    """Quantity for a notional entered at ``entry_price``."""
    notional = _require_positive(position_notional, "position_notional")
    entry = _require_positive(entry_price, "entry_price")
    return notional / entry


def gross_pnl(direction: str, entry_price: Any, exit_price: Any, quantity: Any) -> Decimal:
    """Gross PnL ignoring fees.

    LONG: ``(exit_price - entry_price) * quantity``
    SHORT: ``(entry_price - exit_price) * quantity``
    """
    _validate_direction(direction)
    entry = _require_positive(entry_price, "entry_price")
    exit_value = _require_positive(exit_price, "exit_price")
    qty = _require_positive(quantity, "quantity")
    if direction == DIRECTION_LONG:
        return (exit_value - entry) * qty
    return (entry - exit_value) * qty


def entry_fee(position_notional: Any, fee_rate: Any) -> Decimal:
    """Fee charged against the entry notional."""
    notional = _require_positive(position_notional, "position_notional")
    rate = _require_nonnegative(fee_rate, "fee_rate")
    return notional * rate


def exit_fee(exit_price: Any, quantity: Any, fee_rate: Any) -> Decimal:
    """Fee charged against the exit notional (exit_price * quantity)."""
    exit_value = _require_positive(exit_price, "exit_price")
    qty = _require_positive(quantity, "quantity")
    rate = _require_nonnegative(fee_rate, "fee_rate")
    return exit_value * qty * rate


def total_fee(
    position_notional: Any,
    exit_price: Any,
    quantity: Any,
    fee_rate: Any,
) -> Decimal:
    """Entry fee plus exit fee for one closed position."""
    return entry_fee(position_notional, fee_rate) + exit_fee(exit_price, quantity, fee_rate)


def net_pnl(
    direction: str,
    entry_price: Any,
    exit_price: Any,
    quantity: Any,
    position_notional: Any,
    fee_rate: Any,
) -> Decimal:
    """Net PnL: gross PnL minus entry and exit fees."""
    gross = gross_pnl(direction, entry_price, exit_price, quantity)
    fees = total_fee(position_notional, exit_price, quantity, fee_rate)
    return gross - fees


def balance_after_close(
    balance: Any,
    direction: str,
    entry_price: Any,
    exit_price: Any,
    quantity: Any,
    position_notional: Any,
    fee_rate: Any,
) -> Decimal:
    """Available balance after closing a position with net PnL.

    The balance is updated with net PnL only (AGENTS.md section 15). Reserved
    margin is not part of Phase 8 balance accounting, so it never appears here.
    """
    available = _require_nonnegative(balance, "balance")
    return available + net_pnl(
        direction,
        entry_price,
        exit_price,
        quantity,
        position_notional,
        fee_rate,
    )


def update_peak_equity(peak_equity: Any, equity_value: Any) -> Decimal:
    """New peak equity after observing ``equity_value``."""
    peak = _require_nonnegative(peak_equity, "peak_equity")
    value = _require_nonnegative(equity_value, "equity")
    return value if value > peak else peak


def current_drawdown(peak_equity: Any, equity_value: Any) -> Decimal:
    """Current absolute drawdown in USDT from the peak equity."""
    peak = _require_nonnegative(peak_equity, "peak_equity")
    value = _require_nonnegative(equity_value, "equity")
    loss = peak - value
    return loss if loss > 0 else Decimal("0")


__all__ = [
    "DEFAULT_INITIAL_BALANCE",
    "DEFAULT_MARGIN_PER_TRADE",
    "DEFAULT_LEVERAGE",
    "DEFAULT_RISK_PERCENT",
    "DEFAULT_FEE_RATE",
    "DIRECTION_LONG",
    "DIRECTION_SHORT",
    "DIRECTIONS",
    "DemoAccount",
    "DemoAccountError",
    "DemoCalculationError",
    "DemoConfig",
    "DemoConfigError",
    "balance_after_close",
    "current_drawdown",
    "entry_fee",
    "exit_fee",
    "fee_rate_from_percent",
    "fee_rate_to_percent",
    "gross_pnl",
    "net_pnl",
    "position_quantity",
    "position_size",
    "risk_amount",
    "total_fee",
    "update_peak_equity",
    "validate_config",
]
