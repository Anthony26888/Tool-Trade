"""Backtest configuration and error vocabulary (Phase 12).

The backtest engine is fully isolated from production state: it never touches
SQLite, never calls Binance, never invokes an LLM by default, and never places
orders. ``BacktestConfig`` carries the deterministic inputs that define a run:
which pair/timeframes to replay, the Phase 8 demo-account money model reused
verbatim (margin x leverage, fee rate), optional slippage, optional funding
rates, and the indicator warm-up window.

Every configuration rule is enforced eagerly so an invalid run fails before
any market data is consumed. Errors are named ``Backtest*`` subclasses so
callers can tell backtest failures apart from other packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from binance.indicators import MIN_REQUIRED_CANDLES
from binance.market_data import validate_interval, validate_symbol
from demo.account import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_LEVERAGE,
    DEFAULT_MARGIN_PER_TRADE,
    DEFAULT_RISK_PERCENT,
)

#: How many most-recent closed candles a decision provider is shown. Mirrors
#: the Phase 5 context look-back so a real analyzer can be plugged in as-is.
DEFAULT_MAX_CANDLES = 20

#: The default decision (analysis) timeframe for a backtest.
DEFAULT_DECISION_TIMEFRAME = "1h"

#: The default execution (entry/TP/SL confirmation) timeframe.
DEFAULT_EXECUTION_INTERVAL = "1m"

#: Binance kline open-time spacing in milliseconds, per supported interval.
INTERVAL_MS_BY_INTERVAL: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


class BacktestError(Exception):
    """Base class for all backtest failures."""


class BacktestConfigError(BacktestError, ValueError):
    """A :class:`BacktestConfig` value is invalid."""


class BacktestDataError(BacktestError, ValueError):
    """Historical market data is missing, malformed, or inconsistent."""


class BacktestExecutionError(BacktestError, ValueError):
    """A runtime failure during a backtest run (e.g. a malformed decision)."""


class BacktestMetricError(BacktestError, ValueError):
    """Statistics cannot be computed from an inconsistent result."""


class BacktestExportError(BacktestError, OSError):
    """A backtest result could not be exported to disk."""


def interval_to_ms(interval: str) -> int:
    """Milliseconds between kline open times for a supported interval."""
    try:
        return INTERVAL_MS_BY_INTERVAL[validate_interval(interval)]
    except Exception as exc:
        raise BacktestConfigError(str(exc)) from exc


def _decimal_bounded(value: Any, field: str, *, minimum: Decimal) -> Decimal:
    if isinstance(value, bool):
        raise BacktestConfigError(f"{field} must be a number")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BacktestConfigError(f"{field} must be a decimal number") from exc
    if not decimal_value.is_finite():
        raise BacktestConfigError(f"{field} must be finite")
    if decimal_value < minimum:
        raise BacktestConfigError(f"{field} must be >= {minimum}")
    return decimal_value


def _positive(value: Any, field: str) -> Decimal:
    decimal_value = _decimal_bounded(value, field, minimum=Decimal("0"))
    if decimal_value <= 0:
        raise BacktestConfigError(f"{field} must be positive")
    return decimal_value


def _nonnegative(value: Any, field: str) -> Decimal:
    return _decimal_bounded(value, field, minimum=Decimal("0"))


@dataclass(frozen=True)
class BacktestConfig:
    """Validated, deterministic configuration of one backtest run.

    Money fields are ``Decimal``; ``fee_rate`` and ``slippage_bps`` are the
    only ratios. ``funding_rates`` maps a funding timestamp (epoch ms, UTC) to
    a funding rate applied to the position notional while a trade is open;
    ``None`` (the default) disables funding entirely.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = DEFAULT_DECISION_TIMEFRAME
    execution_interval: str = DEFAULT_EXECUTION_INTERVAL
    initial_balance: Decimal = DEFAULT_INITIAL_BALANCE
    margin_per_trade: Decimal = DEFAULT_MARGIN_PER_TRADE
    leverage: int = DEFAULT_LEVERAGE
    risk_percent: Decimal = DEFAULT_RISK_PERCENT
    fee_rate: Decimal = DEFAULT_FEE_RATE
    slippage_bps: Decimal = Decimal("0")
    funding_rates: Mapping[int, Decimal] | None = None
    min_candles: int = MIN_REQUIRED_CANDLES
    max_candles: int = DEFAULT_MAX_CANDLES

    def __post_init__(self) -> None:
        try:
            symbol = validate_symbol(self.symbol)
            timeframe = validate_interval(self.timeframe)
            execution_interval = validate_interval(self.execution_interval)
        except Exception as exc:
            raise BacktestConfigError(str(exc)) from exc
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "execution_interval", execution_interval)
        if self.funding_rates is not None:
            normalized = {
                timestamp: _nonnegative(rate, f"funding_rates[{timestamp}]")
                for timestamp, rate in self.funding_rates.items()
            }
            object.__setattr__(self, "funding_rates", normalized)
        validate_config(self)


def validate_config(config: BacktestConfig) -> None:
    """Validate every :class:`BacktestConfig` field, raising on the first bad one."""
    if not isinstance(config, BacktestConfig):
        raise BacktestConfigError("config must be a BacktestConfig")
    try:
        validate_symbol(config.symbol)
        interval_to_ms(config.timeframe)
        interval_to_ms(config.execution_interval)
    except Exception as exc:
        raise BacktestConfigError(str(exc)) from exc
    _positive(config.initial_balance, "initial_balance")
    _positive(config.margin_per_trade, "margin_per_trade")
    _positive(config.risk_percent, "risk_percent")
    if isinstance(config.leverage, bool) or not isinstance(config.leverage, int):
        raise BacktestConfigError("leverage must be an integer")
    if config.leverage < 1:
        raise BacktestConfigError("leverage must be at least 1")
    fee_rate = _nonnegative(config.fee_rate, "fee_rate")
    if fee_rate >= 1:
        raise BacktestConfigError("fee_rate must be a fraction below 1")
    _nonnegative(config.slippage_bps, "slippage_bps")
    if config.min_candles < 1:
        raise BacktestConfigError("min_candles must be at least 1")
    if config.max_candles < 1:
        raise BacktestConfigError("max_candles must be at least 1")
    if config.funding_rates is not None:
        if not isinstance(config.funding_rates, Mapping):
            raise BacktestConfigError("funding_rates must be a mapping or None")
        for timestamp, rate in config.funding_rates.items():
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise BacktestConfigError("funding rate timestamps must be int epoch ms")
            if timestamp < 0:
                raise BacktestConfigError("funding rate timestamps must be non-negative")
            _nonnegative(rate, f"funding_rates[{timestamp}]")


__all__ = [
    "DEFAULT_DECISION_TIMEFRAME",
    "DEFAULT_EXECUTION_INTERVAL",
    "DEFAULT_MAX_CANDLES",
    "INTERVAL_MS_BY_INTERVAL",
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestDataError",
    "BacktestError",
    "BacktestExecutionError",
    "BacktestExportError",
    "BacktestMetricError",
    "interval_to_ms",
    "validate_config",
]
