"""Demo trading package (Phase 8 + Phase 14).

Phase 8 (:mod:`demo.account`) is pure accounting arithmetic over
:class:`decimal.Decimal`: no database, no Binance, no LLM, no orders.

Phase 14 adds the persistence/execution integration on top of the same
formulas — the formula functions in :mod:`demo.account` are never modified:

- :mod:`demo.position` — domain records for the demo account, positions and
  trades (raw SQLite rows converted to safe ``Decimal`` dataclasses).
- :mod:`demo.executor` — ``DemoExecutor``: opens/closes demo positions and
  records trades when a signal opens/closes, idempotently and restart-safe.
- :mod:`demo.statistics` — win/loss statistics (win rate, profit factor,
  average R) and max drawdown derived purely from the persisted ledger.

The demo never places a Binance order, never uses a trading API key, and only
simulates execution from public market data (AGENTS.md sections 11, 12 and 18).
"""

from .account import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_LEVERAGE,
    DEFAULT_MARGIN_PER_TRADE,
    DEFAULT_RISK_PERCENT,
    DemoAccount,
    DemoAccountError,
    DemoCalculationError,
    DemoConfig,
    DemoConfigError,
    balance_after_close,
    current_drawdown,
    entry_fee,
    exit_fee,
    fee_rate_from_percent,
    fee_rate_to_percent,
    gross_pnl,
    net_pnl,
    position_quantity,
    position_size,
    risk_amount,
    total_fee,
    update_peak_equity,
    validate_config,
)
from .executor import DemoExecutor
from .position import (
    DemoAccountRecord,
    DemoPosition,
    DemoTrade,
    direction_long,
    direction_short,
)
from .statistics import DemoStatistics, demo_statistics

__all__ = [
    "DEFAULT_FEE_RATE",
    "DEFAULT_INITIAL_BALANCE",
    "DEFAULT_LEVERAGE",
    "DEFAULT_MARGIN_PER_TRADE",
    "DEFAULT_RISK_PERCENT",
    "DemoAccount",
    "DemoAccountError",
    "DemoAccountRecord",
    "DemoCalculationError",
    "DemoConfig",
    "DemoConfigError",
    "DemoExecutor",
    "DemoPosition",
    "DemoStatistics",
    "DemoTrade",
    "balance_after_close",
    "current_drawdown",
    "demo_statistics",
    "direction_long",
    "direction_short",
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
