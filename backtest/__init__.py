"""Backtest engine (Phase 12) for the BTCUSDT Futures strategy.

The engine replays the production signal lifecycle — AI decision, PENDING_ENTRY,
OPEN, TP/SL close — against historical candles with strict no-look-ahead rules
and the Phase 8 money model. It is fully isolated: no SQLite, no Binance calls,
no LLM by default, and no order execution.
"""

from __future__ import annotations

from .config import (
    DEFAULT_DECISION_TIMEFRAME,
    DEFAULT_EXECUTION_INTERVAL,
    INTERVAL_MS_BY_INTERVAL,
    BacktestConfig,
    BacktestConfigError,
    BacktestDataError,
    BacktestError,
    BacktestExecutionError,
    BacktestExportError,
    BacktestMetricError,
    interval_to_ms,
    validate_config,
)
from .data import (
    HistoricalData,
    HistoricalDataProvider,
    JsonFileHistoricalDataProvider,
    MemoryHistoricalDataProvider,
    candle_from_dict,
    candle_to_dict,
    load_historical_data_json,
    save_historical_data_json,
    validate_historical_data,
)
from .engine import (
    AmbiguityEvent,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    BlockedEvent,
    DecisionRecord,
    PendingEvent,
    RejectedEvent,
)
from .metrics import BacktestStatistics
from .provider import (
    BacktestDecisionProvider,
    FunctionDecisionProvider,
    ReplayDecisionProvider,
    WaitDecisionProvider,
    coerce_decision,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DECISION_TIMEFRAME",
    "DEFAULT_EXECUTION_INTERVAL",
    "INTERVAL_MS_BY_INTERVAL",
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestDataError",
    "BacktestError",
    "BacktestExecutionError",
    "BacktestExportError",
    "BacktestMetricError",
    "BacktestEngine",
    "BacktestResult",
    "BacktestStatistics",
    "BacktestTrade",
    "AmbiguityEvent",
    "BlockedEvent",
    "DecisionRecord",
    "PendingEvent",
    "RejectedEvent",
    "BacktestDecisionProvider",
    "FunctionDecisionProvider",
    "HistoricalData",
    "HistoricalDataProvider",
    "JsonFileHistoricalDataProvider",
    "MemoryHistoricalDataProvider",
    "ReplayDecisionProvider",
    "WaitDecisionProvider",
    "candle_from_dict",
    "candle_to_dict",
    "coerce_decision",
    "interval_to_ms",
    "load_historical_data_json",
    "save_historical_data_json",
    "validate_config",
    "validate_historical_data",
]
