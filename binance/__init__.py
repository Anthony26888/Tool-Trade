"""BTCUSDT Binance USDT-M Futures public package (Phases 1-2).

Public market data and deterministic technical indicators. No order execution
and no API credentials. Kept isolated from the TradingAgents vendor/yfinance
dataflows.
"""

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    BinanceConnectionError,
    BinanceError,
    BinanceFuturesClient,
    BinanceHTTPError,
    BinanceRateLimitError,
    BinanceServerError,
)
from .indicators import (
    MIN_REQUIRED_CANDLES,
    WARM_UP_CANDLES,
    EmptyIndicatorDataError,
    FormingCandleError,
    IndicatorError,
    IndicatorSnapshot,
    InsufficientDataError,
    compute_indicator_matrix,
    latest_indicators,
)
from .market_data import (
    DEFAULT_SYMBOL,
    SUPPORTED_INTERVALS,
    BinanceMarketData,
    Candle,
    EmptyKlineError,
    InvalidIntervalError,
    InvalidSymbolError,
    MalformedKlineError,
    candles_to_dataframe,
    is_candle_closed,
    parse_klines,
    validate_interval,
    validate_symbol,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_SYMBOL",
    "DEFAULT_TIMEOUT_SECONDS",
    "SUPPORTED_INTERVALS",
    "MIN_REQUIRED_CANDLES",
    "WARM_UP_CANDLES",
    "BinanceConnectionError",
    "BinanceError",
    "BinanceFuturesClient",
    "BinanceHTTPError",
    "BinanceMarketData",
    "BinanceRateLimitError",
    "BinanceServerError",
    "Candle",
    "EmptyIndicatorDataError",
    "EmptyKlineError",
    "FormingCandleError",
    "IndicatorError",
    "IndicatorSnapshot",
    "InsufficientDataError",
    "InvalidIntervalError",
    "InvalidSymbolError",
    "MalformedKlineError",
    "candles_to_dataframe",
    "compute_indicator_matrix",
    "is_candle_closed",
    "latest_indicators",
    "parse_klines",
    "validate_interval",
    "validate_symbol",
]
