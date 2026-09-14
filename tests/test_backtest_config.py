"""Phase 12: BacktestConfig validation and interval helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from backtest.config import (
    BacktestConfig,
    BacktestConfigError,
    interval_to_ms,
    validate_config,
)

SUPPORTED_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def test_defaults_validate():
    config = BacktestConfig()
    assert config.initial_balance == Decimal("1000")
    assert config.margin_per_trade == Decimal("50")
    assert config.leverage == 10
    assert config.fee_rate == Decimal("0.0004")
    assert config.slippage_bps == Decimal("0")
    assert config.funding_rates is None
    assert config.min_candles == 200
    assert config.max_candles == 40
    validate_config(config)


def test_symbol_and_interval_normalization():
    config = BacktestConfig(symbol="btcusdt", timeframe="1H", execution_interval="1M")
    assert config.symbol == "BTCUSDT"
    assert config.timeframe == "1h"
    assert config.execution_interval == "1m"
    validate_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", ""),
        ("symbol", "BTC/X"),
        ("timeframe", "2h"),
        ("execution_interval", "1d"),
        ("initial_balance", "0"),
        ("initial_balance", "-10"),
        ("initial_balance", True),
        ("initial_balance", "abc"),
        ("initial_balance", "NaN"),
        ("margin_per_trade", "0"),
        ("margin_per_trade", "-1"),
        ("leverage", 0),
        ("leverage", -3),
        ("leverage", 1.5),
        ("leverage", True),
        ("fee_rate", "-0.001"),
        ("fee_rate", "1"),
        ("fee_rate", "1.5"),
        ("fee_rate", "abc"),
        ("slippage_bps", "-1"),
        ("slippage_bps", "abc"),
        ("min_candles", 0),
        ("min_candles", -2),
        ("max_candles", 0),
    ],
)
def test_config_rejects_invalid_field(field, value):
    with pytest.raises(BacktestConfigError):
        BacktestConfig(**{field: value})


def test_funding_rates_validation():
    good = BacktestConfig(funding_rates={1728000000: "0.0001"})
    assert good.funding_rates[1728000000] == Decimal("0.0001")
    with pytest.raises(BacktestConfigError):
        BacktestConfig(funding_rates={1.5: "0.0001"})
    with pytest.raises(BacktestConfigError):
        BacktestConfig(funding_rates={True: "0.0001"})
    with pytest.raises(BacktestConfigError):
        BacktestConfig(funding_rates={-1: "0.0001"})
    with pytest.raises(BacktestConfigError):
        BacktestConfig(funding_rates={1: "abc"})
    with pytest.raises(BacktestConfigError):
        BacktestConfig(funding_rates={1: "-0.0001"})


def test_validate_config_rejects_non_config():
    with pytest.raises(BacktestConfigError):
        validate_config(None)


def test_config_is_frozen():
    config = BacktestConfig()
    with pytest.raises(FrozenInstanceError):
        config.initial_balance = Decimal("5")


@pytest.mark.parametrize(("interval", "expected"), SUPPORTED_MS.items())
def test_interval_to_ms(interval, expected):
    assert interval_to_ms(interval) == expected


def test_interval_to_ms_rejects_unsupported():
    with pytest.raises(BacktestConfigError):
        interval_to_ms("1d")
