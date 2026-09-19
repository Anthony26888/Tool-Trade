"""Phase 8 tests: demo account configuration and pure accounting math.

The frozen Phase 8 scope is exactly two files: ``demo/account.py`` and this
test module. The ``demo`` package has no ``__init__.py`` on purpose (namespace
package), so ``demo.account`` must stay importable with no side modules.
"""

from __future__ import annotations

import inspect
from decimal import Decimal

import pytest

from demo.account import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_LEVERAGE,
    DEFAULT_MARGIN_PER_TRADE,
    DEFAULT_RISK_PERCENT,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    DemoAccount,
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
    resolve_quantity,
    risk_amount,
    slippage_factor,
    total_fee,
    update_peak_equity,
    validate_config,
)

LONG = DIRECTION_LONG
SHORT = DIRECTION_SHORT


class TestDefaults:
    def test_default_values(self):
        assert Decimal("1000") == DEFAULT_INITIAL_BALANCE
        assert Decimal("50") == DEFAULT_MARGIN_PER_TRADE
        assert DEFAULT_LEVERAGE == 10
        assert Decimal("1") == DEFAULT_RISK_PERCENT
        assert Decimal("0.0004") == DEFAULT_FEE_RATE

    def test_default_config_is_valid(self):
        config = DemoConfig()
        validate_config(config)
        assert config.initial_balance == Decimal("1000")
        assert config.margin_per_trade == Decimal("50")
        assert config.leverage == 10
        assert config.risk_percent == Decimal("1")
        assert config.fee_rate == Decimal("0.0004")

    def test_with_percent_fee_converts_to_fraction(self):
        config = DemoConfig.with_percent_fee(fee_rate_percent=Decimal("0.04"))
        assert config.fee_rate == Decimal("0.0004")

    def test_custom_config_is_accepted(self):
        config = DemoConfig(
            initial_balance=Decimal("2000"),
            margin_per_trade=Decimal("100"),
            leverage=5,
            risk_percent=Decimal("2"),
            fee_rate=Decimal("0.001"),
        )
        validate_config(config)
        assert config.fee_rate == Decimal("0.001")


class TestFeeConversions:
    def test_percent_to_fraction(self):
        assert fee_rate_from_percent(Decimal("0.04")) == Decimal("0.0004")

    def test_fraction_to_percent(self):
        assert fee_rate_to_percent(Decimal("0.0004")) == Decimal("0.04")

    def test_roundtrip(self):
        assert fee_rate_to_percent(fee_rate_from_percent(Decimal("0.5"))) == Decimal("0.5")

    def test_zero_fee_allowed(self):
        assert fee_rate_from_percent(0) == Decimal("0")


class TestDemoConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"initial_balance": Decimal("0")},
            {"initial_balance": Decimal("-1")},
            {"margin_per_trade": Decimal("0")},
            {"margin_per_trade": Decimal("-5")},
            {"leverage": 0},
            {"leverage": -2},
            {"leverage": 2.5},
            {"leverage": True},
            {"risk_percent": Decimal("0")},
            {"risk_percent": Decimal("-1")},
            {"fee_rate": Decimal("-0.01")},
            {"fee_rate": Decimal("1")},
            {"slippage_bps": Decimal("-1")},
            {"slippage_bps": "wide"},
        ],
    )
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(DemoConfigError):
            DemoConfig(**kwargs)

    def test_non_decimal_balance_rejected(self):
        with pytest.raises(DemoConfigError):
            DemoConfig(initial_balance=Decimal("NaN"))

    def test_validate_config_type_check(self):
        with pytest.raises(DemoConfigError):
            validate_config("not a config")  # type: ignore[arg-type]


class TestSizingAndQuantity:
    def test_position_size_margin_times_leverage(self):
        assert position_size(Decimal("50"), 10) == Decimal("500")

    def test_quantity(self):
        assert position_quantity(Decimal("500"), Decimal("100")) == Decimal("5")

    def test_quantity_realistic_prices(self):
        size = position_size(Decimal("50"), 10)
        assert position_quantity(size, Decimal("61000")) == Decimal("500") / Decimal("61000")

    def test_position_size_rejects_bad_margin(self):
        with pytest.raises(DemoCalculationError):
            position_size(Decimal("0"), 10)

    def test_quantity_rejects_nonpositive_entry(self):
        with pytest.raises(DemoCalculationError):
            position_quantity(Decimal("500"), Decimal("0"))


class TestResolveQuantity:
    """Phase F: explicit margin-vs-risk sizing rule (AGENTS.md section 16)."""

    def test_margin_rule_wins_on_tight_stop(self):
        # 500/61000 = 0.00819... vs risk 10/1000 = 0.01 -> margin binds.
        qty, rule = resolve_quantity(1000, 50, 10, 1, 61000.0, 60000.0)
        assert rule == "margin"
        assert qty == Decimal("500") / Decimal("61000")

    def test_risk_rule_wins_on_wide_stop(self):
        # 500/61000 = 0.00819... vs risk 10/5000 = 0.002 -> risk binds.
        qty, rule = resolve_quantity(1000, 50, 10, 1, 61000.0, 56000.0)
        assert rule == "risk"
        assert qty == Decimal("0.002")

    def test_tie_prefers_margin(self):
        # 500/50000 = 0.01 vs risk 10/1000 = 0.01 -> margin (legacy).
        qty, rule = resolve_quantity(1000, 50, 10, 1, 50000.0, 49000.0)
        assert rule == "margin"
        assert qty == Decimal("0.01")

    def test_short_uses_absolute_distance(self):
        qty, rule = resolve_quantity(1000, 50, 10, 1, 59000.0, 60000.0)
        assert rule == "margin"
        assert qty == Decimal("500") / Decimal("59000")

    def test_zero_stop_distance_rejected(self):
        with pytest.raises(DemoCalculationError):
            resolve_quantity(1000, 50, 10, 1, 61000.0, 61000.0)

    def test_invalid_inputs_rejected(self):
        with pytest.raises(DemoCalculationError):
            resolve_quantity(0, 50, 10, 1, 61000.0, 60000.0)
        with pytest.raises(DemoCalculationError):
            resolve_quantity(1000, 50, 10, 1, -61000.0, 60000.0)


class TestSlippageFactor:
    def test_zero_is_identity(self):
        assert slippage_factor(0, direction=LONG, entering=True) == Decimal("1")
        assert slippage_factor(0, direction=SHORT, entering=False) == Decimal("1")

    def test_adverse_direction(self):
        # 10 bps: LONG pays up entering, gives up exiting; SHORT mirrors.
        assert slippage_factor(10, direction=LONG, entering=True) == Decimal("1.001")
        assert slippage_factor(10, direction=LONG, entering=False) == Decimal("0.999")
        assert slippage_factor(10, direction=SHORT, entering=True) == Decimal("0.999")
        assert slippage_factor(10, direction=SHORT, entering=False) == Decimal("1.001")

    def test_negative_rejected(self):
        with pytest.raises(DemoCalculationError):
            slippage_factor(-1, direction=LONG, entering=True)


class TestGrossPnl:
    def test_long_gross(self):
        assert gross_pnl(LONG, Decimal("100"), Decimal("110"), Decimal("5")) == Decimal("50")

    def test_long_negative_gross(self):
        assert gross_pnl(LONG, Decimal("100"), Decimal("95"), Decimal("5")) == Decimal("-25")

    def test_short_gross(self):
        assert gross_pnl(SHORT, Decimal("100"), Decimal("90"), Decimal("5")) == Decimal("50")

    def test_short_negative_gross(self):
        assert gross_pnl(SHORT, Decimal("100"), Decimal("105"), Decimal("5")) == Decimal("-25")

    def test_unknown_direction_rejected(self):
        with pytest.raises(DemoCalculationError):
            gross_pnl("SIDEWAYS", Decimal("100"), Decimal("110"), Decimal("5"))


class TestFees:
    def test_entry_fee_default_rate(self):
        assert entry_fee(Decimal("500"), DEFAULT_FEE_RATE) == Decimal("0.20")

    def test_exit_fee_default_rate(self):
        assert exit_fee(Decimal("110"), Decimal("5"), DEFAULT_FEE_RATE) == Decimal("0.22")

    def test_total_fee(self):
        assert total_fee(Decimal("500"), Decimal("110"), Decimal("5"), DEFAULT_FEE_RATE) == Decimal(
            "0.42"
        )

    def test_zero_fee_rate(self):
        assert total_fee(Decimal("500"), Decimal("110"), Decimal("5"), Decimal("0")) == Decimal("0")

    def test_fee_negative_rejected(self):
        with pytest.raises(DemoCalculationError):
            entry_fee(Decimal("500"), Decimal("-1"))


class TestNetPnl:
    def test_long_net(self):
        assert net_pnl(
            LONG,
            Decimal("100"),
            Decimal("110"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        ) == Decimal("50") - Decimal("0.20") - Decimal("0.22")

    def test_short_net(self):
        assert net_pnl(
            SHORT,
            Decimal("100"),
            Decimal("90"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        ) == Decimal("50") - Decimal("0.20") - Decimal("0.18")

    def test_breakeven_price_after_fees_is_negative(self):
        gross_zero = net_pnl(
            LONG,
            Decimal("100"),
            Decimal("100"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        )
        assert gross_zero == -(Decimal("0.20") + Decimal("0.20"))


class TestBalanceUpdate:
    def test_long_win_raises_balance(self):
        after = balance_after_close(
            Decimal("1000"),
            LONG,
            Decimal("100"),
            Decimal("110"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        )
        assert after == Decimal("1049.58")

    def test_long_loss_lowers_balance(self):
        after = balance_after_close(
            Decimal("1000"),
            LONG,
            Decimal("100"),
            Decimal("95"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        )
        assert after == Decimal("1000") - Decimal("25") - Decimal("0.20") - Decimal("0.19")

    def test_short_win_raises_balance(self):
        after = balance_after_close(
            Decimal("1000"),
            SHORT,
            Decimal("100"),
            Decimal("90"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        )
        assert after == Decimal("1049.62")

    def test_balance_matches_net_pnl_decomposition(self):
        start = Decimal("1000")
        after = balance_after_close(
            start,
            LONG,
            Decimal("100"),
            Decimal("110"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        )
        assert after == start + net_pnl(
            LONG,
            Decimal("100"),
            Decimal("110"),
            Decimal("5"),
            Decimal("500"),
            DEFAULT_FEE_RATE,
        )
        assert after == start + gross_pnl(
            LONG, Decimal("100"), Decimal("110"), Decimal("5")
        ) - total_fee(Decimal("500"), Decimal("110"), Decimal("5"), DEFAULT_FEE_RATE)


class TestAccount:
    def test_account_from_config_starts_at_initial_balance(self):
        account = DemoAccount.from_config(DemoConfig())
        assert account.initial_balance == Decimal("1000")
        assert account.balance == Decimal("1000")
        assert account.leverage == 10
        assert account.margin_per_trade == Decimal("50")
        assert account.risk_percent == Decimal("1")
        assert account.fee_rate == Decimal("0.0004")
        assert account.equity() == Decimal("1000")

    def test_equity_equals_balance_in_phase_8(self):
        account = DemoAccount.from_config(DemoConfig())
        assert account.equity() == account.balance


class TestEquityAndDrawdown:
    def test_peak_equity_updates_upward_only(self):
        assert update_peak_equity(Decimal("1000"), Decimal("1050")) == Decimal("1050")
        assert update_peak_equity(Decimal("1000"), Decimal("900")) == Decimal("1000")

    def test_current_drawdown_two_percent(self):
        peak = Decimal("1000")
        assert current_drawdown(peak, Decimal("900")) == Decimal("100")

    def test_current_drawdown_zero_when_at_peak(self):
        assert current_drawdown(Decimal("1000"), Decimal("1000")) == Decimal("0")

    def test_current_drawdown_zero_when_above_peak(self):
        assert current_drawdown(Decimal("1000"), Decimal("1100")) == Decimal("0")

    def test_drawdown_from_losing_trade(self):
        account = DemoAccount.from_config(DemoConfig())
        after = balance_after_close(
            account.balance,
            LONG,
            Decimal("100"),
            Decimal("95"),
            Decimal("5"),
            Decimal("500"),
            account.fee_rate,
        )
        assert current_drawdown(Decimal("1000"), after) == Decimal("1000") - after


class TestRisk:
    def test_risk_amount_one_percent_of_balance(self):
        assert risk_amount(Decimal("1000"), Decimal("1")) == Decimal("10")

    def test_risk_amount_two_percent(self):
        assert risk_amount(Decimal("1000"), Decimal("2")) == Decimal("20")


class TestCalculationValidation:
    def test_string_number_coerced_exactly(self):
        assert gross_pnl(LONG, "100", "110", "5") == Decimal("50")

    def test_bool_rejected(self):
        with pytest.raises(DemoCalculationError):
            position_quantity(True, Decimal("100"))

    def test_infinite_rejected(self):
        with pytest.raises(DemoCalculationError):
            position_quantity(Decimal("Infinity"), Decimal("100"))


class TestPureModuleSafety:
    def test_no_dependency_imports(self):
        import demo.account as account_module

        module_source = inspect.getsource(account_module)
        for forbidden in (
            "import database",
            "from database",
            "import binance",
            "from binance",
            "import signal_engine",
            "from signal_engine",
        ):
            assert forbidden not in module_source

    def test_no_submodule_imports(self):
        import demo.account as account_module

        module_source = inspect.getsource(account_module)
        for forbidden in (
            "from demo.position",
            "import demo.position",
            "from demo.executor",
            "import demo.executor",
            "from demo.statistics",
            "import demo.statistics",
        ):
            assert forbidden not in module_source

    def test_no_order_execution_words(self):
        import demo.account as account_module

        for forbidden in ("order_api", "place_order", "postOrder", "trade.endpoint"):
            assert forbidden not in inspect.getsource(account_module)
