"""Backtest statistics (Phase 12), computed deterministically from integers/Decimals.

Statistics are derived only from closed trades and the engine's event counts.
Equity equals balance (Phase 8 semantics: no mark-to-market), so the drawdown is
computed over the recorded realized-equity curve. Ambiguous candles are never
counted as wins or losses — they are reported separately, exactly like the live
monitor.

Notation matching the plan (section 17):
``gross_pnl`` = sum of gross over completed trades; ``net_pnl`` = gross - fees -
funding over the same set. ``profit_factor`` is None when the gross loss is zero.
Per-direction stats only count completed trades. ``None`` values mean "not
computable" (e.g. no completed trades), never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .config import BacktestConfig, BacktestMetricError
from .engine import CLOSE_REASON_SL, CLOSE_REASON_TP, BacktestResult, BacktestTrade


def _completed(trades: tuple[BacktestTrade, ...]) -> list[BacktestTrade]:
    return [
        trade
        for trade in trades
        if trade.outcome in (CLOSE_REASON_TP, CLOSE_REASON_SL)
    ]


def _percent(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    if denominator == 0:
        return None
    return numerator / denominator * Decimal(100)


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _finite(value: Decimal | None) -> bool:
    return value is not None and value.is_finite()


@dataclass(frozen=True)
class BacktestStatistics:
    """Read-only summary of a backtest run (all amounts USDT unless stated)."""

    symbol: str
    timeframe: str
    execution_interval: str

    eligible: int
    analyzed: int
    wait_count: int
    long_decisions: int
    short_decisions: int
    rejected_count: int
    blocked_count: int

    pending_created: int
    entries_hit: int
    trades_completed: int
    open_at_end: int
    pending_at_end: int
    ambiguous_total: int
    entry_ambiguity_count: int
    exit_ambiguity_count: int

    long_created: int
    short_created: int

    wins: int = 0
    losses: int = 0
    win_rate: Decimal | None = None
    long_wins: int = 0
    long_losses: int = 0
    short_wins: int = 0
    short_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    long_net_pnl: Decimal = Decimal("0")
    short_net_pnl: Decimal = Decimal("0")
    long_win_rate: Decimal | None = None
    short_win_rate: Decimal | None = None

    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    gross_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    total_funding: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    profit_factor: Decimal | None = None
    expectancy: Decimal | None = None
    average_pnl: Decimal | None = None
    average_win: Decimal | None = None
    average_loss: Decimal | None = None
    average_r: Decimal | None = None
    average_holding_time_ms: int | None = None
    max_holding_time_ms: int | None = None

    peak_balance: Decimal = Decimal("0")
    final_balance: Decimal = Decimal("0")
    total_return_pct: Decimal | None = None
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: Decimal | None = None
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    @classmethod
    def compute(cls, config: BacktestConfig, result: BacktestResult) -> BacktestStatistics:
        """Derive statistics from a completed run; raises on inconsistent input."""
        if not isinstance(config, BacktestConfig) or not isinstance(result, BacktestResult):
            raise BacktestMetricError("config and result are required")
        trades = _completed(result.trades)

        wins = 0
        losses = 0
        gross_profit = Decimal("0")
        gross_loss = Decimal("0")
        gross_sum = Decimal("0")
        net_sum = Decimal("0")
        fees_sum = Decimal("0")
        funding_sum = Decimal("0")
        slippage_informational = Decimal("0")
        long_wins = 0
        long_losses = 0
        short_wins = 0
        short_losses = 0
        long_trades = 0
        short_trades = 0
        long_net = Decimal("0")
        short_net = Decimal("0")
        r_values: list[Decimal] = []
        holding_times: list[int] = []

        consecutive_wins = 0
        consecutive_losses = 0
        best_streak = 0
        worst_streak = 0

        for trade in trades:
            gross = trade.gross_pnl
            net = trade.net_pnl
            if not _finite(gross) or not _finite(net):
                raise BacktestMetricError(
                    f"completed trade {trade.index} missing PnL values"
                )
            gross_sum += gross
            net_sum += net
            if _finite(trade.entry_fee) and _finite(trade.exit_fee):
                fees_sum += trade.entry_fee + trade.exit_fee
            if _finite(trade.funding_cost):
                funding_sum += trade.funding_cost
            if gross > 0:
                gross_profit += gross
                wins += 1
                if trade.direction == "LONG":
                    long_wins += 1
                else:
                    short_wins += 1
                consecutive_wins += 1
                consecutive_losses = 0
            else:
                gross_loss += -gross
                losses += 1
                if trade.direction == "LONG":
                    long_losses += 1
                else:
                    short_losses += 1
                consecutive_losses += 1
                consecutive_wins = 0
            best_streak = max(best_streak, consecutive_wins)
            worst_streak = max(worst_streak, consecutive_losses)
            if trade.direction == "LONG":
                long_trades += 1
                long_net += net
            else:
                short_trades += 1
                short_net += net
            if _finite(trade.r_multiple):
                r_values.append(trade.r_multiple)
            if trade.holding_time_ms is not None:
                holding_times.append(trade.holding_time_ms)
            if trade.entry_price is not None and trade.exit_price is not None:
                level_gross = _level_pnl(trade)
                slippage_informational += abs(level_gross - trade.gross_pnl)

        completed = len(trades)
        wins_decimal = Decimal(wins)

        r_sum = sum(r_values, Decimal("0"))
        average_r = _ratio(r_sum, Decimal(completed)) if completed else None
        average_pnl = _ratio(net_sum, Decimal(completed)) if completed else None
        average_win = _ratio(gross_profit, Decimal(wins)) if wins else None
        average_loss = _ratio(gross_loss, Decimal(losses)) if losses else None

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else None
        )
        win_rate = _percent(wins_decimal, Decimal(completed))
        long_win_rate = _percent(Decimal(long_wins), Decimal(long_trades))
        short_win_rate = _percent(Decimal(short_wins), Decimal(short_trades))

        holding_total = sum(holding_times)
        average_holding = (
            holding_total // completed if holding_times and completed else None
        )
        max_holding = max(holding_times) if holding_times else None

        equity = list(result.equity_curve)
        if not equity:
            raise BacktestMetricError("equity curve is empty")
        peak = Decimal(equity[0])
        max_dd = Decimal("0")
        max_dd_peak = peak
        for point in equity:
            point_value = Decimal(point)
            if point_value > peak:
                peak = point_value
            drawdown = peak - point_value
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_peak = peak
        max_dd_pct = _percent(max_dd, max_dd_peak)

        final_balance = Decimal(result.final_balance)
        return cls(
            symbol=result.symbol,
            timeframe=result.timeframe,
            execution_interval=result.execution_interval,
            eligible=result.eligible,
            analyzed=result.analyzed,
            wait_count=result.wait_count,
            long_decisions=result.long_decisions,
            short_decisions=result.short_decisions,
            rejected_count=result.rejected_count,
            blocked_count=result.blocked_count,
            pending_created=result.pending_created,
            entries_hit=result.entries_hit,
            trades_completed=result.trades_completed,
            open_at_end=result.open_at_end,
            pending_at_end=result.pending_at_end,
            ambiguous_total=result.ambiguous_total,
            entry_ambiguity_count=result.entry_ambiguity_count,
            exit_ambiguity_count=result.exit_ambiguity_count,
            long_created=result.long_created,
            short_created=result.short_created,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            long_wins=long_wins,
            long_losses=long_losses,
            short_wins=short_wins,
            short_losses=short_losses,
            long_trades=long_trades,
            short_trades=short_trades,
            long_net_pnl=long_net,
            short_net_pnl=short_net,
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            gross_pnl=gross_sum,
            total_fees=fees_sum,
            total_funding=funding_sum,
            slippage_cost=slippage_informational,
            net_pnl=net_sum,
            profit_factor=profit_factor,
            expectancy=_ratio(net_sum, Decimal(completed)) if completed else None,
            average_pnl=average_pnl if completed else None,
            average_win=average_win if wins else None,
            average_loss=average_loss if losses else None,
            average_r=average_r if completed else None,
            average_holding_time_ms=average_holding,
            max_holding_time_ms=max_holding,
            peak_balance=peak,
            final_balance=final_balance,
            total_return_pct=_percent(
                final_balance - Decimal(config.initial_balance),
                Decimal(config.initial_balance),
            ),
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            consecutive_wins=best_streak,
            consecutive_losses=worst_streak,
        )

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> BacktestStatistics:
        """Rebuild from a dict exported by ``to_dict`` (instrumental for tests)."""
        return cls(
            symbol=mapping["symbol"],
            timeframe=mapping["timeframe"],
            execution_interval=mapping["execution_interval"],
            eligible=int(mapping["eligible"]),
            analyzed=int(mapping["analyzed"]),
            wait_count=int(mapping["wait_count"]),
            long_decisions=int(mapping["long_decisions"]),
            short_decisions=int(mapping["short_decisions"]),
            rejected_count=int(mapping["rejected_count"]),
            blocked_count=int(mapping["blocked_count"]),
            pending_created=int(mapping["pending_created"]),
            entries_hit=int(mapping["entries_hit"]),
            trades_completed=int(mapping["trades_completed"]),
            open_at_end=int(mapping["open_at_end"]),
            pending_at_end=int(mapping["pending_at_end"]),
            ambiguous_total=int(mapping["ambiguous_total"]),
            entry_ambiguity_count=int(mapping["entry_ambiguity_count"]),
            exit_ambiguity_count=int(mapping["exit_ambiguity_count"]),
            long_created=int(mapping["long_created"]),
            short_created=int(mapping["short_created"]),
            wins=int(mapping["wins"]),
            losses=int(mapping["losses"]),
            win_rate=_opt_decimal(mapping["win_rate"]),
            long_wins=int(mapping["long_wins"]),
            long_losses=int(mapping["long_losses"]),
            short_wins=int(mapping["short_wins"]),
            short_losses=int(mapping["short_losses"]),
            long_trades=int(mapping["long_trades"]),
            short_trades=int(mapping["short_trades"]),
            long_net_pnl=_decimal(mapping["long_net_pnl"]),
            short_net_pnl=_decimal(mapping["short_net_pnl"]),
            long_win_rate=_opt_decimal(mapping["long_win_rate"]),
            short_win_rate=_opt_decimal(mapping["short_win_rate"]),
            gross_profit=_decimal(mapping["gross_profit"]),
            gross_loss=_decimal(mapping["gross_loss"]),
            gross_pnl=_decimal(mapping["gross_pnl"]),
            total_fees=_decimal(mapping["total_fees"]),
            total_funding=_decimal(mapping["total_funding"]),
            slippage_cost=_decimal(mapping["slippage_cost"]),
            net_pnl=_decimal(mapping["net_pnl"]),
            profit_factor=_opt_decimal(mapping["profit_factor"]),
            expectancy=_opt_decimal(mapping["expectancy"]),
            average_pnl=_opt_decimal(mapping["average_pnl"]),
            average_win=_opt_decimal(mapping["average_win"]),
            average_loss=_opt_decimal(mapping["average_loss"]),
            average_r=_opt_decimal(mapping["average_r"]),
            average_holding_time_ms=_opt_int(mapping["average_holding_time_ms"]),
            max_holding_time_ms=_opt_int(mapping["max_holding_time_ms"]),
            peak_balance=_decimal(mapping["peak_balance"]),
            final_balance=_decimal(mapping["final_balance"]),
            total_return_pct=_opt_decimal(mapping["total_return_pct"]),
            max_drawdown=_decimal(mapping["max_drawdown"]),
            max_drawdown_pct=_opt_decimal(mapping["max_drawdown_pct"]),
            consecutive_wins=int(mapping["consecutive_wins"]),
            consecutive_losses=int(mapping["consecutive_losses"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize every field to JSON-safe primitives (str for Decimals/None)."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "execution_interval": self.execution_interval,
            "eligible": self.eligible,
            "analyzed": self.analyzed,
            "wait_count": self.wait_count,
            "long_decisions": self.long_decisions,
            "short_decisions": self.short_decisions,
            "rejected_count": self.rejected_count,
            "blocked_count": self.blocked_count,
            "pending_created": self.pending_created,
            "entries_hit": self.entries_hit,
            "trades_completed": self.trades_completed,
            "open_at_end": self.open_at_end,
            "pending_at_end": self.pending_at_end,
            "ambiguous_total": self.ambiguous_total,
            "entry_ambiguity_count": self.entry_ambiguity_count,
            "exit_ambiguity_count": self.exit_ambiguity_count,
            "long_created": self.long_created,
            "short_created": self.short_created,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": _str_or_none(self.win_rate),
            "long_wins": self.long_wins,
            "long_losses": self.long_losses,
            "short_wins": self.short_wins,
            "short_losses": self.short_losses,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "long_net_pnl": str(self.long_net_pnl),
            "short_net_pnl": str(self.short_net_pnl),
            "long_win_rate": _str_or_none(self.long_win_rate),
            "short_win_rate": _str_or_none(self.short_win_rate),
            "gross_profit": str(self.gross_profit),
            "gross_loss": str(self.gross_loss),
            "gross_pnl": str(self.gross_pnl),
            "total_fees": str(self.total_fees),
            "total_funding": str(self.total_funding),
            "slippage_cost": str(self.slippage_cost),
            "net_pnl": str(self.net_pnl),
            "profit_factor": _str_or_none(self.profit_factor),
            "expectancy": _str_or_none(self.expectancy),
            "average_pnl": _str_or_none(self.average_pnl),
            "average_win": _str_or_none(self.average_win),
            "average_loss": _str_or_none(self.average_loss),
            "average_r": _str_or_none(self.average_r),
            "average_holding_time_ms": _str_or_none(self.average_holding_time_ms),
            "max_holding_time_ms": _str_or_none(self.max_holding_time_ms),
            "peak_balance": str(self.peak_balance),
            "final_balance": str(self.final_balance),
            "total_return_pct": _str_or_none(self.total_return_pct),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": _str_or_none(self.max_drawdown_pct),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
        }


def _level_pnl(trade: BacktestTrade) -> Decimal:
    """Gross PnL using the configured exit level for the actual outcome."""
    try:
        from demo.account import gross_pnl as _gross
    except ImportError:  # pragma: no cover - defensive
        return Decimal("0")
    exit_level = (
        trade.take_profit_level if trade.outcome == CLOSE_REASON_TP else trade.stop_level
    )
    if trade.direction == "LONG":
        return _gross("LONG", trade.entry_level, exit_level, trade.quantity)
    return _gross("SHORT", trade.entry_level, exit_level, trade.quantity)


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BacktestMetricError(f"invalid decimal value {value!r}") from exc


def _opt_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


__all__ = ["BacktestStatistics"]
