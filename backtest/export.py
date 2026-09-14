"""Deterministic export of backtest results (Phase 12).

CSV (trades) and JSON (full result + statistics) writers are pure: they never
open a database or touch the network and they write only to an explicit output
path chosen by the caller, so a run never writes into the repository by itself.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .config import BacktestExportError
from .engine import BacktestResult, BacktestTrade
from .metrics import BacktestStatistics


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def trade_to_record(trade: BacktestTrade) -> dict[str, Any]:
    """One JSON/CSV row for a trade, ``Decimal`` values rendered as strings."""
    return {
        "index": trade.index,
        "signal_candle_ts": _iso(trade.signal_candle_ts),
        "signal_candle_ts_ms": trade.signal_candle_ts,
        "direction": trade.direction,
        "entry_level": str(trade.entry_level),
        "stop_level": str(trade.stop_level),
        "take_profit_level": str(trade.take_profit_level),
        "entry_time": _iso(trade.entry_time_ms),
        "entry_time_ms": trade.entry_time_ms,
        "exit_time": _iso(trade.exit_time_ms),
        "exit_time_ms": trade.exit_time_ms,
        "entry_price": _str_or_none(trade.entry_price),
        "exit_price": _str_or_none(trade.exit_price),
        "quantity": _str_or_none(trade.quantity),
        "position_size": str(trade.position_size),
        "entry_fee": _str_or_none(trade.entry_fee),
        "exit_fee": _str_or_none(trade.exit_fee),
        "funding_cost": _str_or_none(trade.funding_cost),
        "gross_pnl": _str_or_none(trade.gross_pnl),
        "net_pnl": _str_or_none(trade.net_pnl),
        "risk_amount": str(trade.risk_amount),
        "r_multiple": _str_or_none(trade.r_multiple),
        "outcome": _str_or_none(trade.outcome),
        "was_ambiguous": bool(trade.was_ambiguous),
        "holding_time_ms": _str_or_none(trade.holding_time_ms),
    }


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def export_trades_csv(path: str | os.PathLike[str], result: BacktestResult) -> str:
    """Write the trades ledger to a CSV file; returns the absolute path used."""
    if not isinstance(result, BacktestResult):
        raise BacktestExportError("result must be a BacktestResult")
    records = [trade_to_record(trade) for trade in result.trades]
    columns = list(records[0]) if records else [
        "index",
        "signal_candle_ts",
        "signal_candle_ts_ms",
        "direction",
        "entry_level",
        "stop_level",
        "take_profit_level",
        "entry_time",
        "entry_time_ms",
        "exit_time",
        "exit_time_ms",
        "entry_price",
        "exit_price",
        "quantity",
        "position_size",
        "entry_fee",
        "exit_fee",
        "funding_cost",
        "gross_pnl",
        "net_pnl",
        "risk_amount",
        "r_multiple",
        "outcome",
        "was_ambiguous",
        "holding_time_ms",
    ]
    try:
        with open(str(path), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    except OSError as exc:
        raise BacktestExportError(f"could not write trades CSV: {exc}") from exc
    return str(path)


def result_to_dict(result: BacktestResult) -> dict[str, Any]:
    """Serialize a result without statistics (raw events, deterministic)."""
    return {
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "execution_interval": result.execution_interval,
        "counts": {
            "eligible": result.eligible,
            "analyzed": result.analyzed,
            "wait_count": result.wait_count,
            "long_decisions": result.long_decisions,
            "short_decisions": result.short_decisions,
            "rejected_count": result.rejected_count,
            "blocked_count": result.blocked_count,
            "long_created": result.long_created,
            "short_created": result.short_created,
            "pending_created": result.pending_created,
            "entries_hit": result.entries_hit,
            "trades_completed": result.trades_completed,
            "open_at_end": result.open_at_end,
            "pending_at_end": result.pending_at_end,
            "ambiguous_total": result.ambiguous_total,
            "entry_ambiguity_count": result.entry_ambiguity_count,
            "exit_ambiguity_count": result.exit_ambiguity_count,
        },
        "trades": [trade_to_record(trade) for trade in result.trades],
        "pending_events": [
            {
                "signal_candle_ts": event.signal_candle_ts,
                "decision_time_ms": event.decision_time_ms,
                "direction": event.direction,
                "entry_level": str(event.entry_level),
                "stop_level": str(event.stop_level),
                "take_profit_level": str(event.take_profit_level),
                "entered": event.entered,
                "entered_at_ms": event.entered_at_ms,
                "closed": event.closed,
                "closed_at_ms": event.closed_at_ms,
                "outcome": event.outcome,
                "trade_index": event.trade_index,
            }
            for event in result.pending_events
        ],
        "ambiguous_events": [vars(event) for event in result.ambiguous_events],
        "blocked_events": [
            {
                "candle_ts": event.candle_ts,
                "decision_time_ms": event.decision_time_ms,
                "reason": event.reason,
                "active_direction": event.active_direction,
                "active_signal_candle_ts": event.active_signal_candle_ts,
            }
            for event in result.blocked_events
        ],
        "rejected_events": [vars(event) for event in result.rejected_events],
        "decisions": [
            {
                "candle_ts": decision.candle_ts,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "candle_close_price": decision.candle_close_price,
                "provider": decision.provider,
                "model": decision.model,
            }
            for decision in result.decisions
        ],
        "equity_curve": [str(point) for point in result.equity_curve],
        "final_balance": str(result.final_balance),
    }


def export_result_json(
    path: str | os.PathLike[str],
    result: BacktestResult,
    statistics: BacktestStatistics | None = None,
    *,
    indent: int = 2,
) -> str:
    """Write the full result (with optional statistics) to a JSON file."""
    if not isinstance(result, BacktestResult):
        raise BacktestExportError("result must be a BacktestResult")
    payload = result_to_dict(result)
    if statistics is not None:
        payload["statistics"] = statistics.to_dict()
    try:
        with open(str(path), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
            handle.write("\n")
    except OSError as exc:
        raise BacktestExportError(f"could not write result JSON: {exc}") from exc
    return str(path)


__all__ = [
    "export_result_json",
    "export_trades_csv",
    "result_to_dict",
    "trade_to_record",
]
