"""Comparison table between two Phase 13 model benchmark summaries.

The comparison is deliberately judgment-free: it lays the two models' metrics
side by side and does NOT pick a winner. The human decides, using the priority
order documented in the plan:

    profit factor -> expectancy -> net PnL -> net return -> max drawdown ->
    consistency (streak/win stability) -> signal count -> latency -> win rate

``None``/missing values are rendered as an empty string; ``Decimal``-strings are
kept verbatim so no rounding is introduced by the comparison tool.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any

from .models import model_label

#: (metric row, label) pairs read from a summary's ``statistics`` / bench fields.
_METRIC_PATHS: list[tuple[str, str]] = [
    ("statistics.profit_factor", "Profit factor"),
    ("statistics.expectancy", "Expectancy"),
    ("statistics.net_pnl", "Net PnL (USDT)"),
    ("statistics.total_return_pct", "Net return %"),
    ("statistics.max_drawdown_pct", "Max drawdown %"),
    ("statistics.average_r", "Avg R multiple"),
    ("statistics.win_rate", "Win rate %"),
    ("statistics.long_win_rate", "LONG win rate %"),
    ("statistics.short_win_rate", "SHORT win rate %"),
    ("statistics.trades_completed", "Trades"),
    ("model_signal_quality.signal_count", "Signals (LONG+SHORT)"),
    ("statistics.ambiguous_total", "Ambiguous candles"),
    ("statistics.average_holding_time_ms", "Avg holding (ms)"),
    ("latency.avg_sec", "Avg inference latency (s)"),
    ("latency.failed_calls", "Failed LLM calls"),
    ("model_signal_quality.invalid_rate", "Invalid output rate"),
    ("model_signal_quality.long_short_ratio", "LONG/SHORT ratio"),
    ("model_signal_quality.wait_ratio", "WAIT ratio"),
]


class ComparisonError(ValueError):
    """A comparison could not be built from the given summaries."""


def _get_path(mapping: dict[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for segment in dotted.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _render(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def compare_summaries(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Build the side-by-side table (no winner logic) from two summaries."""
    for summary in (first, second):
        if not isinstance(summary, dict):
            raise ComparisonError("each comparison input must be a summary dict")

    def model_id(summary: dict[str, Any]) -> str:
        model = summary.get("model")
        if isinstance(model, dict) and model.get("id"):
            return str(model["id"])
        return "unknown"

    table = [
        {
            "metric": label,
            "first": _render(_get_path(first, path)),
            "second": _render(_get_path(second, path)),
        }
        for path, label in _METRIC_PATHS
    ]
    return {
        "benchmark_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first": {"model": model_id(first), "label": model_label(model_id(first))},
        "second": {"model": model_id(second), "label": model_label(model_id(second))},
        "table": table,
        "note": (
            "Human decision aid only: no automatic winner is selected. "
            "Suggested analysis priority: profit factor -> expectancy -> "
            "net PnL -> net return -> max drawdown -> consistency -> signal "
            "count -> latency -> win rate."
        ),
    }


def write_comparison(
    comparison: dict[str, Any],
    out_dir: str | os.PathLike[str],
) -> tuple[str, str]:
    """Write comparison.json + comparison.csv; returns the written paths."""
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    json_path = os.path.join(out, "comparison.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
        handle.write("\n")

    csv_path = os.path.join(out, "comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "first", "second"])
        for row in comparison["table"]:
            writer.writerow([row["metric"], row["first"], row["second"]])
    return json_path, csv_path


__all__ = ["ComparisonError", "compare_summaries", "write_comparison"]
