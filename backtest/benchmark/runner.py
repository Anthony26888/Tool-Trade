"""Sequential Phase 13 benchmark runner for one local LLM model.

Runs the isolated Phase 12 engine with an :class:`LLMDecisionProvider` for a
single model over a single dataset/config, then writes the per-model result
files:

- ``decisions.jsonl``  every recorded model decision (also the replay/cache store)
- ``summary.json``     trading + signal-quality + latency statistics
- ``backtest_result.json`` full engine result (Phase 12 export)
- ``backtest_trades.csv``  per-trade ledger (Phase 12 export)
- ``equity_curve.csv`` realized-equity curve

Models are benchmarked strictly sequentially: one invocation of
:func:`run_benchmark` handles exactly one model, so Qwen3 4B and Gemma 4B are
run one at a time on purpose.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from backtest.config import BacktestConfig, BacktestConfigError, interval_to_ms
from backtest.data import HistoricalData
from backtest.engine import BacktestEngine, BacktestResult
from backtest.export import export_result_json, export_trades_csv
from backtest.metrics import BacktestStatistics
from signal_engine.llm import LLMConfig

from .cache import DecisionCache, compute_scope
from .models import (
    CONTEXT_VERSION,
    DecisionStatus,
    ModelDecision,
    bucket_for_confidence,
    model_label,
)


class BenchmarkError(BacktestConfigError):
    """A Phase 13 benchmark run could not be executed."""


@dataclass(frozen=True)
class RunSummary:
    """Result of one model benchmark run."""

    model: str
    provider: str
    result: BacktestResult
    statistics: BacktestStatistics
    summary: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict)


def build_llm_decision_provider(
    *,
    llm_config: LLMConfig,
    cache: DecisionCache,
    context_version: int = CONTEXT_VERSION,
    dataset_id: str | None = None,
    force_fresh: bool = False,
    analyze_fn: Any = None,
):  # pragma: no cover - import indirection for testability
    """Build the real LLM-backed provider for one benchmark run.

    Kept as a module-level factory so tests and the CLI can inject a provider
    with a fake analyzer without touching production code paths.
    """
    from .provider import LLMDecisionProvider

    return LLMDecisionProvider(
        model=llm_config.model,
        provider=llm_config.provider,
        cache=cache,
        config=llm_config,
        context_version=context_version,
        dataset_id=dataset_id,
        force_fresh=force_fresh,
        analyze_fn=analyze_fn,
    )


def verify_llm_temperature(llm_config: LLMConfig) -> None:
    """Reject a benchmark temperature that is clearly outside the legal range."""
    if llm_config.temperature is None:
        return
    if isinstance(llm_config.temperature, bool) or not (
        0.0 <= float(llm_config.temperature) <= 2.0
    ):
        raise BenchmarkError(
            f"temperature must be in [0, 2], got {llm_config.temperature}"
        )


def build_positioning_wiring(
    *,
    history: Any,
    period_ms: int,
    analyze_base: Any,
) -> tuple[Any, list]:
    """Phase P1 benchmark wiring for a primed ``PositioningHistory``.

    Returns ``(analyze_fn, frozen)``: the analyzer wrapper threading the
    historical positioning note + funding rate into each decision (same
    prompt shape and guardrail input as live), and the JSON snapshot for
    ``positioning.json``. Accepts and forwards ``event_note``/``htf_note``/
    ``regime`` so it composes with the other wirings.
    """

    def _pos_analyze(
        candles: list,
        indicators: Any,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = 20,
        event_note: str | None = None,
        htf_note: str | None = None,
        regime: str | None = None,
    ) -> Any:
        now_ms = int(candles[-1].timestamp) + period_ms
        return analyze_base(
            candles,
            indicators,
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
            event_note=event_note,
            positioning=history.note_at(now_ms),
            funding_rate=history.funding_at(now_ms),
            htf_note=htf_note,
            regime=regime,
        )

    return _pos_analyze, history.frozen()


def build_htf_wiring(*, analyze_base: Any) -> Any:
    """Phase P2 benchmark wiring: 4H bias rolled up from the 1H dataset.

    Rolls the decision window's 1H candles into 4H candles with the same
    ``rollup_1h_to_4h`` the live path's data already satisfies, classifies
    with the same ``classify_htf``, and threads the note + regime inward.
    Accepts and forwards the outer wirings' notes (innermost wrapper).
    """
    from binance.htf import REGIME_NONE, classify_htf, rollup_1h_to_4h

    def _htf_analyze(
        candles: list,
        indicators: Any,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = 20,
        event_note: str | None = None,
        positioning: str | None = None,
        funding_rate: float | None = None,
    ) -> Any:
        try:
            bias = classify_htf(rollup_1h_to_4h(list(candles)))
        except Exception:
            bias = None
        regime = (
            bias.regime
            if bias is not None and bias.regime != REGIME_NONE
            else None
        )
        return analyze_base(
            candles,
            indicators,
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
            event_note=event_note,
            positioning=positioning,
            funding_rate=funding_rate,
            htf_note=bias.note if bias is not None else None,
            regime=regime,
        )

    return _htf_analyze


def build_event_wiring(    *,
    event_calendar: Any,
    hours: list,
    period_ms: int,
    analyze_base: Any,
) -> tuple[Any, Any, list]:
    """Phase N benchmark wiring for a primed ``EventCalendar``.

    Loads the historical range, freezes it for deterministic replay, and
    returns ``(blackout_fn, analyze_fn, frozen_events)``: the backtest gate,
    the event-note analyzer wrapper, and the JSON snapshot for ``events.json``.
    """
    event_calendar.load_range(
        int(hours[0].timestamp), int(hours[-1].timestamp) + period_ms
    )
    frozen = event_calendar.frozen()

    def _blackout(decision_ms: int) -> str | None:
        event = event_calendar.blackout_at(decision_ms)
        return event.title if event is not None else None

    def _noted_analyze(
        candles: list,
        indicators: Any,
        *,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = 20,
    ) -> Any:
        note = event_calendar.event_note(
            candles, int(candles[-1].timestamp) + period_ms
        )
        return analyze_base(
            candles,
            indicators,
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
            event_note=note,
        )

    return _blackout, _noted_analyze, frozen


def run_benchmark(
    *,
    config: BacktestConfig,
    data: HistoricalData,
    out_dir: str | os.PathLike[str],
    llm: LLMConfig,
    context_version: int = CONTEXT_VERSION,
    dataset_id: str | None = None,
    force_fresh: bool = False,
    event_calendar: Any = None,
    positioning: Any = None,
    htf: bool = False,
) -> RunSummary:
    """Run one model benchmark over one dataset and export its artifacts.

    ``event_calendar`` (Phase N, optional) is a dedicated
    :class:`signal_engine.event_calendar.EventCalendar`: its historical range
    is loaded for the dataset span, frozen to ``events.json`` for a
    deterministic replay, candles whose decision time falls in a blackout are
    skipped like the live gate, and analyzed candles carry the same event
    annotation the live prompt would show.

    Raises:
        BenchmarkError: for invalid benchmark configuration or a failed run.
    """
    verify_llm_temperature(llm)

    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)
    decisions_path = os.path.join(out, "decisions.jsonl")

    cache = DecisionCache(decisions_path)
    if force_fresh:
        cache._prepare_fresh()
    else:
        cache.load()

    start = time.perf_counter()
    blackout_fn = None
    analyze_fn = None
    if event_calendar is not None or positioning is not None or htf:
        from signal_engine.analysis import SignalAnalyzer

        analyze_base = SignalAnalyzer(llm).analyze
        period_ms = interval_to_ms(config.timeframe)
        if htf:
            # Innermost wrapper: computes the 4H bias from the window and
            # accepts the outer wirings' notes.
            analyze_base = build_htf_wiring(analyze_base=analyze_base)
        if positioning is not None:
            # Inner wrapper (closest to the analyzer): threads the funding
            # note + rate and forwards any outer event_note.
            analyze_base, frozen_pos = build_positioning_wiring(
                history=positioning,
                period_ms=period_ms,
                analyze_base=analyze_base,
            )
            pos_path = os.path.join(out, "positioning.json")
            with open(pos_path, "w", encoding="utf-8") as handle:
                json.dump(frozen_pos, handle, indent=2)
                handle.write("\n")
        if event_calendar is not None:
            blackout_fn, analyze_fn, frozen_events = build_event_wiring(
                event_calendar=event_calendar,
                hours=list(data.hour_candles),
                period_ms=period_ms,
                analyze_base=analyze_base,
            )
            events_path = os.path.join(out, "events.json")
            with open(events_path, "w", encoding="utf-8") as handle:
                json.dump(frozen_events, handle, indent=2)
                handle.write("\n")
        else:
            analyze_fn = analyze_base if positioning is not None else None
    provider_kwargs: dict[str, Any] = {
        "llm_config": llm,
        "cache": cache,
        "context_version": context_version,
        "dataset_id": dataset_id,
        "force_fresh": force_fresh,
    }
    if analyze_fn is not None:
        provider_kwargs["analyze_fn"] = analyze_fn
    provider = build_llm_decision_provider(**provider_kwargs)
    load_time = time.perf_counter() - start
    provider.stats.set_load_time(load_time)

    try:
        result = BacktestEngine(config, data, provider, blackout=blackout_fn).run()
    except Exception as exc:
        raise BenchmarkError(f"benchmark engine failed: {exc}") from exc

    statistics = BacktestStatistics.compute(config, result)
    stats_snapshot = provider.stats.snapshot()

    scope = compute_scope(
        provider=llm.provider,
        model=llm.model,
        base_url=llm.base_url,
        temperature=llm.temperature,
        timeout=llm.timeout,
        max_retries=llm.max_retries,
        max_tokens=llm.max_tokens,
        symbol=config.symbol,
        timeframe=config.timeframe,
        max_candles=config.max_candles,
        context_version=context_version,
        dataset_id=dataset_id,
    )
    records = [record for record in cache.records() if record.scope == scope]
    cached_hits = stats_snapshot.n_cached
    signal_quality = build_signal_quality(records, result, cached_hits=cached_hits)
    latency = stats_snapshot.to_dict()

    summary = build_summary(
        config=config,
        llm=llm,
        data=data,
        result=result,
        statistics=statistics,
        signal_quality=signal_quality,
        latency=latency,
        context_version=context_version,
        dataset_id=dataset_id,
    )

    files = _write_artifacts(result, statistics, out, decisions_path)
    events_path = os.path.join(out, "events.json")
    if os.path.exists(events_path):
        files["events"] = events_path
    pos_path = os.path.join(out, "positioning.json")
    if os.path.exists(pos_path):
        files["positioning"] = pos_path
    summary["files"] = files
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    files["summary"] = os.path.join(out, "summary.json")
    return RunSummary(
        model=llm.model,
        provider=llm.provider,
        result=result,
        statistics=statistics,
        summary=summary,
        files=files,
    )


def build_signal_quality(
    records: list[ModelDecision],
    result: BacktestResult,
    *,
    cached_hits: int = 0,
) -> dict[str, Any]:
    """Signal-quality metrics derived ONLY from the model's own records.

    The engine's direction counters include the invalid failure-fallback LONG
    (a FAILED call surfaces as an engine rejection), so the model's own intent
    is measured here. ``status`` reflects the model's live behavior: FAILED and
    INVALID outputs never count toward LONG/SHORT/WAIT signal ratios.
    ``cached_hits`` is the number of decisions replayed from the cache in this
    run (from the live latency snapshot).
    """
    ok = [r for r in records if r.status == DecisionStatus.OK]
    invalid = [r for r in records if r.status == DecisionStatus.INVALID]
    failed = [r for r in records if r.status == DecisionStatus.FAILED]
    analyzed = len(records)

    longs = [r for r in ok if r.decision == "LONG"]
    shorts = [r for r in ok if r.decision == "SHORT"]
    waits = [r for r in ok if r.decision == "WAIT"]

    confidences = [
        r.confidence
        for r in ok
        if r.decision in ("LONG", "SHORT") and r.confidence is not None
    ]
    buckets: dict[str, int] = {}
    for label in ("0-39", "40-69", "70-89", "90-100"):
        buckets[label] = 0
    for confidence in confidences:
        buckets[bucket_for_confidence(confidence)] += 1

    eligible = result.eligible
    return {
        "analyzed": analyzed,
        "ok": len(ok),
        "invalid": len(invalid),
        "failed": len(failed),
        "cached": cached_hits,
        "long": len(longs),
        "short": len(shorts),
        "wait": len(waits),
        "signal_count": len(longs) + len(shorts),
        "signal_frequency": (
            round((len(longs) + len(shorts)) / eligible, 6) if eligible else None
        ),
        "wait_ratio": round(len(waits) / analyzed, 6) if analyzed else None,
        "long_short_ratio": (
            round(len(longs) / len(shorts), 6) if shorts else None
        ),
        "avg_confidence": (
            round(sum(confidences) / len(confidences), 2) if confidences else None
        ),
        "confidence_buckets": buckets,
        "invalid_rate": round(len(invalid) / analyzed, 6) if analyzed else None,
        "failed_rate": round(len(failed) / analyzed, 6) if analyzed else None,
        "blocked": result.blocked_count,
        "note": (
            "FAILED/INVALID outputs are never counted as model signal choices. "
            "Engine counters (statistics.long_decisions/short_decisions) include "
            "the invalid failure-fallback LONG, so prefer these model_* fields "
            "for signal quality and the engine statistics for execution results."
        ),
    }


def build_summary(
    *,
    config: BacktestConfig,
    llm: LLMConfig,
    data: HistoricalData,
    result: BacktestResult,
    statistics: BacktestStatistics,
    signal_quality: dict[str, Any],
    latency: dict[str, Any],
    context_version: int,
    dataset_id: str | None,
) -> dict[str, Any]:
    """Assemble the human-readable ``summary.json`` for one model run."""
    hours = data.hour_candles
    return {
        "benchmark": {"phases": "13", "context_version": context_version},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "id": llm.model,
            "provider": llm.provider,
            "label": model_label(llm.model),
        },
        "configuration": {
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "execution_interval": config.execution_interval,
            "initial_balance": str(config.initial_balance),
            "margin_per_trade": str(config.margin_per_trade),
            "leverage": config.leverage,
            "fee_rate": str(config.fee_rate),
            "slippage_bps": str(config.slippage_bps),
            "funding_rates_enabled": config.funding_rates is not None,
            "min_candles": config.min_candles,
            "max_candles": config.max_candles,
            "dataset_id": dataset_id,
            "llm": {
                "base_url": llm.base_url,
                "temperature": llm.temperature,
                "timeout": llm.timeout,
                "max_retries": llm.max_retries,
                "max_tokens": llm.max_tokens,
            },
        },
        "determinism": {
            "temperature_zero": (
                llm.temperature == 0 if llm.temperature is not None else None
            ),
            "seed_supported": False,
            "note": (
                "Both models receive the identical closed-candle prompt/context. "
                "Temperature is configured to 0 for determinism where the "
                "provider supports it; the shared TradingAgents client does not "
                "expose an Ollama seed, so token-level determinism is NOT "
                "guaranteed. Confidence is a model self-report, never a quality "
                "metric."
            ),
        },
        "data": {
            "symbol": data.symbol,
            "timeframe": data.timeframe,
            "execution_interval": data.execution_interval,
            "hour_candles": len(data.hour_candles),
            "minute_candles": len(data.minute_candles),
            "first_hour_ts": hours[0].timestamp if hours else None,
            "last_hour_ts": hours[-1].timestamp if hours else None,
        },
        "engine_counts": {
            "eligible": result.eligible,
            "analyzed": result.analyzed,
            "wait_count": result.wait_count,
            "long_decisions": result.long_decisions,
            "short_decisions": result.short_decisions,
            "rejected_count": result.rejected_count,
            "blocked_count": result.blocked_count,
            "pending_created": result.pending_created,
            "entries_hit": result.entries_hit,
            "trades_completed": result.trades_completed,
            "open_at_end": result.open_at_end,
            "pending_at_end": result.pending_at_end,
            "ambiguous_total": result.ambiguous_total,
            "entry_ambiguity_count": result.entry_ambiguity_count,
            "exit_ambiguity_count": result.exit_ambiguity_count,
        },
        "statistics": statistics.to_dict(),
        "model_signal_quality": signal_quality,
        "latency": latency,
        "files": {},
    }


def _write_artifacts(
    result: BacktestResult,
    statistics: BacktestStatistics,
    out: str,
    decisions_path: str,
) -> dict[str, str]:
    result_path = export_result_json(
        os.path.join(out, "backtest_result.json"), result, statistics
    )
    trades_path = export_trades_csv(os.path.join(out, "backtest_trades.csv"), result)
    equity_path = _write_equity_curve(os.path.join(out, "equity_curve.csv"), result)
    return {
        "decisions": decisions_path,
        "backtest_result": result_path,
        "trades": trades_path,
        "equity_curve": equity_path,
    }


def _write_equity_curve(path: str | os.PathLike[str], result: BacktestResult) -> str:
    """Export the realized-equity curve (one row per balance update)."""
    target = os.fspath(path)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "equity"])
        for index, point in enumerate(result.equity_curve):
            writer.writerow([index, str(Decimal(point))])
    return target


__all__ = [
    "BenchmarkError",
    "RunSummary",
    "build_llm_decision_provider",
    "build_signal_quality",
    "build_summary",
    "run_benchmark",
    "verify_llm_temperature",
]
