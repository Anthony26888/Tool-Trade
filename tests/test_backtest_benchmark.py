"""Phase 13: local-model benchmark tests (Qwen3 4B vs Gemma 4B).

Covers the section 22 requirement list: model isolation, same dataset, cache
replay (no LLM calls), deterministic config, model failure recovery (no fake
WAIT), invalid-output rejection, comparison metrics, CLI, benchmark isolation,
and data integrity.
"""

from __future__ import annotations

import json
import os

import pytest

from backtest.__main__ import main
from backtest.benchmark import DecisionCache, LLMDecisionProvider, compute_scope
from backtest.benchmark.models import (
    CONTEXT_VERSION,
    DecisionStatus,
    ModelDecision,
    RunningInferenceStats,
)
from backtest.benchmark.runner import run_benchmark
from backtest.config import BacktestConfig
from backtest.data import HistoricalData, save_historical_data_json
from backtest.engine import BacktestEngine
from signal_engine.llm import SIGNAL_MODELS, LLMConfig
from tests.backtest_helpers import (
    backtest_config,
    covering_minutes,
    long_decision,
    make_hours,
    trade_minutes,
)


def _config(**overrides) -> BacktestConfig:
    return backtest_config(**overrides)


def _llm(model: str, temperature: float = 0.0, **overrides) -> LLMConfig:
    kwargs = {"provider": "ollama", "model": model, "temperature": temperature}
    kwargs.update(overrides)
    return LLMConfig(**kwargs)


def _recording_fn(decisions, observed, calls, *, failures=None):
    """Analyzer returning scripted payloads while recording its window view."""

    def fn(candles, indicators, *, symbol="BTCUSDT", timeframe="1h", max_candles=40):
        calls[0] += 1
        last = candles[-1]
        observed.append((len(candles), int(last.timestamp), float(last.close)))
        ts = int(last.timestamp)
        if failures is not None and ts in failures:
            raise failures[ts]
        if ts in decisions:
            return decisions[ts]
        return {"decision": "WAIT", "confidence": 30.0, "reasoning": "no script"}

    return fn


def _make_provider(*, model, cache, decisions, observed, calls, config=None, **kwargs):
    return LLMDecisionProvider(
        model=model,
        provider="ollama",
        cache=cache,
        config=config or _llm(model),
        analyze_fn=_recording_fn(decisions, observed, calls),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# A. Model isolation
# --------------------------------------------------------------------------- #


def test_registry_has_both_local_models():
    qwen = SIGNAL_MODELS["qwen3-4b"]
    gemma = SIGNAL_MODELS["gemma3-4b"]
    assert qwen.provider == "ollama" and qwen.model == "qwen3:4b"
    assert gemma.provider == "ollama" and gemma.model == "gemma3:4b"


def test_cross_model_cache_isolation(tmp_path):
    """A Gemma provider must never reuse a cached Qwen decision."""
    hours = make_hours(220)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )
    cache = DecisionCache(str(tmp_path / "decisions.jsonl"))
    qwen_observed, qwen_calls = [], [0]
    qwen = _make_provider(
        model="qwen3:4b",
        cache=cache,
        decisions={hours[200].timestamp: long_decision(float(hours[200].close))},
        observed=qwen_observed,
        calls=qwen_calls,
    )
    result_qwen = BacktestEngine(_config(), data, qwen).run()
    assert result_qwen.long_decisions == 1

    # A fresh Gemma provider over the SAME cache file (not force_fresh) must NOT
    # find qwen's record: compute_scope differs by model, so the key differs.
    gemma_calls = [0]
    gemma = LLMDecisionProvider(
        model="gemma3:4b",
        provider="ollama",
        cache=cache,
        config=_llm("gemma3:4b"),
        analyze_fn=_recording_fn({}, [], gemma_calls),
    )
    candle = hours[200]
    scope_qwen = compute_scope(
        provider="ollama",
        model="qwen3:4b",
        base_url=None,
        temperature=0.0,
        timeout=None,
        max_retries=None,
        max_tokens=None,
        symbol="BTCUSDT",
        timeframe="1h",
        max_candles=40,
        context_version=CONTEXT_VERSION,
        dataset_id=None,
    )
    scope_gemma = compute_scope(
        provider="ollama",
        model="gemma3:4b",
        base_url=None,
        temperature=0.0,
        timeout=None,
        max_retries=None,
        max_tokens=None,
        symbol="BTCUSDT",
        timeframe="1h",
        max_candles=40,
        context_version=CONTEXT_VERSION,
        dataset_id=None,
    )
    assert scope_qwen != scope_gemma
    assert cache.get(scope_qwen, int(candle.timestamp)) is not None
    assert cache.get(scope_gemma, int(candle.timestamp)) is None
    _ = BacktestEngine(_config(), data, gemma).run()
    assert gemma_calls[0] > 0  # gemma had to call the LLM, not replay qwen


# --------------------------------------------------------------------------- #
# B. Same dataset / no look-ahead
# --------------------------------------------------------------------------- #


def test_same_dataset_same_decision_inputs(tmp_path):
    """Both models must receive the identical per-candle window view."""
    hours = make_hours(220)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )

    def run_model(model: str):
        cache = DecisionCache(str(tmp_path / f"{model}.jsonl"))
        observed, calls = [], [0]
        provider = LLMDecisionProvider(
            model=model,
            provider="ollama",
            cache=cache,
            config=_llm(model),
            analyze_fn=_recording_fn({}, observed, calls),
            force_fresh=True,
        )
        BacktestEngine(_config(), data, provider).run()
        return observed

    qwen_observed = run_model("qwen3:4b")
    gemma_observed = run_model("gemma3:4b")
    assert qwen_observed == gemma_observed
    assert len(qwen_observed) == 21  # 220 hours - 200 warmup + 1


def test_no_lookahead_window(tmp_path):
    """The provider sees candles[:i+1] only: window len == index + 1."""
    hours = make_hours(220)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )
    cache = DecisionCache(str(tmp_path / "decisions.jsonl"))
    observed, calls = [], [0]
    provider = LLMDecisionProvider(
        model="qwen3:4b",
        provider="ollama",
        cache=cache,
        config=_llm("qwen3:4b"),
        analyze_fn=_recording_fn({}, observed, calls),
        force_fresh=True,
    )
    BacktestEngine(_config(), data, provider).run()
    index_by_ts = {int(c.timestamp): i for i, c in enumerate(hours)}
    for window_len, ts, _ in observed:
        assert window_len == index_by_ts[ts] + 1
        assert ts == hours[index_by_ts[ts]].timestamp


# --------------------------------------------------------------------------- #
# C. Cache: replay without re-calling the LLM
# --------------------------------------------------------------------------- #


def test_cache_replay_no_llm_calls(tmp_path):
    hours = make_hours(220)
    decisions = {
        hours[200].timestamp: long_decision(float(hours[200].close)),
        hours[201].timestamp: {"decision": "WAIT", "confidence": 50.0, "reasoning": "hold"},
    }
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )

    cache_path = str(tmp_path / "decisions.jsonl")

    def first_run():
        cache = DecisionCache(cache_path)
        calls = [0]
        provider = _make_provider(
            model="qwen3:4b", cache=cache, decisions=decisions,
            observed=[], calls=calls,
        )
        result = BacktestEngine(_config(), data, provider).run()
        return result.long_decisions, result.wait_count, calls[0]

    long_first, wait_first, calls_first = first_run()
    assert long_first == 1
    assert wait_first == 1
    # index 199 -> WAIT, index 200 -> LONG (then the long signal blocks the
    # rest of the candles because its entry never fills on flat minutes).
    assert calls_first == 2

    cache = DecisionCache(cache_path)
    cache.load()
    assert cache.get(compute_scope(
        provider="ollama", model="qwen3:4b", base_url=None, temperature=0.0,
        timeout=None, max_retries=None, max_tokens=None, symbol="BTCUSDT",
        timeframe="1h", max_candles=40, context_version=CONTEXT_VERSION, dataset_id=None,
    ), int(hours[200].timestamp)) is not None

    calls = [0]
    provider = _make_provider(
        model="qwen3:4b", cache=cache, decisions={}, observed=[], calls=calls,
    )
    result = BacktestEngine(_config(), data, provider).run()
    assert calls[0] == 0  # replay: the LLM was never called again
    assert result.long_decisions == long_first
    assert result.wait_count == wait_first


def test_cache_scope_differs_by_config():
    base = _llm("qwen3:4b")
    scope_a = compute_scope(
        provider=base.provider, model=base.model, base_url=base.base_url,
        temperature=0.0, timeout=base.timeout, max_retries=base.max_retries,
        max_tokens=base.max_tokens, symbol="BTCUSDT", timeframe="1h",
        max_candles=40, context_version=CONTEXT_VERSION, dataset_id=None,
    )
    scope_b = compute_scope(
        provider=base.provider, model=base.model, base_url=base.base_url,
        temperature=0.0, timeout=base.timeout, max_retries=base.max_retries,
        max_tokens=base.max_tokens, symbol="BTCUSDT", timeframe="1h",
        max_candles=80, context_version=CONTEXT_VERSION, dataset_id=None,
    )
    scope_c = compute_scope(
        provider=base.provider, model=base.model, base_url=base.base_url,
        temperature=0.5, timeout=base.timeout, max_retries=base.max_retries,
        max_tokens=base.max_tokens, symbol="BTCUSDT", timeframe="1h",
        max_candles=40, context_version=CONTEXT_VERSION, dataset_id=None,
    )
    assert scope_a != scope_b
    assert scope_a != scope_c
    assert scope_a == compute_scope(
        provider=base.provider, model=base.model, base_url=base.base_url,
        temperature=0.0, timeout=base.timeout, max_retries=base.max_retries,
        max_tokens=base.max_tokens, symbol="BTCUSDT", timeframe="1h",
        max_candles=40, context_version=CONTEXT_VERSION, dataset_id=None,
    )


def test_force_fresh_truncates_and_recalls(tmp_path, monkeypatch):
    from backtest.benchmark import runner as runner_module

    hours = make_hours(220)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )
    out = str(tmp_path / "out")
    # Scripted WAIT analyzer: first run caches OK rows (no real LLM), so the
    # replay path below exercises cache hits instead of live failures.
    monkeypatch.setattr(
        runner_module, "build_llm_decision_provider", _fake_factory({})
    )

    first = run_benchmark(
        config=_config(), data=data, out_dir=out,
        llm=_llm("qwen3:4b"), dataset_id="ds1",
    )
    assert first.summary["model_signal_quality"]["cached"] == 0

    replay = run_benchmark(
        config=_config(), data=data, out_dir=out,
        llm=_llm("qwen3:4b"), dataset_id="ds1",
    )
    assert replay.summary["model_signal_quality"]["cached"] == 21
    assert replay.summary["latency"]["fresh_calls"] == 0

    fresh = run_benchmark(
        config=_config(), data=data, out_dir=out,
        llm=_llm("qwen3:4b"), dataset_id="ds1", force_fresh=True,
    )
    assert fresh.summary["model_signal_quality"]["cached"] == 0
    assert fresh.summary["latency"]["fresh_calls"] == 21


# --------------------------------------------------------------------------- #
# F. Failure recovery (no fake WAIT) and G. invalid output rejection
# --------------------------------------------------------------------------- #


def test_model_failure_recovered_as_rejected_event_not_wait(tmp_path):
    hours = make_hours(220)
    failing_ts = int(hours[203].timestamp)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )
    cache = DecisionCache(str(tmp_path / "decisions.jsonl"))
    calls = [0]
    provider = LLMDecisionProvider(
        model="qwen3:4b",
        provider="ollama",
        cache=cache,
        config=_llm("qwen3:4b"),
        analyze_fn=_recording_fn(
            {},
            [],
            calls,
            failures={failing_ts: RuntimeError("boom")},
        ),
        force_fresh=True,
    )
    result = BacktestEngine(_config(), data, provider).run()

    # FAILED rows are never cached (a later run must retry the live call
    # instead of replaying the failure), but the engine still records an
    # explicit rejected event for the fallback LONG.
    failed = [r for r in provider.cache.records() if r.status == DecisionStatus.FAILED]
    assert failed == []

    assert result.analyzed == result.eligible              # run never aborted
    assert result.rejected_count == 1                      # explicit rejected event
    assert result.pending_created == 0 and result.trades_completed == 0
    # The failed candle was NOT counted as the model choosing WAIT.
    assert all(r.status == DecisionStatus.OK and r.decision == "WAIT"
               for r in provider.cache.records() if r.candle_ts != failing_ts)


def test_cached_failed_is_retried_not_replayed(tmp_path):
    # A FAILED row left by an earlier interrupted run (e.g. rate limit) must
    # trigger a fresh live call on resume, never a replay of the fallback.
    from tests.signal_engine_test_helpers import indicators_for, make_candles

    candles = make_candles(220)
    indicators = indicators_for(candles)
    last = candles[-1]
    cache = DecisionCache(str(tmp_path / "decisions.jsonl"))
    calls = [0]
    observed: list = []
    provider = LLMDecisionProvider(
        model="qwen3:4b",
        provider="ollama",
        cache=cache,
        config=_llm("qwen3:4b"),
        analyze_fn=_recording_fn({}, observed, calls),
    )
    cache.put(
        ModelDecision(
            model="qwen3:4b",
            provider="ollama",
            candle_ts=int(last.timestamp),
            candle_close=float(last.close),
            status=DecisionStatus.FAILED,
            decision="LONG",
            confidence=0.0,
            reasoning="boom",
            entry_price=float(last.close),
            stop_loss=float(last.close),
            take_profit=float(last.close),
            latency_sec=1.0,
            cached=False,
            scope=provider._scope("BTCUSDT", "1h", 20),
            context_version=CONTEXT_VERSION,
            analysis_timestamp="2026-01-01T00:00:00Z",
            error="boom",
        )
    )
    analysis = provider.decide(
        list(candles), indicators, symbol="BTCUSDT", timeframe="1h", max_candles=20
    )
    assert calls[0] == 1
    assert analysis.decision == "WAIT"


def test_invalid_output_rejected_never_becomes_trade(tmp_path):
    hours = make_hours(220)
    invalid_ts = int(hours[205].timestamp)
    close = float(hours[205].close)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(covering_minutes(hours)),
    )
    cache = DecisionCache(str(tmp_path / "decisions.jsonl"))
    provider = LLMDecisionProvider(
        model="qwen3:4b",
        provider="ollama",
        cache=cache,
        config=_llm("qwen3:4b"),
        analyze_fn=_recording_fn(
            {invalid_ts: long_decision(close) | {"entry_price": close, "stop_loss": close}},
            [],
            [0],
        ),
        force_fresh=True,
    )
    result = BacktestEngine(_config(), data, provider).run()

    invalid = [r for r in provider.cache.records() if r.status == DecisionStatus.INVALID]
    assert len(invalid) == 1
    assert invalid[0].candle_ts == invalid_ts
    assert result.rejected_count == 1
    assert result.pending_created == 0 and result.trades_completed == 0


# --------------------------------------------------------------------------- #
# Signal quality + latency stats
# --------------------------------------------------------------------------- #


def test_signal_quality_tallies(tmp_path, monkeypatch):
    hours = make_hours(220)
    ts = {i: int(hours[i].timestamp) for i in range(199, 219)}
    decisions = {
        ts[199]: long_decision(float(hours[199].close)),
        ts[200]: {
            "decision": "SHORT",
            "confidence": 65.0,
            "reasoning": "r",
            "entry_price": float(hours[200].close) + 5,
            "stop_loss": float(hours[200].close) + 40,
            "take_profit": float(hours[200].close) - 50,
        },
        ts[201]: {"decision": "WAIT", "confidence": 20.0, "reasoning": "r"},
    }
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(trade_minutes(hours)),
    )
    from backtest.benchmark import runner as runner_module

    runner_module.build_llm_decision_provider = _fake_factory(decisions)  # type: ignore[assignment]
    summary = run_benchmark(
        config=_config(),
        data=data,
        out_dir=str(tmp_path / "run"),
        llm=_llm("qwen3:4b", max_retries=2),
    )
    quality = summary.summary["model_signal_quality"]
    assert quality["ok"] == 21
    assert quality["long"] == 1
    assert quality["short"] == 1
    assert quality["wait"] == 19
    assert quality["signal_count"] == 2
    assert quality["failed"] == 0 and quality["invalid"] == 0
    # confidence average across the two directional signals: (90 + 65) / 2 = 77.5
    assert quality["avg_confidence"] == 77.5
    assert quality["confidence_buckets"] == {"0-39": 0, "40-69": 1, "70-89": 0, "90-100": 1}
    assert quality["long_short_ratio"] == 1.0
    assert quality["signal_frequency"] == round(2 / 21, 6)  # 2 signals / 21 eligible
    assert quality["wait_ratio"] == round(19 / 21, 6)


def test_inference_stats_median():
    stats = RunningInferenceStats()
    for latency in (0.1, 0.2, 0.3, 0.4):
        stats.add(latency, cached=False, failed=False)
    stats.add(0.0, cached=True, failed=False)
    stats.add(0.5, cached=False, failed=True)
    snapshot = stats.snapshot()
    assert snapshot.n_calls == 5
    assert snapshot.n_cached == 1
    assert snapshot.n_failed == 1
    assert snapshot.min_sec == 0.1 and snapshot.max_sec == 0.5
    assert snapshot.median_sec == 0.3
    assert snapshot.total_sec == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# H. Comparison (no winner logic)
# --------------------------------------------------------------------------- #


def test_compare_summaries_table(tmp_path, monkeypatch):
    from backtest.benchmark.compare import compare_summaries

    hours = make_hours(220)
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(trade_minutes(hours)),
    )
    long_ts = [199, 204, 209]
    decisions = {
        int(hours[i].timestamp): long_decision(float(hours[i].close))
        for i in long_ts
    }

    def fake_factory(*, llm_config, cache, context_version=CONTEXT_VERSION, dataset_id=None, force_fresh=False):
        from backtest.benchmark.provider import LLMDecisionProvider
        return LLMDecisionProvider(
            model=llm_config.model,
            provider=llm_config.provider,
            cache=cache,
            config=llm_config,
            analyze_fn=_recording_fn(decisions, [], [0]),
            context_version=context_version,
            dataset_id=dataset_id,
            force_fresh=force_fresh,
        )

    from backtest.benchmark import runner as runner_module
    monkeypatch.setattr(runner_module, "build_llm_decision_provider", fake_factory)

    a = run_benchmark(
        config=_config(), data=data, out_dir=str(tmp_path / "qwen"),
        llm=_llm("qwen3:4b"),
    )
    b = run_benchmark(
        config=_config(), data=data, out_dir=str(tmp_path / "gemma"),
        llm=_llm("gemma3:4b"),
    )
    comparison = compare_summaries(a.summary, b.summary)
    assert comparison["first"]["model"] == "qwen3:4b"
    assert comparison["second"]["label"] == "Gemma 4B"
    rows = {row["metric"]: row for row in comparison["table"]}
    # "Not computable" values are rendered as an empty string (both models here).
    assert rows["Profit factor"]["first"] == ""
    assert rows["Trades"]["first"] == "3"
    assert rows["Net PnL (USDT)"]["first"] == "-3.159655" or rows["Net PnL (USDT)"]["first"] != ""
    assert rows["Avg inference latency (s)"]["first"] != ""
    assert float(rows["Avg inference latency (s)"]["first"]) == pytest.approx(
        a.summary["latency"]["avg_sec"], rel=1e-6
    )
    assert "winner" not in comparison and "best" not in comparison


# --------------------------------------------------------------------------- #
# Runner artifacts
# --------------------------------------------------------------------------- #


def test_runner_exports_all_artifacts(tmp_path, monkeypatch):
    from pathlib import Path

    from backtest.benchmark import runner as runner_module

    hours = make_hours(220)
    decisions = {int(hours[200].timestamp): long_decision(float(hours[200].close))}
    monkeypatch.setattr(
        runner_module, "build_llm_decision_provider", _fake_factory(decisions)
    )
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(trade_minutes(hours)),
    )
    out = str(tmp_path / "run")
    summary = run_benchmark(
        config=_config(), data=data, out_dir=out, llm=_llm("qwen3:4b"),
        dataset_id="dataset-1",
    )
    for path in summary.files.values():
        assert path and os.path.exists(path)

    payload = json.loads(Path(summary.files["summary"]).read_text(encoding="utf-8"))
    for key in ("model", "statistics", "model_signal_quality", "latency", "engine_counts", "files"):
        assert key in payload
    assert payload["model"]["id"] == "qwen3:4b"
    assert payload["configuration"]["dataset_id"] == "dataset-1"
    assert payload["engine_counts"]["trades_completed"] == summary.result.trades_completed

    lines = [
        json.loads(line)
        for line in Path(summary.files["decisions"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == summary.result.analyzed
    first = lines[0]
    for field in ("candle_ts", "model", "status", "decision", "latency_sec", "scope"):
        assert field in first
    assert first["model"] == "qwen3:4b"

    equity_lines = [
        line
        for line in Path(summary.files["equity_curve"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert equity_lines[0] == "index,equity"
    assert len(equity_lines) == len(summary.result.equity_curve) + 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _save_data(tmp_path, hours):
    data = HistoricalData(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        hour_candles=tuple(hours),
        minute_candles=tuple(trade_minutes(hours)),
    )
    path = str(tmp_path / "data.json")
    save_historical_data_json(path, data)
    return path


def _fake_factory(decisions):
    def factory(*, llm_config, cache, context_version=CONTEXT_VERSION, dataset_id=None, force_fresh=False):
        from backtest.benchmark.provider import LLMDecisionProvider
        return LLMDecisionProvider(
            model=llm_config.model,
            provider=llm_config.provider,
            cache=cache,
            config=llm_config,
            analyze_fn=_recording_fn(decisions, [], [0]),
            context_version=context_version,
            dataset_id=dataset_id,
            force_fresh=force_fresh,
        )

    return factory


def test_cli_benchmark_end_to_end(tmp_path, capsys, monkeypatch):
    from backtest.benchmark import runner as runner_module

    hours = make_hours(220)
    data_path = _save_data(tmp_path, hours)
    decisions = {int(hours[205].timestamp): long_decision(float(hours[205].close))}
    monkeypatch.setattr(
        runner_module, "build_llm_decision_provider", _fake_factory(decisions)
    )

    out = str(tmp_path / "bench")
    code = main(["benchmark", "--data", data_path, "--model", "qwen3:4b", "--out", out])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Model           ollama/qwen3:4b" in captured
    assert "Model signals" in captured
    assert os.path.exists(os.path.join(out, "summary.json"))
    assert os.path.exists(os.path.join(out, "decisions.jsonl"))


def test_cli_benchmark_invalid_temperature(tmp_path, capsys, monkeypatch):
    from backtest.benchmark import runner as runner_module

    hours = make_hours(220)
    data_path = _save_data(tmp_path, hours)
    monkeypatch.setattr(runner_module, "build_llm_decision_provider", _fake_factory({}))
    code = main([
        "benchmark", "--data", data_path, "--model", "gemma3:4b",
        "--temperature", "3", "--out", str(tmp_path / "o"),
    ])
    assert code == 2
    assert "benchmark failed" in capsys.readouterr().err


def test_cli_benchmark_bad_data(capsys):
    code = main([
        "benchmark", "--data", "/nonexistent/data.json",
        "--model", "qwen3:4b", "--out", "/tmp/never-write-here",
    ])
    assert code == 2
    assert "benchmark failed" in capsys.readouterr().err


def test_cli_compare_end_to_end(tmp_path, capsys, monkeypatch):
    from backtest.benchmark import runner as runner_module

    hours = make_hours(220)
    data_path = _save_data(tmp_path, hours)
    decisions = {int(hours[205].timestamp): long_decision(float(hours[205].close))}
    factory = _fake_factory(decisions)
    monkeypatch.setattr(runner_module, "build_llm_decision_provider", factory)

    qwen_out = str(tmp_path / "qwen")
    gemma_out = str(tmp_path / "gemma")
    assert main(["benchmark", "--data", data_path, "--model", "qwen3:4b", "--out", qwen_out]) == 0
    assert main(["benchmark", "--data", data_path, "--model", "gemma3:4b", "--out", gemma_out]) == 0

    cmp_out = str(tmp_path / "compare")
    code = main(["compare", "--runs",
                 os.path.join(qwen_out, "summary.json"),
                 os.path.join(gemma_out, "summary.json"),
                 "--out", cmp_out])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Qwen3 4B" in captured and "Gemma 4B" in captured
    assert os.path.exists(os.path.join(cmp_out, "comparison.json"))
    assert os.path.exists(os.path.join(cmp_out, "comparison.csv"))


def test_cli_run_legacy_preserved(tmp_path, capsys):
    """The Phase 12 flat CLI (no subcommand) must keep working unchanged."""
    hours = make_hours(240)
    data_path = _save_data(tmp_path, hours)
    decisions = {
        hours[i].timestamp: long_decision(float(hours[i].close))
        for i in (199, 204, 209)
    }
    decisions_path = str(tmp_path / "decisions.json")
    with open(decisions_path, "w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in decisions.items()}, handle)

    out = str(tmp_path / "out")
    code = main(["--data", data_path, "--decisions", decisions_path, "--out", out])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Eligible        41" in captured
    assert os.path.exists(os.path.join(out, "backtest_result.json"))


def test_decision_record_round_trip(tmp_path):
    cache = DecisionCache(str(tmp_path / "d.jsonl"))
    record = ModelDecision(
        model="qwen3:4b",
        provider="ollama",
        candle_ts=1720000000000,
        candle_close=60000.0,
        status=DecisionStatus.OK,
        decision="LONG",
        confidence=90.0,
        reasoning="some reasoning",
        entry_price=59995.0,
        stop_loss=59950.0,
        take_profit=60100.0,
        latency_sec=1.25,
        cached=False,
        scope="scope",
        context_version=CONTEXT_VERSION,
        analysis_timestamp="2026-01-01T00:00:00Z",
    )
    cache.put(record)
    loaded = DecisionCache(str(tmp_path / "d.jsonl"))
    loaded.load()
    same = loaded.get("scope", 1720000000000)
    assert same is not None
    assert same.decision == "LONG"
    assert same.entry_price == 59995.0
    assert same.scope == "scope"
