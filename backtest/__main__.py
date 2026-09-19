"""CLI for ``python -m backtest ...``.

Subcommands
-----------
- (default / ``run``)   deterministic Phase 12 backtest from a JSON data file and
  a replay decision file — fully offline, no LLM.
- ``benchmark``         Phase 13: run ONE local model over a dataset/config and
  write per-model result files (decisions.jsonl, summary.json, trades CSV,
  equity curve). Sequential by design: one model per invocation. Re-runs with
  the same configuration scope replay recorded decisions instead of re-calling
  the LLM unless ``--force-fresh`` is passed.
- ``compare``           Phase 13: build a side-by-side Qwen3 4B vs Gemma 4B
  metrics table (no automatic winner selection).

Exit codes: 0 on success, 2 when the run could not be completed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal, InvalidOperation

from signal_engine.llm import LLMConfig

from .config import BacktestConfig, BacktestError, BacktestMetricError
from .data import JsonFileHistoricalDataProvider
from .engine import BacktestEngine
from .export import export_result_json, export_trades_csv
from .metrics import BacktestStatistics
from .provider import ReplayDecisionProvider

SUBCOMMANDS = ("run", "benchmark", "compare")


def _decimal_arg(text: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal {text!r}") from exc


def _float_arg(text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number {text!r}") from exc


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--execution-interval", default="1m")
    parser.add_argument("--initial-balance", type=_decimal_arg, default=Decimal("1000"))
    parser.add_argument("--margin-per-trade", type=_decimal_arg, default=Decimal("50"))
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--fee-rate", type=_decimal_arg, default=Decimal("0.0004"))
    parser.add_argument("--slippage-bps", type=_decimal_arg, default=Decimal("0"))
    parser.add_argument(
        "--min-candles",
        type=int,
        default=200,
        help="warmup candles before the first eligible decision (default 200 for EMA200)",
    )


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest run",
        description="Deterministic, isolated backtest of the BTCUSDT strategy (Phase 12).",
    )
    parser.add_argument("--data", required=True, help="backtest data file (JSON)")
    parser.add_argument(
        "--decisions",
        required=True,
        help="replay decision file (JSON: candle_open_time_ms -> decision)",
    )
    parser.add_argument(
        "--out",
        default="results/backtest",
        help="output directory for result.json and trades.csv",
    )
    _add_config_args(parser)
    return parser


def _config_from_args(args) -> BacktestConfig:
    return BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        execution_interval=args.execution_interval,
        initial_balance=args.initial_balance,
        margin_per_trade=args.margin_per_trade,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        min_candles=args.min_candles,
    )


def _load_decisions(path: str) -> dict[int, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read decisions file {path!r}: {exc}") from None
    if not isinstance(payload, dict):
        raise SystemExit("decisions file must contain a JSON object (ts -> decision)")
    parsed: dict[int, object] = {}
    for key, value in payload.items():
        try:
            timestamp = int(key)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"decision key {key!r} is not an epoch-ms int") from exc
        parsed[timestamp] = value
    return parsed


def _cmd_run(argv: list[str]) -> int:
    parser = _build_run_parser()
    args = parser.parse_args(argv)

    config = _config_from_args(args)
    try:
        data = JsonFileHistoricalDataProvider(args.data).get_historical_data()
        decisions = _load_decisions(args.decisions)
        result = BacktestEngine(config, data, ReplayDecisionProvider(decisions)).run()
        statistics = BacktestStatistics.compute(config, result)
    except BacktestError as exc:
        print(f"backtest failed: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    result_path = export_result_json(
        os.path.join(args.out, "backtest_result.json"), result, statistics
    )
    trades_path = export_trades_csv(os.path.join(args.out, "backtest_trades.csv"), result)

    print(
        f"Eligible        {result.eligible}\n"
        f"Analyzed        {result.analyzed} (WAIT {result.wait_count} / "
        f"LONG {result.long_decisions} / SHORT {result.short_decisions})\n"
        f"Blocked (1 pos) {result.blocked_count}\n"
        f"Rejected        {result.rejected_count}\n"
        f"Marginal pending/trades: created {result.pending_created}, "
        f"entries hit {result.entries_hit}, completed {result.trades_completed}\n"
        f"Ambiguous candles     {result.ambiguous_total} "
        f"(entry {result.entry_ambiguity_count} / exit {result.exit_ambiguity_count})\n"
        f"Trades/Wins/Losses    {result.trades_completed} / "
        f"{statistics.wins} / {statistics.losses}\n"
        f"Win rate              {_fmt_dec(statistics.win_rate)}\n"
        f"Net PnL                {statistics.net_pnl}\n"
        f"Profit factor          {_fmt_dec(statistics.profit_factor)}\n"
        f"Max drawdown           {statistics.max_drawdown} "
        f"({_fmt_dec(statistics.max_drawdown_pct)})\n"
        f"Final balance          {statistics.final_balance}\n"
        f"Wrote {result_path}\nWrote {trades_path}"
    )
    return 0


def _build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest benchmark",
        description="Phase 13: run ONE local LLM model over a dataset (sequential).",
    )
    parser.add_argument("--data", required=True, help="backtest data file (JSON)")
    parser.add_argument(
        "--model", required=True, help="Ollama/local model id, e.g. qwen3:4b or gemma3:4b"
    )
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--base-url", default=None, help="provider base URL override")
    parser.add_argument("--temperature", type=_float_arg, default=0.0, help="LLM temperature")
    parser.add_argument("--timeout", type=_float_arg, default=None, help="request timeout (s)")
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--out",
        default="results/benchmark",
        help="output directory (one subdirectory per model)",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="optional cache-scope discriminant (e.g. the data file label)",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="ignore cached decisions and re-call the LLM (truncates decisions.jsonl)",
    )
    parser.add_argument("--context-version", type=int, default=None)
    parser.add_argument(
        "--event-calendar",
        action="store_true",
        help="Phase N: load historical FOMC/CPI/NFP blackouts for the dataset "
        "span (frozen to events.json), skip blackout candles like the live "
        "gate, and annotate analyzed candles with the event note.",
    )
    parser.add_argument(
        "--positioning",
        action="store_true",
        help="Phase P1: fetch historical funding + long/short history for the "
        "dataset span (frozen to positioning.json) and thread it into prompts "
        "with the funding guardrail, mirroring live.",
    )
    parser.add_argument(
        "--htf",
        action="store_true",
        help="Phase P2: roll the dataset's 1H candles up to 4H, classify the "
        "trend bias with the live code path, and veto counter-regime entries.",
    )
    _add_config_args(parser)
    return parser


def _cmd_benchmark(argv: list[str]) -> int:
    from .benchmark import run_benchmark
    from .benchmark.models import CONTEXT_VERSION

    parser = _build_benchmark_parser()
    args = parser.parse_args(argv)

    config = _config_from_args(args)
    llm = LLMConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    out_dir = args.out
    event_calendar = None
    if args.event_calendar:
        from signal_engine.event_calendar import EventCalendar

        event_calendar = EventCalendar()
    positioning = None
    if args.positioning:
        positioning = True  # resolved to history after data loads
    try:
        data = JsonFileHistoricalDataProvider(args.data).get_historical_data()
        if positioning is True:
            from binance.client import BinanceFuturesClient
            from binance.positioning import PositioningHistory

            hours = list(data.hour_candles)
            positioning = (
                PositioningHistory.fetch(
                    BinanceFuturesClient(),
                    config.symbol,
                    int(hours[0].timestamp),
                    int(hours[-1].timestamp),
                )
                if hours
                else PositioningHistory()
            )
        summary = run_benchmark(
            config=config,
            data=data,
            out_dir=out_dir,
            llm=llm,
            context_version=args.context_version
            if args.context_version is not None
            else CONTEXT_VERSION,
            dataset_id=args.dataset_id,
            force_fresh=args.force_fresh,
            event_calendar=event_calendar,
            positioning=positioning,
            htf=args.htf,
        )
    except (BacktestError, BacktestMetricError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 2

    result = summary.result
    statistics = summary.statistics
    quality = summary.summary["model_signal_quality"]
    latency = summary.summary["latency"]
    resolved_version = (
        args.context_version if args.context_version is not None else CONTEXT_VERSION
    )
    print(
        f"Model           {llm.provider}/{llm.model} ({resolved_version})\n"
        f"Eligible        {result.eligible}\n"
        f"Analyzed        {result.analyzed} (WAIT {result.wait_count} / "
        f"LONG {result.long_decisions} / SHORT {result.short_decisions}) [engine]\n"
        f"Model signals   LONG {quality['long']} / SHORT {quality['short']} / "
        f"WAIT {quality['wait']} (OK {quality['ok']}, invalid {quality['invalid']}, "
        f"failed {quality['failed']})\n"
        f"Blocked (1 pos) {result.blocked_count}\n"
        f"Rejected        {result.rejected_count}\n"
        f"Easy trades     {result.trades_completed} (wins {statistics.wins} / "
        f"losses {statistics.losses}, win rate {_fmt_dec(statistics.win_rate)})\n"
        f"Net PnL          {statistics.net_pnl} "
        f"({_fmt_dec(statistics.total_return_pct)})\n"
        f"Profit factor    {_fmt_dec(statistics.profit_factor)}\n"
        f"Max drawdown     {statistics.max_drawdown} "
        f"({_fmt_dec(statistics.max_drawdown_pct)})\n"
        f"Latency          avg {_fmt_dec(latency.get('avg_sec'))} s, "
        f"median {_fmt_dec(latency.get('median_sec'))} s, "
        f"failed {latency.get('failed_calls')}\n"
        f"Wrote {summary.files['summary']}\n"
        f"Wrote {summary.files['decisions']}\n"
        f"Wrote {summary.files['trades']}"
    )
    return 0


def _build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest compare",
        description="Phase 13: side-by-side benchmark comparison (no winner logic).",
    )
    parser.add_argument(
        "--runs",
        nargs=2,
        metavar=("SUMMARY_A", "SUMMARY_B"),
        required=True,
        help="the two summary.json files from each model benchmark run",
    )
    parser.add_argument(
        "--out",
        default="results/benchmark/compare",
        help="output directory for comparison.json and comparison.csv",
    )
    return parser


def _load_summary(path: str) -> dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read summary file {path!r}: {exc}") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"summary file {path!r} must contain a JSON object")
    return payload


def _cmd_compare(argv: list[str]) -> int:
    from .benchmark.compare import compare_summaries, write_comparison

    parser = _build_compare_parser()
    args = parser.parse_args(argv)

    first = _load_summary(args.runs[0])
    second = _load_summary(args.runs[1])
    comparison = compare_summaries(first, second)
    json_path, csv_path = write_comparison(comparison, args.out)

    first_model = comparison["first"]["label"]
    second_model = comparison["second"]["label"]
    header = f"{'Metric':<24} {first_model:<14} {second_model:<14}"
    print(header)
    print("-" * len(header))
    for row in comparison["table"]:
        print(f"{row['metric']:<24} {row['first']:<14} {row['second']:<14}")
    print(f"\nWrote {json_path}\nWrote {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in SUBCOMMANDS:
        command, rest = args_list[0], args_list[1:]
        if command == "benchmark":
            return _cmd_benchmark(rest)
        if command == "compare":
            return _cmd_compare(rest)
        return _cmd_run(rest)
    # Legacy flat invocation (no subcommand) keeps the Phase 12 interface.
    return _cmd_run(args_list)


def _fmt_dec(value) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
