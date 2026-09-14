# Backtest Engine (Phase 12)

Deterministic, isolated backtest of the BTCUSDT Futures signal strategy. The
engine replays the production lifecycle — AI decision, `PENDING_ENTRY`, `OPEN`,
`TP`/`SL` close — against historical candles and reuses the Phase 8 money model
(margin x leverage, fee rate).

> **Disclaimer**: Backtest performance does NOT guarantee live profitability.
> This backtester is for research and signal evaluation only. It never places
> real orders, never accesses the Binance trading API, and should not be used
> to make financial decisions.

## Isolation

The backtest never:

- touches SQLite or any production trading state,
- calls the Binance API or any network endpoint,
- invokes an LLM by default (no API keys are ever read),
- writes anywhere except an explicit output path you choose.

## Running

```bash
python -m backtest \
    --data results/backtest/data_btcusdt_1h.json \
    --decisions results/backtest/decisions.json \
    --out results/backtest
```

`--data` is a JSON file produced by `backtest.data.save_historical_data_json`
(e.g. exported from a previous real run), `--decisions` is a JSON mapping of
`1H candle open time (epoch ms)` to a decision value:

```json
{
  "1728000000": "LONG",
  "1728003600": "WAIT",
  "1728007200": {"decision": "SHORT", "confidence": 70, "reasoning": "..."}
}
```

Decision values may be a `SignalAnalysis`, a `SignalDecisionModel`, a dict, or
a plain `"LONG"` / `"SHORT"` / `"WAIT"` string. Missing candle keys produce
`WAIT`.

## Configuration

`BacktestConfig` (all money is `Decimal`):

| Field                | Default | Meaning                                         |
| -------------------- | ------- | ----------------------------------------------- |
| `symbol`             | BTCUSDT | futures pair                                     |
| `timeframe`          | 1h      | decision (AI analysis) timeframe                 |
| `execution_interval` | 1m      | entry/TP/SL confirmation timeframe               |
| `initial_balance`    | 1000    | starting balance (USDT)                          |
| `margin_per_trade`   | 50      | margin per position (USDT)                       |
| `leverage`           | 10      | leverage multiplier                              |
| `fee_rate`           | 0.0004  | fee rate as a fraction (0.04%)                   |
| `slippage_bps`       | 0       | slippage in basis points applied to execution    |
| `funding_rates`      | None    | `{epoch_ms: rate}` applied to open-position notional |

## No look-ahead

The core guarantee (`AGENTS.md` section 10):

- The decision for candle `i` sees only `candles[:i+1]` — indicators are
  recomputed from that exact window every time.
- Decision time is `open_time + interval_ms` (the candle is closed). The
  `close_time` field is never used as the decision instant.
- Entry/TP/SL confirmation uses closed 1m candles with `timestamp >=`
  decision time; 1m candles inside the signal candle are never used.
- A candle touching both TP and SL is `AMBIGUOUS`: nothing is closed and
  monitoring continues. The intrabar order is never guessed from OHLC.

## Money model

- `notional = margin_per_trade * leverage`
- `quantity = notional / entry_execution_price`
- `gross = (exit - entry) * quantity` (LONG) or `(entry - exit) * quantity`
  (SHORT), using slippage-adjusted execution prices
- `net = gross - entry_fee - exit_fee - funding`
- balance updates by `net` only; equity equals balance (no mark-to-market)

Slippage only affects execution price, never touch detection. The total
slippage impact is reported as an informational metric and never double-counted.

## Writing your own data

```python
from backtest.data import HistoricalData, save_historical_data_json
from binance.market_data import Candle

data = HistoricalData(
    symbol="BTCUSDT", timeframe="1h", execution_interval="1m",
    hour_candles=tuple(hour_candles),   # closed Candle objects
    minute_candles=tuple(minute_candles),
)
save_historical_data_json("data_btcusdt_1h.json", data)
```

## Decision providers

- `WaitDecisionProvider` — nothing to do; smoke-test only.
- `FunctionDecisionProvider(fn)` — wraps a pure function or (later) the real
  `SignalAnalyzer.analyze` with the identical signature.
- `ReplayDecisionProvider({ts: decision})` — deterministic replay.

## Known limitations (Phase 12)

- No liquidation/insolvency modeling; the balance can go negative in theory.
- Same-candle TP+SL ambiguity is reported and left unresolved unless a later
  unambiguous candle resolves it (production parity). A finer-resolution
  provider interface exists but is not yet consumed.
- No funding-rate scheduling beyond the explicit `funding_rates` map.

# Local LLM Benchmark (Phase 13)

A side-by-side benchmark of two local Ollama models — **Qwen3 4B** and
**Gemma 4B (gemma3:4b)** — driving the identical backtest data, configuration,
indicator inputs and prompt context. Only the model differs, so the results are
evidence for a human to pick one model for `BTCUSDT_LLM_MODEL`. The benchmark
never changes the production model and never runs live.

> **Disclaimer**: All benchmark numbers are historical research results on a
> single dataset. They do NOT guarantee future or live profitability, and a
> higher/lower score here does not mean the strategy is profitable. No model
> should be traded on this evidence alone.

## Requirements

- A running Ollama server (`ollama serve`) on `http://localhost:11434`.
- One model pulled per run, e.g.:

```bash
ollama pull qwen3:4b
ollama pull gemma3:4b
```

## Running

Run each model separately, to its own output directory, using the **same data
file, same config, same prompt**:

```bash
python -m backtest benchmark \
    --data results/benchmark/data_btcusdt_1h.json \
    --model qwen3:4b --provider ollama \
    --out results/benchmark/qwen3_4b

python -m backtest benchmark \
    --data results/benchmark/data_btcusdt_1h.json \
    --model gemma3:4b --provider ollama \
    --out results/benchmark/gemma_4b
```

Defaults are set for max-control: `--temperature 0.0` (enforced),
`--max-retries 2`, `--timeout 120`, `--max-tokens 4096`. Any other config flag
accepted by the `run` subcommand is available as well. Sequential runs only —
run one model to completion before the other. Do not run both concurrently; the
benchmark is not a race.

### Output

Each `--out` directory receives:

| File                   | Contents                                                      |
| ---------------------- | ------------------------------------------------------------- |
| `decisions.jsonl`      | one record per analyzed candle (status, decision, prices, latency, scope) |
| `summary.json`         | model, config, determinism, engine statistics, signal quality, latency |
| `backtest_result.json` | full `BacktestResult` dump                                    |
| `backtest_trades.csv`  | per-trade rows                                                |
| `equity_curve.csv`     | equity after each executed trade                              |

Re-running the same command replays cached decisions (`cached` in
`summary.json`'s `model_signal_quality`). Use `--force-fresh` to force a real
LLM call for every eligible candle.

## Comparing

```bash
python -m backtest compare \
    results/benchmark/qwen3_4b/summary.json \
    results/benchmark/gemma_4b/summary.json \
    --out results/benchmark
```

Writes `comparison.json` and `comparison.csv` with an 18-metric table (profit
factor, expectancy, net PnL, return, max drawdown, average R, win rates, trades,
signal quality, latency, failures, wait ratio). **No winner is selected** — the
human picks the model.

## Isolation and safety

- The benchmark reuses the Phase 12 backtest path: no SQLite, no Binance API,
  no real orders, no production state.
- The prompt, indicators, candle windows, symbol (`BTCUSDT`), timeframe (`1h`)
  and `max_candles` are identical across models; only the model differs.
- Decisions are scoped by (provider, model, base URL, temperature, retries,
  tokens, symbol, timeframe, `max_candles`, context version, dataset id), so a
  cached decision for Qwen is never reused by Gemma.
- `BTCUSDT_LLM_MODEL` is never read or modified by the benchmark. To use a
  different model in production, change that env var yourself.
- Determinism note: temperature is forced to 0 and `seed_supported` is reported
  in `summary.json`, but Ollama does not guarantee byte-identical tokens; flag
  `--force-fresh` for a fully independent re-run.
- Peak memory is not measured by the benchmark (process-wide RSS is not
  attributable per model) and is reported as `"not available"`.