"""Deterministic analysis context for the BTCUSDT AI signal pipeline.

Phase 5: this module builds a bounded, closed-candle-only context that is the
single input handed to the TradingAgents LLM. It contains no news, no social
sentiment, no fundamentals and no forming candle. The same inputs always
produce the same rendered text.

The pipeline is::

    Market Data  ->  Indicators  ->  AnalysisContext  ->  LLM  ->  Structured Signal

``build_analysis_context`` enforces that every candle is closed, that the
indicator rows are aligned with those candles, and that the indicator snapshot
for the analysis candle is complete (no NaN). It only ever consumes the latest
closed candle as the analysis candle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from binance.indicators import MIN_REQUIRED_CANDLES, IndicatorSnapshot
from binance.market_data import (
    Candle,
    candles_to_dataframe,
    validate_interval,
    validate_symbol,
)

# Columns that must be present and non-NaN on the analysis candle's snapshot.
# ADX direction columns (adx_plus_di / adx_minus_di / adx14 itself) are kept in
# the context but only adx14 is mandatory, matching the optional ADX rule.
SNAPSHOT_COLUMNS = (
    "ema20",
    "ema50",
    "ema200",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr14",
    "adx14",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "volume_sma20",
    "volume_ratio",
)

# Columns shown in the indicator-history table of the rendered context.
# ``macd_signal`` is snapshot-only: the history keeps the MACD line while the
# latest signal value stays in the snapshot, halving per-row token cost.
HISTORY_COLUMNS = (
    "ema20",
    "ema50",
    "ema200",
    "rsi14",
    "macd",
    "atr14",
    "adx14",
    "volume_ratio",
)

#: How many most-recent closed candles the LLM is shown. Phase A trimmed this
#: from 40 to 20 to halve prompt tokens (the full 200+ candle window is still
#: fetched so EMA200 warms up deterministically).
DEFAULT_MAX_CANDLES = 20
_MISSING = "n/a"


class AnalysisContextError(Exception):
    """The market data or indicator input cannot form a valid analysis context."""


def _validate_closed_candles(candles: list[Candle]) -> None:
    if not isinstance(candles, list) or not candles:
        raise AnalysisContextError("no candles provided")
    if not all(isinstance(c, Candle) for c in candles):
        raise AnalysisContextError("candles must be Candle instances")
    for c in candles:
        if not c.is_closed:
            raise AnalysisContextError(
                f"forming (open) candle at timestamp {c.timestamp} must never "
                "be used as an analysis candle"
            )
    timestamps = [c.timestamp for c in candles]
    if timestamps != sorted(timestamps):
        raise AnalysisContextError("candles must be sorted ascending by timestamp")
    if len(set(timestamps)) != len(timestamps):
        raise AnalysisContextError("duplicate candle timestamps are not allowed")


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _fmt(value: float) -> str:
    """Deterministic fixed-width float rendering (trailing zeros trimmed)."""
    if value != value:  # NaN
        return _MISSING
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class AnalysisContext:
    """Closed-candle-only context handed to the LLM.

    ``candles`` and ``indicators`` are the bounded look-back window (most
    recent closed candles first in the rendered text). ``snapshot`` holds the
    complete indicator values for the analysis candle only.
    """

    symbol: str
    timeframe: str
    candles: pd.DataFrame
    indicators: pd.DataFrame
    snapshot: IndicatorSnapshot
    total_candle_count: int
    max_candles: int
    market_timestamp: str
    closed_at: str
    last_close: float
    rendered: str


def build_analysis_context(
    candles: list[Candle],
    indicators: pd.DataFrame,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    max_candles: int = DEFAULT_MAX_CANDLES,
    event_note: str | None = None,
    positioning: str | None = None,
    htf_note: str | None = None,
) -> AnalysisContext:
    """Build the bounded LLM context from closed candles and their indicators.

    Args:
        candles: closed OHLCV candles (e.g. from ``fetch_closed_klines``).
        indicators: output of ``binance.indicators.compute_indicator_matrix``
            computed from exactly the same ``candles`` (one aligned row each).
        symbol: normalized futures symbol, defaults to ``BTCUSDT``.
        timeframe: candle interval, defaults to ``1h``.
        max_candles: how many most-recent closed candles to show the LLM.
        event_note: optional Phase N scheduled-event annotation (event candle
            and/or upcoming release). Appended verbatim as its own section;
            must reference closed candles only (callers guarantee this).
        positioning: optional Phase P1 futures-positioning note (funding
            rate, long/short split, open interest). Appended verbatim.
        htf_note: optional Phase P2 4H-bias note (regime, structure, ADX).
            Appended verbatim.

    Raises:
        AnalysisContextError: for no/forming/out-of-order candles, insufficient
            data, misaligned or missing indicator columns, or NaN in the
            analysis-candle snapshot.
    """
    normalized_symbol = validate_symbol(symbol)
    normalized_interval = validate_interval(timeframe)

    _validate_closed_candles(candles)

    if len(candles) < MIN_REQUIRED_CANDLES:
        raise AnalysisContextError(
            f"at least {MIN_REQUIRED_CANDLES} closed candles are required for "
            f"the full indicator set, got {len(candles)}"
        )

    candle_df = candles_to_dataframe(candles)
    if not isinstance(indicators, pd.DataFrame):
        raise AnalysisContextError("indicators must be a pandas DataFrame")
    if len(indicators) != len(candle_df):
        raise AnalysisContextError(
            "indicators and candles must have the same number of rows "
            f"({len(indicators)} vs {len(candle_df)})"
        )
    if "timestamp" not in indicators.columns:
        raise AnalysisContextError("indicators must include a 'timestamp' column")
    missing = [c for c in SNAPSHOT_COLUMNS if c not in indicators.columns]
    if missing:
        raise AnalysisContextError(
            f"indicators are missing required columns: {', '.join(missing)}"
        )

    candle_ts = candle_df["timestamp"].to_numpy()
    indicator_ts = indicators["timestamp"].to_numpy()
    if not (indicator_ts == candle_ts).all():
        raise AnalysisContextError(
            "indicators are not aligned with candles: every indicator row must "
            "match the candle with the same timestamp"
        )

    total = len(candle_df)
    lookback = max(1, min(int(max_candles), total))
    candle_window = candle_df.iloc[-lookback:].reset_index(drop=True)
    indicator_window = indicators.iloc[-lookback:].reset_index(drop=True)

    snapshot_row = indicators.iloc[-1]
    for column in SNAPSHOT_COLUMNS:
        value = snapshot_row[column]
        if pd.isna(value):
            raise AnalysisContextError(
                f"indicator column '{column}' has NaN on the analysis candle; "
                "cannot analyze with incomplete indicators"
            )

    analysis_candle = candles[-1]
    snapshot = IndicatorSnapshot(
        timestamp=int(snapshot_row["timestamp"]),
        last_close=float(snapshot_row["close"]),
        ema20=float(snapshot_row["ema20"]),
        ema50=float(snapshot_row["ema50"]),
        ema200=float(snapshot_row["ema200"]),
        rsi14=float(snapshot_row["rsi14"]),
        macd=float(snapshot_row["macd"]),
        macd_signal=float(snapshot_row["macd_signal"]),
        macd_histogram=float(snapshot_row["macd_histogram"]),
        atr14=float(snapshot_row["atr14"]),
        adx14=float(snapshot_row["adx14"]),
        adx_plus_di=float(snapshot_row["adx_plus_di"]),
        adx_minus_di=float(snapshot_row["adx_minus_di"]),
        bb_mid=float(snapshot_row["bb_mid"]),
        bb_upper=float(snapshot_row["bb_upper"]),
        bb_lower=float(snapshot_row["bb_lower"]),
        volume_sma20=float(snapshot_row["volume_sma20"]),
        volume_ratio=float(snapshot_row["volume_ratio"]),
    )

    market_timestamp = _ms_to_iso(int(analysis_candle.timestamp))
    closed_at = _ms_to_iso(int(analysis_candle.close_time))
    last_close = float(analysis_candle.close)
    rendered = _render_context(
        symbol=normalized_symbol,
        timeframe=normalized_interval,
        window=candle_window,
        indicator_window=indicator_window,
        snapshot=snapshot,
        total=total,
        market_timestamp=market_timestamp,
        closed_at=closed_at,
        last_close=last_close,
        event_note=event_note,
        positioning=positioning,
        htf_note=htf_note,
    )

    return AnalysisContext(
        symbol=normalized_symbol,
        timeframe=normalized_interval,
        candles=candle_window,
        indicators=indicator_window,
        snapshot=snapshot,
        total_candle_count=total,
        max_candles=lookback,
        market_timestamp=market_timestamp,
        closed_at=closed_at,
        last_close=last_close,
        rendered=rendered,
    )


def _render_candle_rows(window: pd.DataFrame) -> str:
    lines = []
    for index, row in enumerate(window.iloc[::-1].itertuples(index=False), start=1):
        lines.append(
            f"| {index} | {_ms_to_iso(int(row.timestamp))} "
            f"| {_fmt(float(row.open))} | {_fmt(float(row.high))} "
            f"| {_fmt(float(row.low))} | {_fmt(float(row.close))} "
            f"| {_fmt(float(row.volume))} |"
        )
    return "\n".join(lines)


def _render_indicator_rows(indicator_window: pd.DataFrame) -> str:
    lines = []
    for index, row in enumerate(
        indicator_window.iloc[::-1].itertuples(index=False), start=1
    ):
        cells = " ".join(
            f"{column}:{_fmt(float(getattr(row, column)))}"
            for column in HISTORY_COLUMNS
        )
        lines.append(f"| {index} | {_ms_to_iso(int(row.timestamp))} | {cells} |")
    return "\n".join(lines)


def _render_context(
    *,
    symbol: str,
    timeframe: str,
    window: pd.DataFrame,
    indicator_window: pd.DataFrame,
    snapshot: IndicatorSnapshot,
    total: int,
    market_timestamp: str,
    closed_at: str,
    last_close: float,
    event_note: str | None = None,
    positioning: str | None = None,
    htf_note: str | None = None,
) -> str:
    lines = [
        f"# {symbol} — Binance USDT-M Futures",
        f"Timeframe: {timeframe}",
        "Data source: CLOSED Binance candles + deterministic Python indicators only.",
        "",
        "## Analysis candle (latest CLOSED candle)",
        f"- Candle open time (market timestamp): {market_timestamp}",
        f"- Candle close time (closed_at): {closed_at}",
        f"- Last close price: {_fmt(last_close)}",
        f"- Closed candles supplied: {total}",
        "",
        "## Recent closed candles (most recent first)",
        "| # | open time (UTC) | open | high | low | close | volume |",
        "|---|-----------------|------|------|-----|-------|--------|",
    ]
    lines.append(_render_candle_rows(window))
    lines += [
        "",
        "## Indicator snapshot (analysis candle)",
        (
            f"EMA20={_fmt(snapshot.ema20)} EMA50={_fmt(snapshot.ema50)} "
            f"EMA200={_fmt(snapshot.ema200)}"
        ),
        f"RSI14={_fmt(snapshot.rsi14)}",
        (
            f"MACD={_fmt(snapshot.macd)} signal={_fmt(snapshot.macd_signal)} "
            f"histogram={_fmt(snapshot.macd_histogram)}"
        ),
        f"ATR14={_fmt(snapshot.atr14)}",
        (
            f"ADX14={_fmt(snapshot.adx14)} "
            f"+DI={_fmt(snapshot.adx_plus_di)} -DI={_fmt(snapshot.adx_minus_di)}"
        ),
        (
            f"Bollinger mid={_fmt(snapshot.bb_mid)} upper={_fmt(snapshot.bb_upper)} "
            f"lower={_fmt(snapshot.bb_lower)}"
        ),
        (
            f"Volume SMA20={_fmt(snapshot.volume_sma20)} "
            f"volume ratio={_fmt(snapshot.volume_ratio)}"
        ),
        "",
        "## Indicator history (most recent first)",
        "| # | open time (UTC) | " + ", ".join(HISTORY_COLUMNS) + " |",
        "|---|-----------------|-----------------------------------------------------------|",
    ]
    lines.append(_render_indicator_rows(indicator_window))
    note = (event_note or "").strip()
    if note:
        lines += ["", "## Scheduled event note", note]
    pos = (positioning or "").strip()
    if pos:
        lines += ["", "## Futures positioning", pos]
    htf = (htf_note or "").strip()
    if htf:
        lines += ["", "## 4H bias", htf]
    return "\n".join(lines)
