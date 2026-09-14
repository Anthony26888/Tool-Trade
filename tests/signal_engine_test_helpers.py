"""Shared builders for Phase 5/6 signal-engine tests (not collected by pytest)."""

from __future__ import annotations

import os
import tempfile

from binance.indicators import compute_indicator_matrix
from binance.market_data import Candle
from database.database import Database, SignalRepository
from signal_engine import SignalAnalysis, SignalEngine, SignalState

INTERVAL_MS = 3_600_000
DEFAULT_START_MS = 1_720_000_000_000


def make_candles(
    count: int,
    *,
    start_ms: int = DEFAULT_START_MS,
    base: float = 60000.0,
    trend: float = 0.5,
    volume_base: float = 100.0,
    forming: bool = False,
) -> list[Candle]:
    """Deterministic closed 1H candles with a gentle uptrend."""
    candles = []
    for i in range(count):
        ts = start_ms + i * INTERVAL_MS
        open_price = base + i * trend
        close = open_price + trend
        high = max(open_price, close) + (i % 5)
        low = min(open_price, close) - ((i % 3) + 1)
        volume = volume_base + (i % 7) * 10.0
        candles.append(
            Candle(
                timestamp=ts,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                close_time=ts + INTERVAL_MS - 1,
                is_closed=True,
            )
        )
    if forming:
        ts = start_ms + count * INTERVAL_MS
        candles.append(
            Candle(
                timestamp=ts,
                open=close,
                high=close + 5.0,
                low=close - 5.0,
                close=close + 2.0,
                volume=volume_base,
                close_time=ts + INTERVAL_MS - 1,
                is_closed=False,
            )
        )
    return candles


def indicators_for(candles: list[Candle]):
    """Compute the Phase 2 indicator matrix for a candle list."""
    return compute_indicator_matrix(candles)


def make_analysis(
    decision: str = "LONG",
    *,
    entry_price=61000.0,
    stop_loss=60000.0,
    take_profit=64000.0,
    confidence: float = 80.0,
    reasoning: str = "uptrend continuation",
    provider: str = "ollama",
    model: str = "qwen3:8b",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    analysis_timestamp: str = "2026-09-10T01:00:00.000Z",
    market_timestamp: str = "2026-09-10T00:00:00.000Z",
    closed_at: str = "2026-09-10T01:00:00.000Z",
    candle_close_price: float = 61050.5,
) -> SignalAnalysis:
    """A populated Phase 5 ``SignalAnalysis`` for engine/validator tests."""
    return SignalAnalysis(
        decision=decision,
        confidence=confidence,
        reasoning=reasoning,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        provider=provider,
        model=model,
        symbol=symbol,
        timeframe=timeframe,
        analysis_timestamp=analysis_timestamp,
        market_timestamp=market_timestamp,
        closed_at=closed_at,
        candle_close_price=candle_close_price,
    )


class TempSignalDb:
    """Fresh temporary SQLite database with a ready engine + repository + state."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "signals.db")
        self.db = Database(self.path)
        self.db.initialize()
        self.repository = SignalRepository(self.db)
        self.state = SignalState(self.repository)
        self.engine = SignalEngine(self.repository, self.state)

    def close(self) -> None:
        self._tmp.cleanup()
