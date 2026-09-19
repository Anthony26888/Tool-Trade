"""Higher-timeframe (4H) trend bias (Phase P2, soft version).

The 1H AI only sees ~one day of price action, so a 3-4 candle bounce inside a
weekly downtrend looks like a fresh uptrend. This module adds the missing
context: the 4H regime (EMA stack), its confirmation (swing structure), and
its strength (ADX) — all deterministic Python over closed 4H candles.

Soft-version contract: the hard rule only ever vetoes entries *against* a
clear regime (UP blocks SHORT, DOWN blocks LONG); an unclear regime never
blocks, and structure/ADX are context for the AI, never a veto.

Everything here is fail-soft: bad/short input yields a ``NONE`` bias with no
note (never raises), so a dead 4H feed can never block analysis. Backtests
reuse this exact code over 4H candles rolled up from the 1H dataset.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from binance.indicators import compute_indicator_matrix
from binance.market_data import Candle

logger = logging.getLogger(__name__)

#: 4H candles per fetch (EMA50 + ADX warm-up need ~50; +10 headroom).
HTF_CANDLE_LIMIT = 60
#: Binance 4H candles align to UTC midnight; grouping 1H candles by this
#: modulus reproduces them exactly.
H4_MS = 4 * 3_600_000
#: ADX(14) at/above this reads as a tradable trend; below reads as chop.
ADX_TREND_MIN = 20.0
#: Swing confirmation radius: a high is a swing when strictly hottest in
#: ±radius. Radius 1 keeps two confirmed swings inside a short lookback
#: (radius 2 would admit at most one swing per 7-candle window, so the
#: structure could never confirm).
_SWING_RADIUS = 1
#: Recent 4H candles inspected for swing structure.
_STRUCTURE_LOOKBACK = 9

REGIME_UP = "UP"
REGIME_DOWN = "DOWN"
REGIME_NONE = "NONE"


@dataclass(frozen=True)
class HtfBias:
    """4H bias: regime for the veto, structure/ADX as AI context."""

    regime: str = REGIME_NONE
    structure: str = "NONE"  # HH_HL | LH_LL | NONE
    adx: float | None = None
    rsi: float | None = None
    close: float | None = None
    note: str | None = None


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _swing_highs(highs: list[float], radius: int = _SWING_RADIUS) -> list[int]:
    """Indices of confirmed swing highs (strictly hottest in ±radius)."""
    out: list[int] = []
    for i in range(radius, len(highs) - radius):
        window = highs[i - radius : i + radius + 1]
        if highs[i] == max(window) and window.count(highs[i]) == 1:
            out.append(i)
    return out


def _swing_lows(lows: list[float], radius: int = _SWING_RADIUS) -> list[int]:
    out: list[int] = []
    for i in range(radius, len(lows) - radius):
        window = lows[i - radius : i + radius + 1]
        if lows[i] == min(window) and window.count(lows[i]) == 1:
            out.append(i)
    return out


def detect_structure(candles: list[Candle]) -> str:
    """Swing structure over recent 4H candles (never raises).

    Compares the last two *confirmed* swing highs/lows: HH+HL reads
    ``HH_HL``, LH+LL reads ``LH_LL``; anything else (incl. too few swings)
    reads ``NONE``. The two newest candles are unconfirmable and excluded.
    """
    try:
        window = list(candles)[-_STRUCTURE_LOOKBACK:]
        if len(window) < _STRUCTURE_LOOKBACK:
            return "NONE"
        highs = [float(c.high) for c in window]
        lows = [float(c.low) for c in window]
        swing_h = [i for i in _swing_highs(highs) if i < len(window) - _SWING_RADIUS]
        swing_l = [i for i in _swing_lows(lows) if i < len(window) - _SWING_RADIUS]
        if len(swing_h) < 2 or len(swing_l) < 2:
            return "NONE"
        h1, h2 = highs[swing_h[-2]], highs[swing_h[-1]]
        l1, l2 = lows[swing_l[-2]], lows[swing_l[-1]]
        if h2 > h1 and l2 > l1:
            return "HH_HL"
        if h2 < h1 and l2 < l1:
            return "LH_LL"
        return "NONE"
    except Exception as exc:
        logger.warning("[HTF] structure detection failed: %s", exc)
        return "NONE"


def render_note(
    regime: str, structure: str, adx: float | None, rsi: float | None = None
) -> str:
    """Render the ``## 4H bias`` context line for a classified bias."""
    adx_text = f"ADX {adx:.1f}" if adx is not None else "ADX n/a"
    struct_text = {"HH_HL": "HH/HL", "LH_LL": "LH/LL"}.get(structure, "no clear structure")
    rsi_text = f", RSI {rsi:.1f}" if rsi is not None else ""
    if regime == REGIME_UP:
        return f"4H bias: UPTREND (EMA stack up, {struct_text}, {adx_text}{rsi_text}). Prefer LONG; SHORT needs exceptional 1H evidence."
    if regime == REGIME_DOWN:
        return f"4H bias: DOWNTREND (EMA stack down, {struct_text}, {adx_text}{rsi_text}). Prefer SHORT; LONG needs exceptional 1H evidence."
    if adx is not None and adx < ADX_TREND_MIN:
        return (
            f"4H bias: RANGE ({struct_text}, {adx_text}{rsi_text} — chop, "
            "prefer WAIT unless the 1H setup is decisive)."
        )
    return f"4H bias: UNCLEAR ({struct_text}, {adx_text}{rsi_text})."


def classify_htf(candles_4h: list[Candle] | None) -> HtfBias:
    """Classify the 4H bias from closed 4H candles (never raises).

    Regime comes from the EMA stack on the latest candle
    (close > EMA20 > EMA50 = UP, mirrored = DOWN); anything missing or
    unaligned reads NONE. Structure and ADX are attached as context.
    """
    try:
        if not candles_4h:
            return HtfBias()
        matrix = compute_indicator_matrix(list(candles_4h))
        row = matrix.iloc[-1]
        close = _finite(row["close"])
        ema20 = _finite(row["ema20"])
        ema50 = _finite(row["ema50"])
        adx = _finite(row["adx14"])
        rsi = _finite(row["rsi14"])
        if close is None or ema20 is None or ema50 is None:
            return HtfBias(adx=adx, rsi=rsi, close=close)
        if close > ema20 > ema50:
            regime = REGIME_UP
        elif close < ema20 < ema50:
            regime = REGIME_DOWN
        else:
            regime = REGIME_NONE
        structure = detect_structure(list(candles_4h))
        return HtfBias(
            regime=regime,
            structure=structure,
            adx=adx,
            rsi=rsi,
            close=close,
            note=render_note(regime, structure, adx, rsi),
        )
    except Exception as exc:
        logger.warning("[HTF] classification failed: %s", exc)
        return HtfBias()


def rollup_1h_to_4h(candles_1h: list[Candle]) -> list[Candle]:
    """Roll closed 1H candles into closed 4H candles (backtest parity).

    Groups by UTC-midnight-aligned 4H buckets (order-insensitive); groups
    that are not exactly 4 closed 1H candles (e.g. an incomplete trailing
    group) are dropped.
    """
    buckets: dict[int, list[Candle]] = {}
    for candle in candles_1h:
        buckets.setdefault(int(candle.timestamp) // H4_MS, []).append(candle)
    out: list[Candle] = []
    for key in sorted(buckets):
        group = buckets[key]
        if len(group) != 4 or not all(c.is_closed for c in group):
            continue
        ordered = sorted(group, key=lambda c: int(c.timestamp))
        out.append(
            Candle(
                timestamp=ordered[0].timestamp,
                open=ordered[0].open,
                high=max(float(c.high) for c in ordered),
                low=min(float(c.low) for c in ordered),
                close=ordered[-1].close,
                volume=sum(float(c.volume) for c in ordered),
                close_time=ordered[-1].close_time,
                is_closed=True,
            )
        )
    return out


__all__ = [
    "ADX_TREND_MIN",
    "H4_MS",
    "HTF_CANDLE_LIMIT",
    "REGIME_DOWN",
    "REGIME_NONE",
    "REGIME_UP",
    "HtfBias",
    "classify_htf",
    "detect_structure",
    "render_note",
    "rollup_1h_to_4h",
]
