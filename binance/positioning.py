"""Futures positioning snapshot (Phase P1): funding rate, open interest,
long/short account ratio.

These are Binance USDT-M Futures **public** endpoints (no API key): they tell
how crowded each side is — the one futures-native dataset pure OHLCV cannot
see. Everything here is best-effort: any fetch/parse failure degrades to
``None`` fields, and no function ever raises, so a dead positioning feed can
never block analysis (AGENTS.md 27). Tests inject a fake client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Hard sanity caps; values beyond these are treated as corrupt feed data.
_MAX_ABS_FUNDING = 0.05  # 5% per 8h (real extremes stay well below 1%)


@dataclass(frozen=True)
class PositioningSnapshot:
    """One live positioning read. ``None`` fields mean unavailable."""

    funding_rate: float | None = None
    funding_time_ms: int | None = None
    open_interest: float | None = None
    long_pct: float | None = None
    short_pct: float | None = None


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def _int_ms(value: Any) -> int | None:
    number = _num(value)
    if number is None or number < 0:
        return None
    return int(number)


def _fetch_funding(client: Any, symbol: str) -> tuple[float | None, int | None]:
    try:
        rows = client.get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
    except Exception as exc:
        logger.warning("[Positioning] fundingRate failed for %s: %s", symbol, exc)
        return (None, None)
    row = rows[-1] if isinstance(rows, list) and rows else None
    if not isinstance(row, dict):
        return (None, None)
    rate = _num(row.get("fundingRate"))
    if rate is not None and abs(rate) > _MAX_ABS_FUNDING:
        logger.warning("[Positioning] absurd funding rate %r ignored", rate)
        return (None, None)
    return (rate, _int_ms(row.get("fundingTime")))


def _fetch_open_interest(client: Any, symbol: str) -> float | None:
    try:
        payload = client.get("/fapi/v1/openInterest", {"symbol": symbol})
    except Exception as exc:
        logger.warning("[Positioning] openInterest failed for %s: %s", symbol, exc)
        return None
    if not isinstance(payload, dict):
        return None
    value = _num(payload.get("openInterest"))
    return value if value is not None and value >= 0 else None


def _fetch_long_short(client: Any, symbol: str) -> tuple[float | None, float | None]:
    try:
        rows = client.get(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": "1h", "limit": 1},
        )
    except Exception as exc:
        logger.warning(
            "[Positioning] longShortAccountRatio failed for %s: %s", symbol, exc
        )
        return (None, None)
    row = rows[-1] if isinstance(rows, list) and rows else None
    if not isinstance(row, dict):
        return (None, None)
    ratio = _num(row.get("longShortRatio"))
    if ratio is None or ratio <= 0:
        return (None, None)
    long_pct = ratio / (1.0 + ratio) * 100.0
    return (long_pct, 100.0 - long_pct)


def fetch_positioning(client: Any, symbol: str) -> PositioningSnapshot:
    """Read the live positioning snapshot (never raises)."""
    try:
        funding_rate, funding_time_ms = _fetch_funding(client, symbol)
        open_interest = _fetch_open_interest(client, symbol)
        long_pct, short_pct = _fetch_long_short(client, symbol)
        return PositioningSnapshot(
            funding_rate=funding_rate,
            funding_time_ms=funding_time_ms,
            open_interest=open_interest,
            long_pct=long_pct,
            short_pct=short_pct,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.warning("[Positioning] snapshot failed for %s: %s", symbol, exc)
        return PositioningSnapshot()


def render_note(
    funding_rate: float | None,
    long_pct: float | None,
    short_pct: float | None,
    open_interest: float | None = None,
) -> str | None:
    """Render the ``## Futures positioning`` context lines (None when empty)."""
    parts: list[str] = []
    if funding_rate is not None:
        parts.append(f"funding={funding_rate:+.4%}/8h")
    if long_pct is not None and short_pct is not None:
        parts.append(f"longs {long_pct:.1f}% / shorts {short_pct:.1f}%")
    if open_interest is not None:
        parts.append(f"open interest {open_interest:,.1f} BTC")
    if not parts:
        return None
    note = "Futures positioning: " + " · ".join(parts) + "."
    if funding_rate is not None and abs(funding_rate) >= 0.0005:
        side = "long" if funding_rate > 0 else "short"
        note += (
            f" Funding is extremely crowded {side}; treat {side} entries "
            "with extra skepticism."
        )
    return note


# ── history (backtest) ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _Point:
    ms: int
    value: float


class PositioningHistory:
    """Frozen funding + long/short history for deterministic replay.

    Built either from a frozen artifact (``from_frozen``) or by a one-off
    live fetch over the dataset span (``fetch`` — best-effort, empty on any
    failure). ``value_at`` returns the latest point at or before ``ms``.
    """

    def __init__(
        self,
        funding: list[tuple[int, float]] | None = None,
        long_short: list[tuple[int, float]] | None = None,
    ) -> None:
        self._funding = sorted(funding or [])
        self._long_short = sorted(long_short or [])

    @classmethod
    def from_frozen(cls, items: list[dict[str, Any]]) -> PositioningHistory:
        funding: list[tuple[int, float]] = []
        long_short: list[tuple[int, float]] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            ms = _int_ms(item.get("ms"))
            if ms is None:
                continue
            rate = _num(item.get("funding_rate"))
            if rate is not None and abs(rate) <= _MAX_ABS_FUNDING:
                funding.append((ms, rate))
            ratio = _num(item.get("long_pct"))
            if ratio is not None and 0.0 < ratio < 100.0:
                long_short.append((ms, ratio))
        return cls(funding, long_short)

    def frozen(self) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for ms, rate in self._funding:
            merged.setdefault(ms, {"ms": ms})["funding_rate"] = rate
        for ms, ratio in self._long_short:
            merged.setdefault(ms, {"ms": ms})["long_pct"] = ratio
        return [merged[ms] for ms in sorted(merged)]

    @classmethod
    def fetch(
        cls, client: Any, symbol: str, start_ms: int, end_ms: int
    ) -> PositioningHistory:
        """One-off history fetch over ``[start_ms, end_ms]`` (never raises)."""
        try:
            funding = cls._funding_history(client, symbol, int(start_ms), int(end_ms))
            long_short = cls._ls_history(client, symbol, int(start_ms), int(end_ms))
            logger.info(
                "[Positioning] history: %d funding + %d long/short points",
                len(funding),
                len(long_short),
            )
            return cls(funding, long_short)
        except Exception as exc:
            logger.warning("[Positioning] history fetch failed: %s", exc)
            return cls()

    @staticmethod
    def _funding_history(
        client: Any, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        cursor = start_ms
        while cursor < end_ms:
            rows = client.get(
                "/fapi/v1/fundingRate",
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            )
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ms = _int_ms(row.get("fundingTime"))
                rate = _num(row.get("fundingRate"))
                if ms is not None and rate is not None and abs(rate) <= _MAX_ABS_FUNDING:
                    points.append((ms, rate))
            newest = _int_ms(rows[-1].get("fundingTime")) if isinstance(rows[-1], dict) else None
            if newest is None or newest <= cursor:
                break
            cursor = newest + 1
            if len(points) > 100_000:
                break
        return points

    @staticmethod
    def _ls_history(
        client: Any, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        cursor = start_ms
        while cursor < end_ms:
            rows = client.get(
                "/futures/data/globalLongShortAccountRatio",
                {
                    "symbol": symbol,
                    "period": "1h",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 500,
                },
            )
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ms = _int_ms(row.get("timestamp"))
                ratio = _num(row.get("longShortRatio"))
                if ms is None or ratio is None or ratio <= 0:
                    continue
                points.append((ms, ratio / (1.0 + ratio) * 100.0))
            newest = _int_ms(rows[-1].get("timestamp")) if isinstance(rows[-1], dict) else None
            if newest is None or newest <= cursor:
                break
            cursor = newest + 1
            if len(points) > 100_000:
                break
        return points

    def _latest(self, points: list[tuple[int, float]], ms: int) -> float | None:
        value: float | None = None
        for point_ms, point_value in points:
            if point_ms <= ms:
                value = point_value
            else:
                break
        return value

    def funding_at(self, ms: int) -> float | None:
        return self._latest(self._funding, int(ms))

    def long_pct_at(self, ms: int) -> float | None:
        return self._latest(self._long_short, int(ms))

    def note_at(self, ms: int) -> str | None:
        """Prompt note from history as of ``ms`` (short % unavailable)."""
        funding = self.funding_at(ms)
        long_pct = self.long_pct_at(ms)
        if funding is None and long_pct is None:
            return None
        return render_note(funding, long_pct, None if long_pct is None else 100.0 - long_pct)


__all__ = [
    "PositioningHistory",
    "PositioningSnapshot",
    "fetch_positioning",
    "render_note",
]
