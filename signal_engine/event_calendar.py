"""Scheduled macro-event calendar (Phase N).

BTCUSDT reacts violently to scheduled US macro releases (FOMC decisions, CPI,
NFP). Those releases are known months in advance, so the cheapest correct
handling is a hard rule — never open a new position inside a blackout window
around the release — instead of asking the LLM to guess the news.

Primary source: the free financecalendar.com JSON API (no key, no rate
limits; times verified against official sources). The module fetches
``/next?series=...`` for the tracked series at most once per
``refresh_interval_s`` and keeps everything else in RAM. Any fetch/parse
failure degrades to the curated fallback file
(``signal_engine/event_calendar.json``), and if that is also unavailable the
calendar reports no events (fail-open: analysis continues normally).

Nothing here ever raises to callers: ``refresh()``/``check()`` swallow all
errors into a warning log so a dead calendar feed can never stall the
scheduler (AGENTS.md 27). Tests inject ``fetcher`` so no test touches the
network.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

#: Free economic-calendar JSON API (no key, edge-cached 5 min, CORS open).
#: Terms: attribution link to https://www.financecalendar.com where shown.
API_BASE = "https://www.financecalendar.com/wp-json/fc/v1"

#: Network budget per calendar fetch; the scheduler must never block on this.
API_TIMEOUT_S = 10.0

#: Tracked release series -> (blackout hours before, blackout hours after).
#: FOMC decisions spike both directions for hours; CPI/NFP settle faster.
SERIES_WINDOWS_H: dict[str, tuple[float, float]] = {
    "fomc": (3.0, 3.0),
    "cpi": (2.0, 2.0),
    "nfp": (2.0, 2.0),
}

#: Env knob for the pre-event Telegram heads-up (hours). 0 disables WARN.
ENV_EVENT_WARN_HOURS = "BTCUSDT_EVENT_WARN_HOURS"
DEFAULT_WARN_HOURS = 12.0

#: The AI is told about an upcoming release only when it is this close, so
#: the prompt stays clean most of the time (~10 tokens, only when relevant).
UPCOMING_NOTE_HOURS = 24.0

_VN_OFFSET = timezone(timedelta(hours=7))


def warn_hours_from_env() -> float:
    """Resolve ``BTCUSDT_EVENT_WARN_HOURS`` (per call, no caching)."""
    raw = (os.environ.get(ENV_EVENT_WARN_HOURS) or "").strip()
    if not raw:
        return DEFAULT_WARN_HOURS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_WARN_HOURS
    return max(0.0, value)


def format_vn(ms: int) -> str:
    """Render epoch ms as ``HH:MM DD/MM (+07)`` (dashboard/Telegram display)."""
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(_VN_OFFSET)
    return dt.strftime("%H:%M %d/%m (+07)")


def _parse_ms(value: Any) -> int | None:
    """Parse an ISO-8601 timestamp (``time_utc``) to epoch ms, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return int(
            datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            * 1000
        )
    except ValueError:
        return None


def _default_fetcher(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "tradingagents/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _clip300(value: Any) -> str | None:
    """Clip an optional API text field for prompts/dialogs (None stays None)."""
    return str(value)[:300] if isinstance(value, str) and value else None


@dataclass(frozen=True)
class ScheduledEvent:
    """One timed macro release with its trading blackout window."""

    series: str
    name: str
    title: str
    event_ms: int
    impact: str = ""
    url: str = ""
    consensus: str | None = None
    prior: str | None = None
    actual: str | None = None
    source: str = "api"
    window_before_ms: int = 0
    window_after_ms: int = 0

    @property
    def key(self) -> str:
        """Stable per-release identity for once-per-event notification flags."""
        day = datetime.fromtimestamp(
            self.event_ms / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%d")
        return f"{self.series}:{day}"

    @property
    def blackout_start_ms(self) -> int:
        return self.event_ms - self.window_before_ms

    @property
    def blackout_end_ms(self) -> int:
        return self.event_ms + self.window_after_ms

    def contains(self, ms: int) -> bool:
        return self.blackout_start_ms <= int(ms) <= self.blackout_end_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": self.series,
            "name": self.name,
            "title": self.title,
            "event_ms": self.event_ms,
            "event_utc": datetime.fromtimestamp(
                self.event_ms / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_vn": format_vn(self.event_ms),
            "impact": self.impact,
            "url": self.url,
            "consensus": self.consensus,
            "prior": self.prior,
            "actual": self.actual,
            "source": self.source,
            "trading_paused": True,
            "blackout_start_ms": self.blackout_start_ms,
            "blackout_start_vn": format_vn(self.blackout_start_ms),
            "blackout_end_ms": self.blackout_end_ms,
            "blackout_end_vn": format_vn(self.blackout_end_ms),
        }


@dataclass
class EventCheck:
    """Result of :meth:`EventCalendar.check` for one instant."""

    active: ScheduledEvent | None = None
    upcoming: list[ScheduledEvent] = field(default_factory=list)


class EventCalendar:
    """Cached macro-event source with a static fallback (never raises)."""

    def __init__(
        self,
        *,
        fetcher: Callable[[str, float], dict[str, Any]] | None = None,
        fallback_path: str | None = None,
        refresh_interval_s: float = 3600.0,
        warn_hours: float | None = None,
    ) -> None:
        self._fetcher = fetcher if fetcher is not None else _default_fetcher
        self._fallback_path = (
            fallback_path
            if fallback_path is not None
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_calendar.json")
        )
        self._refresh_interval_ms = int(max(60.0, float(refresh_interval_s)) * 1000)
        self._warn_hours = (
            float(warn_hours) if warn_hours is not None else warn_hours_from_env()
        )
        self._events: list[ScheduledEvent] = []
        self._last_refresh_ms: int | None = None
        self._month_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        #: Where the current cache came from: "live" (API), "fallback"
        #: (static file), or "none" (nothing loaded yet). Surfaced in the UI
        #: so a fallback-only view is never mistaken for full coverage.
        self._source: str = "none"

    @property
    def warn_hours(self) -> float:
        return max(0.0, self._warn_hours)

    @property
    def source(self) -> str:
        return self._source

    def override_warn_hours(self, value: float | None) -> None:
        """Override the WARN horizon (Settings overlay; None = keep current).

        Called per scheduler tick so a Settings save applies with no restart.
        Out-of-range values are ignored (fail safe towards the default).
        """
        try:
            if value is None:
                return
            number = float(value)
            if number != number or number < 0:
                return
            self._warn_hours = number
        except (TypeError, ValueError):
            return

    # -- refresh ----------------------------------------------------------

    def refresh(self, now_ms: int, *, force: bool = False) -> None:
        """Refresh the cached releases (throttled; never raises)."""
        now_ms = int(now_ms)
        if (
            not force
            and self._last_refresh_ms is not None
            and now_ms - self._last_refresh_ms < self._refresh_interval_ms
        ):
            return
        try:
            events = self._fetch_all()
        except Exception as exc:  # defensive: the feed must never stall us
            logger.warning("[Events] calendar refresh failed, keeping cache: %s", exc)
            return
        if events:
            self._events = sorted(events, key=lambda e: e.event_ms)
            self._last_refresh_ms = now_ms
            self._month_cache.clear()
            self._source = "live"
            logger.info("[Events] calendar refreshed: %d upcoming releases", len(events))
        elif not self._events:
            self._load_fallback()

    def _fetch_all(self) -> list[ScheduledEvent]:
        events: list[ScheduledEvent] = []
        for series, (before_h, after_h) in SERIES_WINDOWS_H.items():
            try:
                payload = self._fetcher(
                    f"{API_BASE}/next?series={series}", API_TIMEOUT_S
                )
            except Exception as exc:
                logger.warning("[Events] /next?series=%s failed: %s", series, exc)
                continue
            event = self._parse_next(series, payload, before_h, after_h)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _parse_next(
        series: str, payload: Any, before_h: float, after_h: float
    ) -> ScheduledEvent | None:
        if not isinstance(payload, dict):
            return None
        item = payload.get("next")
        if not isinstance(item, dict):
            return None
        if item.get("all_day"):
            return None
        event_ms = _parse_ms(item.get("time_utc"))
        if event_ms is None:
            return None
        return ScheduledEvent(
            series=series,
            name=str(item.get("name") or series).strip() or series,
            title=str(item.get("title") or item.get("name") or series).strip(),
            event_ms=event_ms,
            impact=str(item.get("impact") or ""),
            url=str(item.get("url") or ""),
            consensus=_clip300(item.get("consensus")),
            prior=_clip300(item.get("prior")),
            actual=_clip300(item.get("actual")),
            source="api",
            window_before_ms=int(before_h * 3_600_000),
            window_after_ms=int(after_h * 3_600_000),
        )

    def _load_fallback(self) -> None:
        try:
            with open(self._fallback_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning(
                "[Events] fallback file unavailable (%s); no scheduled events",
                exc,
            )
            return
        items = payload if isinstance(payload, list) else payload.get("events", [])
        events: list[ScheduledEvent] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            series = str(item.get("series") or "").strip().lower()
            if series not in SERIES_WINDOWS_H:
                continue
            event_ms = _parse_ms(item.get("time_utc"))
            if event_ms is None:
                continue
            before_h, after_h = SERIES_WINDOWS_H[series]
            events.append(
                ScheduledEvent(
                    series=series,
                    name=str(item.get("name") or series),
                    title=str(item.get("title") or item.get("name") or series),
                    event_ms=event_ms,
                    impact=str(item.get("impact") or "high"),
                    url=str(item.get("url") or ""),
                    source="fallback",
                    window_before_ms=int(before_h * 3_600_000),
                    window_after_ms=int(after_h * 3_600_000),
                )
            )
        if events:
            self._events = sorted(events, key=lambda e: e.event_ms)
            self._source = "fallback"
            logger.info(
                "[Events] using fallback file: %d releases", len(self._events)
            )

    # -- queries ----------------------------------------------------------

    def check(self, now_ms: int) -> EventCheck:
        """Return the active blackout event (if any) plus future releases."""
        try:
            self.refresh(now_ms)
            now_ms = int(now_ms)
            active = next((e for e in self._events if e.contains(now_ms)), None)
            upcoming = [e for e in self._events if e.event_ms > now_ms]
            return EventCheck(active=active, upcoming=upcoming)
        except Exception as exc:  # never break the caller
            logger.warning("[Events] check failed, treating as no event: %s", exc)
            return EventCheck()

    def blackout_at(self, ms: int) -> ScheduledEvent | None:
        """Event whose blackout contains ``ms`` (backtest gate; never raises)."""
        try:
            return next((e for e in self._events if e.contains(int(ms))), None)
        except Exception as exc:
            logger.warning("[Events] blackout_at failed: %s", exc)
            return None

    def load_range(self, from_ms: int, to_ms: int) -> list[ScheduledEvent]:
        """Load historical releases for ``[from_ms, to_ms]`` (backtest).

        Fetches ``/calendar`` in <=92-day chunks and keeps only entries that
        map to a trading blackout (FOMC/US CPI/NFP). Replaces the cache; call
        on a dedicated instance, never on the live scheduler's calendar.
        Returns the loaded events. Never raises (empty list on failure).
        """
        try:
            return self._load_range(int(from_ms), int(to_ms))
        except Exception as exc:
            logger.warning("[Events] load_range failed: %s", exc)
            return []

    def _load_range(self, from_ms: int, to_ms: int) -> list[ScheduledEvent]:
        if to_ms < from_ms:
            return []
        events: list[ScheduledEvent] = []
        day_ms = 86_400_000
        chunk_ms = 92 * day_ms
        start = from_ms - (from_ms % day_ms)
        while start <= to_ms:
            end = min(start + chunk_ms - 1, to_ms)
            first = datetime.fromtimestamp(start / 1000.0, tz=timezone.utc)
            last = datetime.fromtimestamp(end / 1000.0, tz=timezone.utc)
            payload = self._fetcher(
                f"{API_BASE}/calendar?from={first.strftime('%Y-%m-%d')}"
                f"&to={last.strftime('%Y-%m-%d')}&limit=500",
                API_TIMEOUT_S,
            )
            items = payload.get("events", []) if isinstance(payload, dict) else []
            for item in items if isinstance(items, list) else []:
                event = self._parse_range_item(item)
                if event is not None and from_ms <= event.event_ms <= to_ms:
                    events.append(event)
            start = end + 1
        # Deduplicate (overlapping chunks/duplicate feed rows) by event key.
        seen: dict[str, ScheduledEvent] = {}
        for event in events:
            seen.setdefault(event.key, event)
        self._events = sorted(seen.values(), key=lambda e: e.event_ms)
        self._month_cache.clear()
        logger.info("[Events] range loaded: %d blackout releases", len(self._events))
        return self._events

    @staticmethod
    def _parse_range_item(item: Any) -> ScheduledEvent | None:
        if not isinstance(item, dict) or item.get("all_day"):
            return None
        event_ms = _parse_ms(item.get("time_utc"))
        if event_ms is None:
            return None
        name = str(item.get("name") or "?")
        title = str(item.get("title") or name)
        window = EventCalendar._match_window(name, title)
        if window is None:
            return None
        before_ms, after_ms = window
        lower = f"{name} {title}".lower()
        if "fomc" in lower:
            series = "fomc"
        elif "non-farm" in lower or "nonfarm" in lower or "employment" in lower:
            series = "nfp"
        else:
            series = "cpi"
        return ScheduledEvent(
            series=series,
            name=name,
            title=title,
            event_ms=event_ms,
            impact=str(item.get("impact") or ""),
            url=str(item.get("url") or ""),
            consensus=_clip300(item.get("consensus")),
            prior=_clip300(item.get("prior")),
            actual=_clip300(item.get("actual")),
            source="api-range",
            window_before_ms=before_ms,
            window_after_ms=after_ms,
        )

    def frozen(self) -> list[dict[str, Any]]:
        """JSON-serializable snapshot of the cache (benchmark artifacts)."""
        try:
            return [e.to_dict() for e in self._events]
        except Exception as exc:
            logger.warning("[Events] frozen failed: %s", exc)
            return []

    def month_events(self, year: int, month: int) -> dict[str, list[dict[str, Any]]]:
        """Timed + all-day releases grouped by ``YYYY-MM-DD`` (tab UI)."""
        cache_key = f"{int(year):04d}-{int(month):02d}"
        if cache_key in self._month_cache:
            return self._month_cache[cache_key]
        try:
            first = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
            nxt = datetime(
                first.year + (1 if first.month == 12 else 0),
                1 if first.month == 12 else first.month + 1,
                1,
                tzinfo=timezone.utc,
            )
            last = nxt - timedelta(days=1)
            payload = self._fetcher(
                f"{API_BASE}/calendar?from={first.strftime('%Y-%m-%d')}"
                f"&to={last.strftime('%Y-%m-%d')}&limit=500",
                API_TIMEOUT_S,
            )
            days = self._group_month(payload)
            self._source = "live"
        except Exception as exc:
            logger.warning(
                "[Events] month fetch failed, falling back to static file: %s", exc
            )
            days = self._fallback_month(int(year), int(month))
            self._source = "fallback"
        self._month_cache[cache_key] = days
        return days

    @staticmethod
    def _group_month(payload: Any) -> dict[str, list[dict[str, Any]]]:
        days: dict[str, list[dict[str, Any]]] = {}
        items = payload.get("events", []) if isinstance(payload, dict) else []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            event_ms = _parse_ms(item.get("time_utc"))
            day = str(item.get("date") or "")
            entry: dict[str, Any] = {
                "name": str(item.get("name") or "?"),
                "title": str(item.get("title") or item.get("name") or "?"),
                "impact": str(item.get("impact") or ""),
                "category": str(item.get("category") or ""),
                "url": str(item.get("url") or ""),
                "consensus": item.get("consensus"),
                "prior": item.get("prior"),
                "actual": item.get("actual"),
                "all_day": bool(item.get("all_day")),
                "trading_paused": False,
            }
            if event_ms is not None:
                entry["event_ms"] = event_ms
                entry["event_vn"] = format_vn(event_ms)
                entry["event_utc"] = datetime.fromtimestamp(
                    event_ms / 1000.0, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC")
                day = datetime.fromtimestamp(
                    event_ms / 1000.0, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                paused = EventCalendar._match_window(entry["name"], entry["title"])
                if paused is not None:
                    before_ms, after_ms = paused
                    entry["trading_paused"] = True
                    entry["blackout_start_vn"] = format_vn(event_ms - before_ms)
                    entry["blackout_end_vn"] = format_vn(event_ms + after_ms)
            if not day:
                continue
            days.setdefault(day, []).append(entry)
        for entries in days.values():
            entries.sort(key=lambda e: (e.get("event_ms") or 0))
        return days

    @staticmethod
    def _match_window(name: str, title: str) -> tuple[int, int] | None:
        """Map a calendar entry to a blackout window (ms), or None.

        Matches the same three series the scheduler gates on: FOMC decisions,
        US CPI and NFP. Everything else (jobless claims, PMI, foreign CPI...)
        is informational only.
        """
        text = f"{name} {title}".lower()
        if "fomc" in text and "minute" not in text and "beige" not in text:
            before_h, after_h = SERIES_WINDOWS_H["fomc"]
            return (int(before_h * 3_600_000), int(after_h * 3_600_000))
        if "non-farm" in text or "nonfarm" in text or "employment situation" in text:
            before_h, after_h = SERIES_WINDOWS_H["nfp"]
            return (int(before_h * 3_600_000), int(after_h * 3_600_000))
        us_markers = ("united states", "u.s.", " u.s ", " us ", "us cpi", "us cpi report")
        if ("cpi" in text or "consumer price" in text) and any(
            m in text or text.startswith("us ") or text.startswith("u.s.")
            for m in us_markers
        ):
            before_h, after_h = SERIES_WINDOWS_H["cpi"]
            return (int(before_h * 3_600_000), int(after_h * 3_600_000))
        return None

    def _fallback_month(self, year: int, month: int) -> dict[str, list[dict[str, Any]]]:
        prefix = f"{year:04d}-{month:02d}"
        days: dict[str, list[dict[str, Any]]] = {}
        for event in self._events:
            day = datetime.fromtimestamp(
                event.event_ms / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            if not day.startswith(prefix):
                continue
            entry = event.to_dict()
            entry["category"] = "fallback"
            days.setdefault(day, []).append(entry)
        return days

    # -- prompt note ------------------------------------------------------

    def event_note(self, candles: list[Any] | None, now_ms: int) -> str | None:
        """Build the Phase N context note (event candle + upcoming release).

        Only closed candles are ever referenced, so this is look-ahead safe in
        both live and backtest use. Returns None when nothing is relevant.
        """
        try:
            return self._event_note(candles, int(now_ms))
        except Exception as exc:  # never break the analysis path
            logger.warning("[Events] event_note failed: %s", exc)
            return None

    def _event_note(self, candles: list[Any] | None, now_ms: int) -> str | None:
        parts: list[str] = []
        if candles:
            oldest_open = int(candles[0].timestamp)
            newest_close = int(candles[-1].timestamp) + 3_600_000
            for event in self._events:
                if oldest_open <= event.event_ms < newest_close:
                    holder = next(
                        (
                            c
                            for c in candles
                            if int(c.timestamp)
                            <= event.event_ms
                            < int(c.timestamp) + 3_600_000
                        ),
                        None,
                    )
                    if holder is None:
                        continue
                    stamp = datetime.fromtimestamp(
                        event.event_ms / 1000.0, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                    candle_iso = datetime.fromtimestamp(
                        int(holder.timestamp) / 1000.0, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                    move = ""
                    try:
                        o, h, low, c = (
                            float(holder.open),
                            float(holder.high),
                            float(holder.low),
                            float(holder.close),
                        )
                        if o > 0:
                            move = f" (range {(h - low) / o * 100.0:+.2f}%, close {(c - o) / o * 100.0:+.2f}%)"
                    except (TypeError, ValueError, AttributeError):
                        move = ""
                    parts.append(
                        f"Event candle: {event.title} released {stamp} inside the "
                        f"{candle_iso} 1H candle{move}. Read that candle as the "
                        "market's reaction to the release, not as normal momentum."
                    )
        for event in self._events:
            delta_h = (event.event_ms - now_ms) / 3_600_000.0
            if 0 < delta_h <= UPCOMING_NOTE_HOURS:
                stamp = datetime.fromtimestamp(
                    event.event_ms / 1000.0, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC")
                parts.append(
                    f"Upcoming: {event.title} in {delta_h:.1f}h ({stamp}). Expect "
                    "elevated volatility into the release; prefer WAIT unless "
                    "the technical setup is decisive."
                )
        if not parts:
            return None
        return "Scheduled-event note: " + " ".join(parts)


__all__ = [
    "API_BASE",
    "API_TIMEOUT_S",
    "DEFAULT_WARN_HOURS",
    "ENV_EVENT_WARN_HOURS",
    "SERIES_WINDOWS_H",
    "UPCOMING_NOTE_HOURS",
    "EventCalendar",
    "EventCheck",
    "ScheduledEvent",
    "format_vn",
    "warn_hours_from_env",
]
