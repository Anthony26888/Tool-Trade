"""Phase N tests: scheduled macro events (FOMC/CPI/NFP blackout + notices).

Covers the event calendar source (API-first with static fallback, never
raising), the scheduler blackout gate (by wall-clock now, not candle time),
once-per-event Telegram transitions (restart-safe), PENDING cancellation at
blackout start, the prompt event note, the backtest gate + wiring, and the
web endpoints + static UI markers. No test touches the real network.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

import pytest

from backtest.benchmark.runner import build_event_wiring
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from binance.market_data import Candle
from database.database import CandleLogRepository, Database
from database.models import STATUS_OPEN, STATUS_PENDING_ENTRY
from signal_engine import OneHourScheduler, SchedulerOutcome
from signal_engine.context import build_analysis_context
from signal_engine.event_calendar import (
    EventCalendar,
    ScheduledEvent,
    format_vn,
    warn_hours_from_env,
)
from tests.backtest_helpers import covering_minutes, dataset, make_hours
from tests.signal_engine_test_helpers import (
    INTERVAL_MS,
    TempSignalDb,
    make_analysis,
    make_candles,
)
from web.server import WebApplication

FOMC_ISO = "2026-10-28T18:00:00+00:00"
FOMC_MS = int(
    datetime(2026, 10, 28, 18, 0, tzinfo=timezone.utc).timestamp() * 1000
)
HOUR_MS = 3_600_000


def dt_ms(year, month, day, hour=0, minute=0) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()
        * 1000
    )


def next_payload(mapping: dict[str, str]) -> dict:
    """Fake ``/next`` fetcher serving fixed ISO times per series."""

    def fetch(url: str, timeout: float) -> dict:
        fetch.calls.append(url)
        series = url.split("series=")[1].split("&")[0]
        return {
            "next": {
                "name": series.upper(),
                "title": f"{series.upper()} test release",
                "time_utc": mapping[series],
                "impact": "high",
                "all_day": False,
                "url": "https://example.invalid/e",
            }
        }

    fetch.calls = []
    return fetch


def calendar_payload(items: list[dict]) -> dict:
    def fetch(url: str, timeout: float) -> dict:
        fetch.calls.append(url)
        if "/calendar" in url:
            return {"events": items}
        series = url.split("series=")[1].split("&")[0]
        raise AssertionError(f"unexpected /next call in range test: {series}")

    fetch.calls = []
    return fetch


class FakeNotifier:
    """Captures Telegram ``send_message`` texts (never raises)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, text: str):
        self.messages.append(text)
        from types import SimpleNamespace

        return SimpleNamespace(ok=True)


# ── N1/N2: calendar source ───────────────────────────────────────────────


@pytest.mark.unit
class TestEventCalendar(unittest.TestCase):
    def test_active_and_upcoming(self):
        cal = EventCalendar(
            fetcher=next_payload({"fomc": FOMC_ISO, "cpi": FOMC_ISO, "nfp": FOMC_ISO}),
            warn_hours=12,
        )
        status = cal.check(FOMC_MS - HOUR_MS)
        self.assertIsNotNone(status.active)
        self.assertEqual(status.active.series, "fomc")
        self.assertEqual(
            [e.series for e in status.upcoming], ["fomc", "cpi", "nfp"]
        )

    def test_refresh_throttled(self):
        fetch = next_payload({"fomc": FOMC_ISO, "cpi": FOMC_ISO, "nfp": FOMC_ISO})
        cal = EventCalendar(fetcher=fetch, refresh_interval_s=3600)
        cal.check(FOMC_MS)
        first_calls = len(fetch.calls)
        self.assertEqual(first_calls, 3)
        cal.check(FOMC_MS + 60_000)
        self.assertEqual(len(fetch.calls), first_calls)
        cal.refresh(FOMC_MS + 2 * HOUR_MS, force=True)
        self.assertEqual(len(fetch.calls), first_calls + 3)

    def test_blackout_boundaries(self):
        cal = EventCalendar(
            fetcher=next_payload({"fomc": FOMC_ISO, "cpi": FOMC_ISO, "nfp": FOMC_ISO}),
            warn_hours=12,
        )
        cal.refresh(FOMC_MS, force=True)
        start = FOMC_MS - 3 * HOUR_MS
        end = FOMC_MS + 3 * HOUR_MS
        self.assertIsNone(cal.blackout_at(start - 1))
        self.assertIsNotNone(cal.blackout_at(start))
        self.assertIsNotNone(cal.blackout_at(end))
        self.assertIsNone(cal.blackout_at(end + 1))

    def test_fallback_file_when_api_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fallback.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "events": [
                            {
                                "series": "fomc",
                                "name": "FOMC decision",
                                "title": "FOMC fallback",
                                "time_utc": FOMC_ISO,
                                "impact": "high",
                            }
                        ]
                    },
                    handle,
                )

            def boom(url: str, timeout: float):
                raise ConnectionError("feed down")

            cal = EventCalendar(fetcher=boom, fallback_path=path, warn_hours=12)
            status = cal.check(FOMC_MS)
            self.assertIsNotNone(status.active)
            self.assertEqual(status.active.source, "fallback")
            self.assertEqual(status.active.title, "FOMC fallback")

    def test_total_failure_is_fail_open(self):
        def boom(url: str, timeout: float):
            raise ConnectionError("feed down")

        cal = EventCalendar(
            fetcher=boom, fallback_path="/nonexistent/ev.json", warn_hours=12
        )
        status = cal.check(FOMC_MS)
        self.assertIsNone(status.active)
        self.assertEqual(status.upcoming, [])
        self.assertIsNone(cal.blackout_at(FOMC_MS))

    def test_all_day_and_bad_rows_skipped(self):
        def fetch(url: str, timeout: float):
            series = url.split("series=")[1].split("&")[0]
            if series == "fomc":
                return {"next": {"all_day": True, "time_utc": None}}
            if series == "cpi":
                return {"next": {"time_utc": "not-a-time"}}
            return {}

        cal = EventCalendar(
            fetcher=fetch, fallback_path="/nonexistent/ev.json", warn_hours=12
        )
        status = cal.check(FOMC_MS)
        self.assertIsNone(status.active)
        self.assertEqual(status.upcoming, [])

    def test_warn_hours_env(self):
        old = os.environ.get("BTCUSDT_EVENT_WARN_HOURS")
        try:
            os.environ["BTCUSDT_EVENT_WARN_HOURS"] = "6"
            self.assertEqual(warn_hours_from_env(), 6.0)
            os.environ["BTCUSDT_EVENT_WARN_HOURS"] = "abc"
            self.assertEqual(warn_hours_from_env(), 12.0)
            os.environ["BTCUSDT_EVENT_WARN_HOURS"] = "-3"
            self.assertEqual(warn_hours_from_env(), 0.0)
            del os.environ["BTCUSDT_EVENT_WARN_HOURS"]
            self.assertEqual(warn_hours_from_env(), 12.0)
        finally:
            if old is None:
                os.environ.pop("BTCUSDT_EVENT_WARN_HOURS", None)
            else:
                os.environ["BTCUSDT_EVENT_WARN_HOURS"] = old

    def test_format_vn(self):
        # 18:00 UTC 28/10 (EDT season) -> 01:00 29/10 +07.
        self.assertEqual(format_vn(FOMC_MS), "01:00 29/10 (+07)")


# ── month grid + matcher ─────────────────────────────────────────────────


@pytest.mark.unit
class TestMonthGrid(unittest.TestCase):
    def month_cal(self):
        items = [
            {
                "date": "2026-10-28",
                "time_utc": "2026-10-28T18:00:00+00:00",
                "all_day": False,
                "name": "FOMC decision",
                "title": "FOMC Rate Decision October 2026",
                "impact": "high",
                "category": "central-banks-monetary-policy",
                "url": "https://example.invalid/fomc",
            },
            {
                "date": "2026-10-14",
                "time_utc": "2026-10-14T12:30:00+00:00",
                "all_day": False,
                "name": "US CPI",
                "title": "US CPI October 2026",
                "impact": "high",
            },
            {
                "date": "2026-10-02",
                "time_utc": "2026-10-02T12:30:00+00:00",
                "all_day": False,
                "name": "US jobs report (NFP)",
                "title": "US Employment Situation (Non-Farm Payrolls) October 2026",
                "impact": "high",
            },
            {
                "date": "2026-10-08",
                "time_utc": "2026-10-08T12:30:00+00:00",
                "all_day": False,
                "name": "Jobless claims",
                "title": "US Initial Jobless Claims: October 8, 2026",
                "impact": "medium",
            },
            {
                "date": "2026-10-21",
                "time_utc": "2026-10-21T00:30:00+00:00",
                "all_day": False,
                "name": "Australia CPI",
                "title": "Australia CPI October 2026",
                "impact": "high",
            },
            {
                "date": "2026-10-07",
                "time_utc": None,
                "all_day": True,
                "name": "FOMC minutes",
                "title": "FOMC minutes October 2026",
                "impact": "high",
            },
        ]
        return EventCalendar(fetcher=calendar_payload(items), warn_hours=12)

    def test_trading_paused_flags(self):
        days = self.month_cal().month_events(2026, 10)
        fomc = days["2026-10-28"][0]
        self.assertTrue(fomc["trading_paused"])
        self.assertEqual(fomc["blackout_start_vn"], "22:00 28/10 (+07)")
        self.assertEqual(fomc["blackout_end_vn"], "04:00 29/10 (+07)")
        cpi = days["2026-10-14"][0]
        self.assertTrue(cpi["trading_paused"])
        nfp = days["2026-10-02"][0]
        self.assertTrue(nfp["trading_paused"])
        # Weekly jobless claims and foreign CPI stay informational.
        self.assertFalse(days["2026-10-08"][0]["trading_paused"])
        self.assertFalse(days["2026-10-21"][0]["trading_paused"])
        # All-day entries are shown but never pause trading.
        minutes = days["2026-10-07"][0]
        self.assertTrue(minutes["all_day"])
        self.assertFalse(minutes["trading_paused"])

    def test_month_cache(self):
        fetch = calendar_payload([])
        cal = EventCalendar(fetcher=fetch, warn_hours=12)
        cal.month_events(2026, 10)
        cal.month_events(2026, 10)
        self.assertEqual(len(fetch.calls), 1)


# ── range load + prompt note ─────────────────────────────────────────────


def hour_candle(ts: int, *, base: float = 60000.0) -> Candle:
    return Candle(
        timestamp=ts,
        open=base,
        high=base + 100.0,
        low=base - 100.0,
        close=base + 50.0,
        volume=10.0,
        close_time=ts + HOUR_MS - 1,
        is_closed=True,
    )


@pytest.mark.unit
class TestRangeAndNote(unittest.TestCase):
    def test_load_range_freeze_and_dedupe(self):
        items = [
            {
                "date": "2026-10-28",
                "time_utc": FOMC_ISO,
                "all_day": False,
                "name": "FOMC decision",
                "title": "FOMC Rate Decision October 2026",
                "impact": "high",
            },
            {
                "date": "2026-10-28",
                "time_utc": FOMC_ISO,
                "all_day": False,
                "name": "FOMC decision",
                "title": "FOMC Rate Decision October 2026",
                "impact": "high",
            },
            {
                "date": "2026-10-08",
                "time_utc": "2026-10-08T12:30:00+00:00",
                "all_day": False,
                "name": "Jobless claims",
                "title": "US Initial Jobless Claims",
                "impact": "medium",
            },
        ]
        cal = EventCalendar(fetcher=calendar_payload(items), warn_hours=12)
        events = cal.load_range(dt_ms(2026, 10, 1), dt_ms(2026, 10, 31))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].series, "fomc")
        frozen = cal.frozen()
        self.assertEqual(len(frozen), 1)
        self.assertTrue(frozen[0]["trading_paused"])
        roundtrip = json.loads(json.dumps(frozen))
        self.assertEqual(roundtrip[0]["event_ms"], FOMC_MS)

    def test_event_note_marks_event_candle(self):
        cal = EventCalendar(fetcher=calendar_payload([]), warn_hours=12)
        cal._events = [
            ScheduledEvent(
                series="fomc",
                name="FOMC decision",
                title="FOMC Rate Decision October 2026",
                event_ms=FOMC_MS,
                source="test",
                window_before_ms=3 * HOUR_MS,
                window_after_ms=3 * HOUR_MS,
            )
        ]
        candles = [
            hour_candle(FOMC_MS - 2 * HOUR_MS),
            hour_candle(FOMC_MS - HOUR_MS),
            hour_candle(FOMC_MS),  # release falls inside this candle
            hour_candle(FOMC_MS + HOUR_MS),
        ]
        note = cal.event_note(candles, FOMC_MS + 2 * HOUR_MS)
        self.assertIsNotNone(note)
        self.assertIn("Event candle", note)
        self.assertIn("FOMC Rate Decision October 2026", note)
        self.assertIn("2026-10-28 18:00 UTC", note)
        self.assertIn("range +0.33%", note)

    def test_event_note_upcoming_and_clean(self):
        cal = EventCalendar(fetcher=calendar_payload([]), warn_hours=12)
        cal._events = [
            ScheduledEvent(
                series="cpi",
                name="US CPI",
                title="US CPI October 2026",
                event_ms=dt_ms(2026, 10, 14, 12, 30),
                source="test",
                window_before_ms=2 * HOUR_MS,
                window_after_ms=2 * HOUR_MS,
            )
        ]
        candles = [hour_candle(dt_ms(2026, 10, 14, h)) for h in range(0, 4)]
        note = cal.event_note(candles, dt_ms(2026, 10, 14, 7, 30))
        self.assertIsNotNone(note)
        self.assertIn("Upcoming", note)
        self.assertIn("5.0h", note)
        # Far-away event with no window overlap: no note.
        quiet = cal.event_note(candles, dt_ms(2026, 10, 1))
        self.assertIsNone(quiet)


# ── N3/N5: scheduler gate + notices ──────────────────────────────────────


class FakeSchedulerData:
    def __init__(self, *candles) -> None:
        self.candles = list(candles)

    def fetch_closed_klines(
        self, symbol, interval, limit, *, end_time_ms=None, now_ms=None
    ):
        # Mirror the real contract: only candles closed as of now_ms.
        return [
            c
            for c in self.candles
            if c.is_closed and (now_ms is None or c.close_time <= now_ms)
        ]


def analyzer_recorder(decision="WAIT", calls=None):
    def analyzer(candles, indicators):
        calls.append(list(candles))
        return make_analysis(decision)

    return analyzer


@pytest.mark.unit
class TestSchedulerBlackout(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.now = datetime(2026, 10, 28, 17, 0, tzinfo=timezone.utc)
        self.now_ms = int(self.now.timestamp() * 1000)
        # Candles ending 16:00 UTC (all closed before the tick).
        self.candles = make_candles(
            220, start_ms=self.now_ms - 220 * INTERVAL_MS
        )
        self.md = FakeSchedulerData(*self.candles)

    def make_scheduler(self, **kwargs):
        kwargs.setdefault("market_data", self.md)
        kwargs.setdefault("engine", self.harness.engine)
        kwargs.setdefault("state", self.harness.state)
        kwargs.setdefault(
            "candle_log", CandleLogRepository(self.harness.db)
        )
        return OneHourScheduler(self.harness.repository, **kwargs)

    def fomc_calendar(self, **kwargs):
        kwargs.setdefault(
            "fetcher",
            next_payload(
                {
                    "fomc": FOMC_ISO,
                    "cpi": "2026-11-11T13:30:00+00:00",
                    "nfp": "2026-12-04T13:30:00+00:00",
                }
            ),
        )
        kwargs.setdefault("warn_hours", 12)
        return EventCalendar(**kwargs)

    def test_blackout_blocks_analysis_and_marks_candle(self):
        calls: list = []
        sched = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", calls),
            event_calendar=self.fomc_calendar(),
        )
        result = sched.tick(now=self.now)
        self.assertEqual(result.outcome, SchedulerOutcome.EVENT_BLACKOUT)
        self.assertEqual(calls, [])
        rows = CandleLogRepository(self.harness.db).list(limit=5)
        self.assertEqual(rows[0].outcome, "EVENT_BLACKOUT")
        # Marked: the same candle is never analyzed afterwards.
        again = sched.tick(now=self.now)
        self.assertEqual(again.outcome, SchedulerOutcome.ALREADY_PROCESSED)
        self.assertEqual(calls, [])

    def test_gate_uses_wall_clock_not_candle_time(self):
        # Newest closed candle (16:00) predates the blackout start (15:00)?
        # No: FOMC 18:00 +-3h = 15:00-21:00; candle 16:00 is INSIDE the
        # window, and now (17:00) is too. The point: even a candle whose own
        # data predates the release is skipped while now is blacked out.
        calls: list = []
        sched = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", calls),
            event_calendar=self.fomc_calendar(),
        )
        result = sched.tick(now=self.now)
        self.assertEqual(result.outcome, SchedulerOutcome.EVENT_BLACKOUT)
        self.assertEqual(result.candle_timestamp, self.candles[-1].timestamp)

    def test_no_calendar_keeps_old_behavior(self):
        calls: list = []
        sched = self.make_scheduler(analyzer=analyzer_recorder("WAIT", calls))
        result = sched.tick(now=self.now)
        self.assertEqual(result.outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(calls), 1)

    def test_warn_start_end_each_once_and_restart_safe(self):
        notifier = FakeNotifier()
        sched = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", []),
            event_calendar=self.fomc_calendar(),
            notifier=notifier,
        )
        # 13h before: nothing (warn window is 12h).
        sched.tick(
            now=datetime(2026, 10, 28, 5, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(notifier.messages, [])
        # 11h before: WARN once (repeat poll does not resend).
        warn_at = datetime(2026, 10, 28, 7, 0, tzinfo=timezone.utc)
        sched.tick(now=warn_at)
        sched.tick(now=warn_at)
        self.assertEqual(len(notifier.messages), 1)
        self.assertIn("FOMC", notifier.messages[0])
        # Blackout start: START once.
        start_at = datetime(2026, 10, 28, 15, 30, tzinfo=timezone.utc)
        first = sched.tick(now=start_at)
        self.assertEqual(first.outcome, SchedulerOutcome.EVENT_BLACKOUT)
        # Same candle again: marked, never re-analyzed.
        self.assertEqual(
            sched.tick(now=start_at).outcome,
            SchedulerOutcome.ALREADY_PROCESSED,
        )
        starts = [m for m in notifier.messages if m.startswith("⏸")]
        self.assertEqual(len(starts), 1)
        self.assertIn("0 lệnh PENDING", starts[0])
        # Blackout end: END once.
        end_at = datetime(2026, 10, 28, 21, 30, tzinfo=timezone.utc)
        sched.tick(now=end_at)
        ends = [m for m in notifier.messages if m.startswith("▶️")]
        self.assertEqual(len(ends), 1)
        # A restarted scheduler on the same DB resends nothing.
        sched2 = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", []),
            event_calendar=self.fomc_calendar(),
            notifier=notifier,
        )
        sched2.tick(now=end_at)
        self.assertEqual(len(notifier.messages), 3)

    def test_start_cancels_pending_but_never_open(self):
        repo = self.harness.repository
        pending = repo.create_signal(
            "BTCUSDT", "1h", "LONG", "61000", "60000", "64000",
            confidence=80,
        )
        self.assertEqual(pending.status, STATUS_PENDING_ENTRY)
        notifier = FakeNotifier()
        sched = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", []),
            event_calendar=self.fomc_calendar(),
            notifier=notifier,
        )
        sched.tick(now=datetime(2026, 10, 28, 15, 30, tzinfo=timezone.utc))
        refreshed = repo.get_signal(pending.id)
        self.assertNotEqual(refreshed.status, STATUS_PENDING_ENTRY)
        starts = [m for m in notifier.messages if m.startswith("⏸")]
        self.assertEqual(len(starts), 1)
        self.assertIn("1 lệnh PENDING", starts[0])

    def test_open_position_survives_blackout_and_lock_wins(self):
        # Fresh DB: the OPEN gate precedes the blackout gate, and an OPEN
        # position is never cancelled at blackout start.
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        repo = harness.repository
        opened = repo.create_signal(
            "BTCUSDT", "1h", "SHORT", "61000", "62000", "59000",
            confidence=80,
        )
        harness.state.transition(opened.id, STATUS_OPEN)
        md = FakeSchedulerData(*self.candles)
        notifier = FakeNotifier()
        sched = OneHourScheduler(
            repo,
            market_data=md,
            engine=harness.engine,
            state=harness.state,
            analyzer=analyzer_recorder("WAIT", []),
            candle_log=CandleLogRepository(harness.db),
            event_calendar=self.fomc_calendar(),
            notifier=notifier,
        )
        at = datetime(2026, 10, 28, 16, 30, tzinfo=timezone.utc)
        result = sched.tick(now=at)
        self.assertEqual(result.outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL)
        self.assertEqual(repo.get_signal(opened.id).status, STATUS_OPEN)
        # START notice still fires (0 PENDING cancelled), exactly once.
        starts = [m for m in notifier.messages if m.startswith("⏸")]
        self.assertEqual(len(starts), 1)
        self.assertIn("0 lệnh PENDING", starts[0])

    def test_no_notifier_never_crashes(self):
        sched = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", []),
            event_calendar=self.fomc_calendar(),
            notifier=None,
        )
        result = sched.tick(now=self.now)
        self.assertEqual(result.outcome, SchedulerOutcome.EVENT_BLACKOUT)

    def test_warn_disabled_but_start_end_sent(self):
        notifier = FakeNotifier()
        sched = self.make_scheduler(
            analyzer=analyzer_recorder("WAIT", []),
            event_calendar=self.fomc_calendar(warn_hours=0),
            notifier=notifier,
        )
        sched.tick(now=datetime(2026, 10, 28, 7, 0, tzinfo=timezone.utc))
        self.assertEqual(notifier.messages, [])
        sched.tick(now=datetime(2026, 10, 28, 15, 30, tzinfo=timezone.utc))
        self.assertEqual(len(notifier.messages), 1)
        self.assertTrue(notifier.messages[0].startswith("⏸"))


# ── N4: prompt note ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestEventNoteContext(unittest.TestCase):
    def test_note_section_rendered(self):
        candles = make_candles(220)
        from binance.indicators import compute_indicator_matrix

        indicators = compute_indicator_matrix(candles)
        plain = build_analysis_context(candles, indicators)
        self.assertNotIn("Scheduled event note", plain.rendered)
        noted = build_analysis_context(
            candles, indicators, event_note="Scheduled-event note: test"
        )
        self.assertIn("## Scheduled event note", noted.rendered)
        self.assertIn("Scheduled-event note: test", noted.rendered)

    def test_blank_note_is_ignored(self):
        candles = make_candles(220)
        from binance.indicators import compute_indicator_matrix

        indicators = compute_indicator_matrix(candles)
        noted = build_analysis_context(candles, indicators, event_note="   ")
        self.assertNotIn("Scheduled event note", noted.rendered)


# ── N7: backtest gate + wiring ───────────────────────────────────────────


def _wait_provider():
    from types import SimpleNamespace

    return SimpleNamespace(
        provider="test",
        model="test",
        decide=lambda *a, **k: make_analysis("WAIT"),
    )


def _backtest_config(min_candles: int = 200) -> BacktestConfig:
    from decimal import Decimal

    return BacktestConfig(
        symbol="BTCUSDT",
        timeframe="1h",
        execution_interval="1m",
        initial_balance=Decimal("1000"),
        margin_per_trade=Decimal("50"),
        leverage=10,
        risk_percent=Decimal("1"),
        fee_rate=Decimal("0.0004"),
        slippage_bps=Decimal("0"),
        min_candles=min_candles,
    )


@pytest.mark.unit
class TestBacktestBlackout(unittest.TestCase):
    def test_blackout_candles_blocked_not_analyzed(self):
        hours = make_hours(205)
        data = dataset(hours, covering_minutes(hours))
        period = HOUR_MS
        first_eligible = 200 - 1
        blocked_ts = hours[first_eligible + 1].timestamp + period

        def blackout(ms: int):
            return "FOMC test" if ms == blocked_ts else None

        engine = BacktestEngine(_backtest_config(), data, _wait_provider(), blackout=blackout)
        result = engine.run()
        self.assertEqual(result.blocked_count, 1)
        self.assertEqual(result.analyzed, 5)
        reasons = [e.reason for e in result.blocked_events]
        self.assertTrue(any("event blackout: FOMC test" in r for r in reasons))

    def test_no_blackout_fn_unchanged(self):
        hours = make_hours(205)
        data = dataset(hours, covering_minutes(hours))
        engine = BacktestEngine(_backtest_config(), data, _wait_provider())
        result = engine.run()
        self.assertEqual(result.blocked_count, 0)
        self.assertEqual(result.analyzed, 6)

    def test_build_event_wiring(self):
        items = [
            {
                "date": "2026-10-28",
                "time_utc": FOMC_ISO,
                "all_day": False,
                "name": "FOMC decision",
                "title": "FOMC Rate Decision October 2026",
                "impact": "high",
            }
        ]
        cal = EventCalendar(fetcher=calendar_payload(items), warn_hours=12)
        seen: dict = {}

        def base(candles, indicators, *, symbol, timeframe, max_candles, event_note=None):
            seen["note"] = event_note
            return make_analysis("WAIT")

        hours = [hour_candle(FOMC_MS - 2 * HOUR_MS + i * HOUR_MS) for i in range(4)]
        blackout_fn, analyze_fn, frozen = build_event_wiring(
            event_calendar=cal,
            hours=hours,
            period_ms=HOUR_MS,
            analyze_base=base,
        )
        self.assertEqual(len(frozen), 1)
        self.assertEqual(
            blackout_fn(FOMC_MS), "FOMC Rate Decision October 2026"
        )
        self.assertIsNone(blackout_fn(FOMC_MS - 10 * HOUR_MS))
        analyze_fn(hours, None, symbol="BTCUSDT", timeframe="1h", max_candles=4)
        self.assertIsNotNone(seen["note"])
        self.assertIn("Event candle", seen["note"])


# ── source tracking ────────────────────────────────────────────────────


@pytest.mark.unit
class TestCalendarSource(unittest.TestCase):
    def test_live_source_after_api_success(self):
        cal = EventCalendar(
            fetcher=next_payload({"fomc": FOMC_ISO, "cpi": FOMC_ISO, "nfp": FOMC_ISO}),
            warn_hours=12,
        )
        self.assertEqual(cal.source, "none")
        cal.check(FOMC_MS)
        self.assertEqual(cal.source, "live")

    def test_fallback_source_when_api_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fallback.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "events": [
                            {
                                "series": "fomc",
                                "name": "FOMC decision",
                                "title": "FOMC fallback",
                                "time_utc": FOMC_ISO,
                                "impact": "high",
                            }
                        ]
                    },
                    handle,
                )

            def boom(url: str, timeout: float):
                raise ConnectionError("feed down")

            cal = EventCalendar(fetcher=boom, fallback_path=path, warn_hours=12)
            cal.check(FOMC_MS)
            self.assertEqual(cal.source, "fallback")
            self.assertIn("2026-10-28", cal.month_events(2026, 10))
            self.assertEqual(cal.source, "fallback")

    def test_recovery_flips_back_to_live(self):
        calls = {"n": 0}

        def flaky(url: str, timeout: float):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise ConnectionError("down")
            return next_payload({"fomc": FOMC_ISO, "cpi": FOMC_ISO, "nfp": FOMC_ISO})(
                url, timeout
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fallback.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"events": []}, handle)
            cal = EventCalendar(fetcher=flaky, fallback_path=path, warn_hours=12)
            cal.refresh(FOMC_MS, force=True)
            self.assertEqual(cal.source, "none")
            cal.refresh(FOMC_MS, force=True)
            self.assertEqual(cal.source, "live")


# ── N6/N8: web endpoints + static UI ────────────────────────────────────


@pytest.mark.unit
class TestWebEvents(unittest.TestCase):
    def setUp(self):
        import urllib.request

        from web.server import WebApplication, WebServer

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "web_events.db"))
        self.db.initialize()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        upcoming_iso = (
            datetime.fromtimestamp(
                (now_ms + 5 * HOUR_MS) / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        def fetch(url: str, timeout: float):
            if "/calendar" in url:
                return {
                    "events": [
                        {
                            "date": "2026-10-28",
                            "time_utc": FOMC_ISO,
                            "all_day": False,
                            "name": "FOMC decision",
                            "title": "FOMC Rate Decision October 2026",
                            "impact": "high",
                        }
                    ]
                }
            return {
                "next": {
                    "name": "FOMC decision",
                    "title": "FOMC Rate Decision October 2026",
                    "time_utc": upcoming_iso,
                    "impact": "high",
                    "all_day": False,
                }
            }

        self.app = WebApplication(
            self.db, event_calendar=EventCalendar(fetcher=fetch, warn_hours=12)
        )
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.bound_port}"
        self._urllib = urllib.request

    def _get(self, path: str):
        with self._urllib.urlopen(self.base + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_events_endpoint(self):
        status, payload = self._get("/api/events")
        self.assertEqual(status, 200)
        self.assertIsNone(payload["active"])
        by_series = {e["series"]: e for e in payload["upcoming"]}
        self.assertEqual(
            by_series["fomc"]["title"], "FOMC Rate Decision October 2026"
        )
        self.assertTrue(by_series["fomc"]["trading_paused"])
        self.assertIn("event_vn", by_series["fomc"])
        self.assertEqual(payload["source"], "live")

    def test_events_calendar_endpoint(self):
        status, payload = self._get("/api/events/calendar?month=2026-10")
        self.assertEqual(status, 200)
        self.assertEqual(payload["month"], "2026-10")
        self.assertIn("2026-10-28", payload["days"])
        fomc = payload["days"]["2026-10-28"][0]
        self.assertTrue(fomc["trading_paused"])
        self.assertIn("financecalendar.com", payload["attribution"])
        self.assertEqual(payload["source"], "live")

    def test_events_calendar_bad_month_falls_back(self):
        status, payload = self._get("/api/events/calendar?month=nope")
        self.assertEqual(status, 200)
        self.assertIn("days", payload)

    def test_fallback_source_reported(self):
        def boom(url: str, timeout: float):
            raise ConnectionError("feed down")

        app = WebApplication(
            self.db,
            event_calendar=EventCalendar(
                fetcher=boom, fallback_path="/nonexistent/ev.json", warn_hours=12
            ),
        )
        payload = app.events()
        self.assertEqual(payload["source"], "none")
        month = app.events_calendar("2026-10")
        self.assertEqual(month["source"], "fallback")
        self.assertEqual(month["days"], {})

    def test_static_markers(self):
        def raw(path: str) -> str:
            with self._urllib.urlopen(self.base + path, timeout=10) as resp:
                return resp.read().decode("utf-8")

        html = raw("/")
        for marker in (
            'id="eventBanner"',
            'id="panel-events"',
            'id="calGrid"',
            'id="eventDialog"',
            'id="eventsBtn"',
            'id="calSource"',
            'data-panel="events"',
            "financecalendar.com",
        ):
            self.assertIn(marker, html)
        js = raw("/static/app.js")
        for marker in (
            "loadEventBanner",
            "loadEvents",
            "renderEventsCalendar",
            "openEventDialog",
            "shortEventName",
            "calSource",
            "/api/events/calendar",
            '"events"',
        ):
            self.assertIn(marker, js)
        css = raw("/static/style.css")
        for marker in (
            ".event-banner",
            ".cal-grid",
            ".cal-day",
            ".cal-chip",
            ".event-dialog",
            ".badge.event-blackout",
            "minmax(0, 1fr)",
            "min-width: 0",
        ):
            self.assertIn(marker, css)

    def test_short_event_name(self):
        import shutil
        import subprocess

        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        with self._urllib.urlopen(self.base + "/static/app.js", timeout=10) as resp:
            source = resp.read().decode("utf-8")
        start = source.index("function shortEventName(")
        depth = 0
        for pos in range(start, len(source)):
            if source[pos] == "{":
                depth += 1
            elif source[pos] == "}":
                depth -= 1
                if depth == 0:
                    fn = source[start : pos + 1]
                    break
        program = (
            fn + "\nconsole.log(JSON.stringify(["
            "shortEventName('FOMC Rate Decision October 2026'),"
            "shortEventName('US Employment Situation (Non-Farm Payrolls) October 2026'),"
            "shortEventName('CPI'),"
            "]));"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-1000:])
        short, clipped, tiny = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(short, "FOMC Rate Decision")
        self.assertTrue(len(clipped) <= 22 and clipped.endswith("…"))
        self.assertEqual(tiny, "CPI")


if __name__ == "__main__":
    unittest.main()
