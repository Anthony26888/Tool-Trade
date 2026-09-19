"""Phase 7 unit tests for the TP/SL Monitor (signal_engine/monitor.py).

Covers every requirement: LONG/SHORT exits, same-candle ambiguity, forming
candle exclusion, no-AI, no-order execution, restart resumption, atomic close,
immutability, data failure safety, Decimal comparisons, and concurrency.
"""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from unittest.mock import patch

import pytest

from binance.market_data import Candle
from database.database import Database, SignalRepository
from database.models import (
    STATUS_PENDING_ENTRY,
    STATUS_SL_HIT,
    STATUS_TP_HIT,
    Signal,
)
from signal_engine import MonitorOutcome, SignalMonitor
from signal_engine.monitor import (
    CLOSE_REASON_SL,
    CLOSE_REASON_TP,
    MONITOR_INTERVAL,
    evaluate_candle,
    evaluate_entry,
)
from signal_engine.state import STATUS_OPEN
from tests.signal_engine_test_helpers import TempSignalDb, make_analysis


def _candle(high, low, *, closed: bool = True, ts: int = 1_000_000) -> Candle:
    """Build a fake candle with a controlled closed flag (no network)."""
    high_f, low_f = float(high), float(low)
    mid = (high_f + low_f) / 2.0
    close_time = ts + (59_999 if closed else 60_000)
    return Candle(
        timestamp=ts,
        open=mid,
        high=high_f,
        low=low_f,
        close=mid,
        volume=1.0,
        close_time=close_time,
        is_closed=closed,
    )


class FakeMarketData:
    """Deterministic replacement for BinanceMarketData (never touches the network)."""

    def __init__(self, *candles: Candle) -> None:
        self.candles = list(candles)
        self.calls: list[tuple[str, str, int, int | None]] = []

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        end_time_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[Candle]:
        self.calls.append((symbol, interval, limit, now_ms))
        return list(self.candles)


class MonitorTestCase(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)

    def _create_long(self, **overrides):
        result = self.harness.engine.process(make_analysis("LONG", **overrides))
        self.assertEqual(result.outcome.value, "CREATED")
        return result.signal

    def _create_short(self, **overrides):
        from signal_engine import GuardrailConfig

        defaults = {"entry_price": 59000.0, "stop_loss": 60000.0, "take_profit": 58000.0}
        defaults.update(overrides)
        # Guardrails relaxed on purpose: monitor tests engineer boundary
        # candles around these symmetric levels; policy gets dedicated tests.
        relaxed = GuardrailConfig(
            min_confidence=0,
            min_risk_reward=Decimal("0"),
            fee_rate=Decimal("0"),
        )
        result = self.harness.engine.process(
            make_analysis("SHORT", **defaults), guardrails=relaxed
        )
        self.assertEqual(result.outcome.value, "CREATED")
        return result.signal

    def _promote_to_open(self, signal: Signal) -> Signal:
        return self.harness.state.transition(signal.id, STATUS_OPEN)

    def _open_long(self, **overrides):
        return self._promote_to_open(self._create_long(**overrides))

    def _open_short(self, **overrides):
        return self._promote_to_open(self._create_short(**overrides))


# ── A. LONG ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLongExit(MonitorTestCase):
    def test_long_tp_hit(self):
        sig = self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(65000, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(r.signal.close_price, Decimal("64000.0"))
        self.assertEqual(r.signal.close_reason, CLOSE_REASON_TP)
        self.assertEqual(r.signal.status, STATUS_TP_HIT)
        self.assertIsNotNone(r.signal.closed_at)
        self.assertEqual(r.signal.id, sig.id)

    def test_long_sl_hit(self):
        self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(60500, 59000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.SL_HIT)
        self.assertEqual(r.signal.close_price, Decimal("60000.0"))
        self.assertEqual(r.signal.close_reason, CLOSE_REASON_SL)
        self.assertEqual(r.signal.status, STATUS_SL_HIT)

    def test_long_no_hit(self):
        self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(63000, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(r.signal.status, "OPEN")
        self.assertIsNone(r.signal.close_price)

    def test_long_tp_boundary(self):
        self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(64000, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)

    def test_long_sl_boundary(self):
        self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(63000, 60000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.SL_HIT)


# ── B. SHORT ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestShortExit(MonitorTestCase):
    def test_short_tp_hit(self):
        self._open_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(59500, 57500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(r.signal.close_price, Decimal("58000.0"))
        self.assertEqual(r.signal.close_reason, CLOSE_REASON_TP)
        self.assertEqual(r.signal.status, STATUS_TP_HIT)

    def test_short_sl_hit(self):
        self._open_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(60500, 58500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.SL_HIT)
        self.assertEqual(r.signal.close_price, Decimal("60000.0"))
        self.assertEqual(r.signal.close_reason, CLOSE_REASON_SL)
        self.assertEqual(r.signal.status, STATUS_SL_HIT)

    def test_short_no_hit(self):
        self._open_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(59500, 58500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(r.signal.status, "OPEN")

    def test_short_tp_boundary(self):
        self._open_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(59500, 58000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)

    def test_short_sl_boundary(self):
        self._open_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(60000, 58500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.SL_HIT)


# ── C. Ambiguity ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAmbiguity(MonitorTestCase):
    def test_long_same_candle_ambiguity(self):
        sig = self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(68000, 55000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.AMBIGUOUS)
        sig_after = self.harness.repository.get_signal(sig.id)
        self.assertEqual(sig_after.status, "OPEN")
        self.assertIsNone(sig_after.closed_at)
        self.assertIsNone(sig_after.close_price)

    def test_short_same_candle_ambiguity(self):
        sig = self._open_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(62000, 57000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.AMBIGUOUS)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, "OPEN")


# ── D. Candle state ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCandleState(MonitorTestCase):
    def test_forming_candle_not_used_for_ohlc_confirmation(self):
        self._open_long()
        forming = _candle(68000, 55000, closed=False)
        closed = _candle(63000, 60500, closed=True)
        md = FakeMarketData(forming, closed)
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(self.harness.repository.get_open_signal().status, "OPEN")

    def test_only_forming_candle_yields_data_unavailable(self):
        self._open_long()
        md = FakeMarketData(_candle(68000, 55000, closed=False))
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.DATA_UNAVAILABLE)

    def test_real_parse_path_forming_excluded(self):
        """End-to-end: real parse_klines + real BinanceMarketData classification."""

        sig = self._open_long()
        closed_time1 = 1_000_000 + 59_999
        open_time2 = 1_000_000 + 60_000
        close_time2 = open_time2 + 59_999
        raw = [
            [1_000_000, 61000.0, 63000.0, 60500.0, 62000.0, 100.0, closed_time1, 1000.0, 100, 50.0, 500.0, "0"],
            [open_time2, 61500.0, 68000.0, 55000.0, 62000.0, 100.0, close_time2, 1000.0, 100, 50.0, 500.0, "0"],
        ]
        now_ms = close_time2  # candle 2 is forming at exactly this time

        received_now: list[int | None] = []

        class InliningBinance:
            def fetch_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
                received_now.append(now_ms)
                from binance.market_data import parse_klines

                return parse_klines(raw, now_ms)

        r = SignalMonitor(self.harness.repository, InliningBinance()).poll(now_ms=now_ms)
        self.assertEqual(received_now, [now_ms])
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, "OPEN")


# ── E. No OPEN ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNoOpen(MonitorTestCase):
    def test_no_open_signal_no_fetch_no_create(self):
        md = FakeMarketData()
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(md.calls, [])
        self.assertEqual(self.harness.repository.list_signals(), [])

    def test_no_llm_called(self):
        md = FakeMarketData()
        with patch("signal_engine.analysis.SignalAnalyzer") as sa, patch(
            "signal_engine.analysis.analyze_signal"
        ) as af:
            SignalMonitor(self.harness.repository, md).poll()
        sa.assert_not_called()
        af.assert_not_called()
        self.assertEqual(self.harness.repository.list_signals(), [])


# ── F. Restart / idempotency ────────────────────────────────────────────────


@pytest.mark.unit
class TestRestart(MonitorTestCase):
    def test_resume_after_reopen(self):
        sig = self._open_long()
        reopened_db = Database(self.harness.path)
        reopened_db.initialize()
        repo2 = SignalRepository(reopened_db)
        md = FakeMarketData(_candle(65000, 60500))
        r = SignalMonitor(repo2, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_TP_HIT)
        reopened_db.connect().close()

    def test_already_closed_not_reopened(self):
        sig = self._open_short()
        SignalMonitor(self.harness.repository, FakeMarketData(_candle(60500, 58500))).poll()
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT)
        r2 = SignalMonitor(self.harness.repository, FakeMarketData(_candle(100_000, 10_000))).poll()
        self.assertEqual(r2.outcome, MonitorOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT)

    def test_repeated_poll_when_idle(self):
        md = FakeMarketData()
        for _ in range(3):
            r = SignalMonitor(self.harness.repository, md).poll()
            self.assertEqual(r.outcome, MonitorOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(md.calls, [])


# ── G. Atomic transition ────────────────────────────────────────────────────


@pytest.mark.unit
class TestAtomicTransition(MonitorTestCase):
    def test_tp_closes_open(self):
        sig = self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(65000, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(r.signal.id, sig.id)
        self.assertEqual(r.signal.status, STATUS_TP_HIT)

    def test_sl_closes_open(self):
        sig = self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(60500, 59000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.SL_HIT)
        self.assertEqual(r.signal.id, sig.id)
        self.assertEqual(r.signal.status, STATUS_SL_HIT)

    def test_invalid_transition_rejected_safely(self):
        sig = self._open_long()
        SignalMonitor(self.harness.repository, FakeMarketData(_candle(60500, 59000))).poll()
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT)
        r2 = SignalMonitor(self.harness.repository, FakeMarketData(_candle(65000, 60500))).poll()
        self.assertEqual(r2.outcome, MonitorOutcome.NO_OPEN_SIGNAL)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT)


# ── H. Immutability ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestImmutability(MonitorTestCase):
    def _check(self, sig_before, sig_after):
        self.assertEqual(sig_after.entry, sig_before.entry)
        self.assertEqual(sig_after.stop_loss, sig_before.stop_loss)
        self.assertEqual(sig_after.take_profit, sig_before.take_profit)
        self.assertEqual(sig_after.direction, sig_before.direction)
        self.assertEqual(sig_after.timeframe, sig_before.timeframe)

    def test_entry_sl_tp_unchanged_after_tp(self):
        before = self._open_long()
        after = SignalMonitor(self.harness.repository, FakeMarketData(_candle(65000, 60500))).poll().signal
        self._check(before, after)

    def test_entry_sl_tp_unchanged_after_sl(self):
        before = self._open_long()
        after = SignalMonitor(self.harness.repository, FakeMarketData(_candle(60500, 59000))).poll().signal
        self._check(before, after)


# ── I. Data failures ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDataFailure(MonitorTestCase):
    def test_empty_data(self):
        self._open_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData()).poll()
        self.assertEqual(r.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_open_signal().status, "OPEN")

    def test_binance_connection_error(self):
        from binance.client import BinanceConnectionError

        self._open_long()
        md = FakeMarketData()
        md.fetch_klines = lambda *a, **k: (_ for _ in ()).throw(
            BinanceConnectionError("timeout")
        )
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_open_signal().status, "OPEN")

    def test_empty_kline_error(self):
        from binance.market_data import EmptyKlineError

        self._open_long()
        md = FakeMarketData()
        md.fetch_klines = lambda *a, **k: (_ for _ in ()).throw(EmptyKlineError("no data"))
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_open_signal().status, "OPEN")

    def test_malformed_candle_values(self):
        self._open_long()
        bad_candle = Candle(
            timestamp=1, open="x", high="x", low="x", close="x",
            volume=1.0, close_time=2, is_closed=True,
        )
        r = SignalMonitor(self.harness.repository, FakeMarketData(bad_candle)).poll()
        self.assertEqual(r.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        self.assertEqual(self.harness.repository.get_open_signal().status, "OPEN")

    def test_unexpected_exception_yields_error(self):
        self._open_long()
        md = FakeMarketData()
        md.fetch_klines = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.ERROR)
        self.assertEqual(self.harness.repository.get_open_signal().status, "OPEN")

    def test_do_not_close_on_missing_data(self):
        """DATA_UNAVAILABLE must never leave the signal partially closed."""
        self._open_long()
        SignalMonitor(self.harness.repository, FakeMarketData()).poll()
        SignalMonitor(self.harness.repository, FakeMarketData()).poll()
        sig = self.harness.repository.get_open_signal()
        self.assertIsNotNone(sig)
        self.assertEqual(sig.status, "OPEN")
        self.assertIsNone(sig.close_price)
        self.assertIsNone(sig.closed_at)


# ── J. Decimal precision ────────────────────────────────────────────────────


@pytest.mark.unit
class TestDecimalPrecision(MonitorTestCase):
    def test_decimal_comparison_used(self):
        # Verify Decimal(str(candle.high)) is used, not candle.high directly
        self._open_long()
        # Candle high at TP minus a tiny amount (Decimal-representable)
        high_float = 64000.0 - 0.0001  # 63999.9999
        candle = _candle(high_float, 60500)
        r = SignalMonitor(self.harness.repository, FakeMarketData(candle)).poll()
        # Should NOT be TP_HIT (63999.9999 < 64000)
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(Decimal(str(high_float)), Decimal("63999.9999"))
        self.assertLess(Decimal(str(high_float)), Decimal("64000.0"))

    def test_decimal_boundary_hit_exact(self):
        self._open_long(take_profit=64000.5)
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(64000.5, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(r.signal.close_price, Decimal("64000.5"))

    def test_high_precision_exact_hit(self):
        self._open_long(take_profit=64000.123456789)
        high = float(Decimal("64000.123456789"))
        self.assertEqual(Decimal(str(high)), Decimal("64000.123456789"))
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(high, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)

    def test_evaluate_candle_uses_decimal(self):
        sig = Signal(id=1, symbol="BTCUSDT", timeframe="1h", direction="LONG",
                     status="OPEN", entry=Decimal("61000"), stop_loss=Decimal("60000"),
                     take_profit=Decimal("64000"), created_at="2026-09-10T01:00:00Z")
        self.assertIs(evaluate_candle(sig, Decimal("64000.00000001"), Decimal("60000.00000001")),
                      MonitorOutcome.TP_HIT)
        self.assertIs(evaluate_candle(sig, Decimal("63999.99999999"), Decimal("60000.00000001")),
                      MonitorOutcome.MONITORING)


# ── K. No AI ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestNoAI(MonitorTestCase):
    def test_signal_analyzer_never_called_on_close(self):
        self._open_long()
        md = FakeMarketData(_candle(65000, 60500))
        with (
            patch("signal_engine.analysis.SignalAnalyzer") as sa,
            patch("signal_engine.analysis.analyze_signal") as af,
            patch("signal_engine.engine.SignalEngine.analyze_and_create") as ac,
            patch("signal_engine.engine.SignalEngine.process") as sp,
        ):
            r = SignalMonitor(self.harness.repository, md).poll()
        sa.assert_not_called()
        af.assert_not_called()
        ac.assert_not_called()
        sp.assert_not_called()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)


# ── L. No private Binance API ───────────────────────────────────────────────


@pytest.mark.unit
class TestNoPrivateBinance(MonitorTestCase):
    def test_no_order_endpoint_called(self):
        self._open_long()
        md = FakeMarketData(_candle(65000, 60500))
        with patch("binance.client.BinanceFuturesClient") as cls:
            r = SignalMonitor(self.harness.repository, md).poll()
        cls.assert_not_called()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(len(md.calls), 1)
        _, interval, _, _ = md.calls[0]
        self.assertEqual(interval, MONITOR_INTERVAL)

    def test_monitor_module_has_no_order_references(self):
        import signal_engine.monitor as mod

        with open(mod.__file__) as f:
            src = f.read()
        for needle in ("/fapi/v1/order", "/fapi/v1/balance"):
            self.assertNotIn(needle, src)


# ── Concurrency ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestConcurrency(MonitorTestCase):
    def test_two_concurrent_tp_polls_one_wins(self):
        sig = self._open_long()
        md = FakeMarketData(_candle(65000, 60500))
        barrier = threading.Barrier(2, timeout=10)

        def poll(_i):
            barrier.wait(timeout=10)
            return SignalMonitor(self.harness.repository, md).poll()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(poll, range(2)))
        outcomes = sorted(r.outcome for r in results)
        self.assertEqual(outcomes, [MonitorOutcome.NO_OPEN_SIGNAL, MonitorOutcome.TP_HIT])
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_TP_HIT)

    def test_two_concurrent_sl_polls_one_wins(self):
        sig = self._open_long()
        md = FakeMarketData(_candle(60500, 59000))
        barrier = threading.Barrier(2, timeout=10)

        def poll(_i):
            barrier.wait(timeout=10)
            return SignalMonitor(self.harness.repository, md).poll()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(poll, range(2)))
        outcomes = sorted(r.outcome for r in results)
        self.assertEqual(outcomes, [MonitorOutcome.NO_OPEN_SIGNAL, MonitorOutcome.SL_HIT])
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_SL_HIT)

    def test_concurrent_entry_promotion_single_open_position(self):
        sig = self._create_long()
        md = FakeMarketData(_candle(61100, 60700))
        barrier = threading.Barrier(2, timeout=10)

        def poll(_i):
            barrier.wait(timeout=10)
            return SignalMonitor(self.harness.repository, md).poll()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(poll, range(2)))
        # Exactly one thread wins the promotion (ENTRY_HIT). The losing thread
        # observes the already-promoted OPEN signal and either re-reports the
        # entry hit or evaluates the open position (MONITORING) - in no case
        # does it open a second position. The database guarantees a single row.
        outcomes = {r.outcome for r in results}
        self.assertIn(MonitorOutcome.ENTRY_HIT, outcomes)
        self.assertTrue(outcomes <= {MonitorOutcome.ENTRY_HIT, MonitorOutcome.MONITORING})
        self.assertEqual(len(self.harness.repository.list_signals()), 1)
        active = self.harness.repository.get_active_signal()
        self.assertEqual(active.status, STATUS_OPEN)
        self.assertEqual(active.id, sig.id)


# ── M. PENDING_ENTRY: entry trigger monitoring ───────────────────────────────


@pytest.mark.unit
class TestPendingEntry(MonitorTestCase):
    def _pending_long(self, **overrides):
        return self._create_long(**overrides)

    def _pending_short(self, **overrides):
        return self._create_short(**overrides)

    def test_long_entry_hit_promotes_to_open(self):
        sig = self._pending_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(61100, 60700))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.ENTRY_HIT)
        promoted = self.harness.repository.get_signal(sig.id)
        self.assertEqual(promoted.status, STATUS_OPEN)
        self.assertIsNotNone(promoted.opened_at)
        self.assertIsNone(promoted.closed_at)

    def test_long_entry_boundary_promotes(self):
        sig = self._pending_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(61000, 60700))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_long_entry_not_reached_stays_pending(self):
        sig = self._pending_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(60999.9999, 60700))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        after = self.harness.repository.get_signal(sig.id)
        self.assertEqual(after.status, STATUS_PENDING_ENTRY)
        self.assertIsNone(after.opened_at)

    def test_short_entry_hit_promotes_to_open(self):
        sig = self._pending_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(59500, 58999.9999))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)

    def test_short_entry_not_reached_stays_pending(self):
        sig = self._pending_short()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(59500, 59000.0001))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_PENDING_ENTRY)

    def test_entry_touched_then_open_signal_closes_at_tp(self):
        sig = self._pending_long()
        SignalMonitor(self.harness.repository, FakeMarketData(_candle(61100, 60700))).poll()
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_OPEN)
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(65000, 60500))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(r.signal.id, sig.id)
        self.assertEqual(r.signal.status, STATUS_TP_HIT)

    def test_forming_candle_never_confirms_entry(self):
        sig = self._pending_long()
        forming = _candle(68000, 55000, closed=False)
        closed = _candle(60900, 60700, closed=True)
        r = SignalMonitor(self.harness.repository, FakeMarketData(forming, closed)).poll()
        self.assertEqual(r.outcome, MonitorOutcome.MONITORING)
        self.assertEqual(self.harness.repository.get_signal(sig.id).status, STATUS_PENDING_ENTRY)

    def test_data_failure_leaves_pending_unchanged(self):
        from binance.client import BinanceConnectionError

        sig = self._pending_long()
        md = FakeMarketData()
        md.fetch_klines = lambda *a, **k: (_ for _ in ()).throw(
            BinanceConnectionError("timeout")
        )
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.DATA_UNAVAILABLE)
        after = self.harness.repository.get_signal(sig.id)
        self.assertEqual(after.status, STATUS_PENDING_ENTRY)
        self.assertIsNone(after.opened_at)

    def test_entry_and_sl_same_candle_ambiguous(self):
        sig = self._pending_long()
        r = SignalMonitor(self.harness.repository, FakeMarketData(_candle(68000, 55000))).poll()
        self.assertEqual(r.outcome, MonitorOutcome.AMBIGUOUS)
        after = self.harness.repository.get_signal(sig.id)
        self.assertEqual(after.status, STATUS_PENDING_ENTRY)
        self.assertIsNone(after.opened_at)
        self.assertIsNone(after.closed_at)
        self.assertIsNone(after.close_price)

    def test_no_ai_and_no_order_while_pending(self):
        sig = self._pending_long()
        md = FakeMarketData(_candle(61100, 60700))
        with (
            patch("signal_engine.analysis.SignalAnalyzer") as sa,
            patch("signal_engine.analysis.analyze_signal") as af,
            patch("signal_engine.engine.SignalEngine.analyze_and_create") as ac,
            patch("binance.client.BinanceFuturesClient") as client_cls,
        ):
            r = SignalMonitor(self.harness.repository, md).poll()
        sa.assert_not_called()
        af.assert_not_called()
        ac.assert_not_called()
        client_cls.assert_not_called()
        self.assertEqual(r.outcome, MonitorOutcome.ENTRY_HIT)
        self.assertEqual(sig.status, STATUS_PENDING_ENTRY)

    def test_evaluate_entry_pure_function(self):
        from database.models import Signal as SignalModel

        sig = SignalModel(id=1, symbol="BTCUSDT", timeframe="1h", direction="LONG",
                          status=STATUS_PENDING_ENTRY, entry=Decimal("61000"),
                          stop_loss=Decimal("60000"), take_profit=Decimal("64000"),
                          created_at="2026-09-10T01:00:00Z")
        self.assertIs(evaluate_entry(sig, Decimal("61000"), Decimal("60000.1")),
                      MonitorOutcome.ENTRY_HIT)
        self.assertIs(evaluate_entry(sig, Decimal("60999.9999"), Decimal("60001")),
                      MonitorOutcome.MONITORING)
        # Entry and SL within the same candle: ambiguous, no guess.
        self.assertIs(evaluate_entry(sig, Decimal("61000"), Decimal("59000")),
                      MonitorOutcome.AMBIGUOUS)

        short = SignalModel(id=2, symbol="BTCUSDT", timeframe="1h", direction="SHORT",
                            status=STATUS_PENDING_ENTRY, entry=Decimal("59000"),
                            stop_loss=Decimal("60000"), take_profit=Decimal("58000"),
                            created_at="2026-09-10T01:00:00Z")
        self.assertIs(evaluate_entry(short, Decimal("59500"), Decimal("59000")),
                      MonitorOutcome.ENTRY_HIT)
        self.assertIs(evaluate_entry(short, Decimal("59500"), Decimal("59500.5")),
                      MonitorOutcome.MONITORING)


@pytest.mark.unit
class TestNoActiveIdle(MonitorTestCase):
    def test_idle_no_active_no_fetch(self):
        md = FakeMarketData(_candle(65000, 30000))
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.NO_ACTIVE_SIGNAL)
        self.assertEqual(md.calls, [])

    def test_no_active_alias_matches_legacy_name(self):
        md = FakeMarketData()
        r = SignalMonitor(self.harness.repository, md).poll()
        self.assertEqual(r.outcome, MonitorOutcome.NO_OPEN_SIGNAL)
        self.assertIs(r.outcome, MonitorOutcome.NO_ACTIVE_SIGNAL)


# ── Monitor init ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestMonitorInit(unittest.TestCase):
    def test_invalid_candle_limit(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        for val in (0, -1, None, "5", 5.0):
            with self.subTest(val=val), self.assertRaises(ValueError):
                SignalMonitor(harness.repository, candle_limit=val)

    def test_valid_candle_limit(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        mon = SignalMonitor(harness.repository, candle_limit=1)
        self.assertEqual(mon.candle_limit, 1)

    def test_now_ms_forwarded(self):
        harness = TempSignalDb()
        self.addCleanup(harness.close)
        harness.engine.process(make_analysis("LONG"))

        received: list[int | None] = []

        class Probe:
            def fetch_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
                received.append(now_ms)
                return []

        SignalMonitor(harness.repository, Probe()).poll(now_ms=999_999)
        self.assertEqual(received, [999_999])


if __name__ == "__main__":
    unittest.main()
