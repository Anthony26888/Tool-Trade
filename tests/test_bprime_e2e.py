"""Plan B' Phase B4 tests: multi-symbol end-to-end on one shared DB.

Drives the real wiring (scheduler -> engine -> monitor -> executor ->
dashboard portfolio) for two symbols sharing one account: BTC holds while
ETH keeps analyzing (the user's core scenario), TP closes free the slot,
and the portfolio line reports SEEKING/HOLDING/FULL. No real network, no
real daemons; the supervisor script is covered in test_supervisor.py.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone

import pytest

from binance.market_data import Candle
from demo.executor import DemoExecutor
from signal_engine import (
    MonitorOutcome,
    OneHourScheduler,
    SchedulerOutcome,
)
from tests.signal_engine_test_helpers import (
    TempSignalDb,
    make_analysis,
    make_candles,
)

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
INTERVAL_MS = 3_600_000


def make_analyzer(decision="WAIT", calls=None):
    def analyzer(candles, indicators):
        calls.append(True)
        return make_analysis(decision)

    return analyzer


class SymbolFeed:
    """Market-data double serving per-symbol 1m scripts to the monitor."""

    def __init__(self, scripts):
        self.scripts = dict(scripts)
        self.seen = []

    def fetch_closed_klines(
        self, symbol, interval, limit, *, end_time_ms=None, now_ms=None
    ):
        raise AssertionError("scheduler must use its own feed in this test")

    def fetch_klines(self, symbol, interval, limit, *, end_time_ms=None, now_ms=None):
        self.seen.append(symbol)
        return list(self.scripts.get(symbol, []))


def klines_1m(*pairs):
    out = []
    for i, (high, low) in enumerate(pairs):
        out.append(
            Candle(
                timestamp=1_000_000 + i * 60_000,
                open=(high + low) / 2.0,
                high=high,
                low=low,
                close=(high + low) / 2.0,
                volume=1.0,
                close_time=1_000_000 + i * 60_000 + 59_999,
                is_closed=True,
            )
        )
    return out


@pytest.mark.unit
class TestMultiSymbolEndToEnd(unittest.TestCase):
    def setUp(self):
        self.harness = TempSignalDb()
        self.addCleanup(self.harness.close)
        self.repo = self.harness.repository
        self.state = self.harness.state
        self.engine = self.harness.engine
        self.executor = DemoExecutor(self.harness.db)
        self.base_ms = int(NOW.timestamp() * 1000) - 220 * INTERVAL_MS

    def _scheduler(self, symbol, analyzer, count=220):
        from tests.test_signal_engine_scheduler import FakeSchedulerData

        # count=220 ends 11:00, count=221 ends 12:00: each analyzed tick
        # needs a candle newer than that symbol's processed marker.
        candles = make_candles(count, start_ms=self.base_ms)
        return OneHourScheduler(
            self.repo,
            market_data=FakeSchedulerData(*candles),
            engine=self.engine,
            state=self.state,
            analyzer=analyzer,
            symbol=symbol,
        )

    def _create_open(self, symbol, direction="LONG"):
        analysis = make_analysis(
            direction,
            symbol=symbol,
            entry_price=61000.0,
            stop_loss=60000.0,
            take_profit=64000.0,
        )
        result = self.engine.process(analysis)
        assert result.outcome.value == "CREATED", result
        signal = result.signal
        self.repo.transition_signal(signal.id, "OPEN")
        position = self.executor.open_position(self.repo.get_signal(signal.id))
        assert position is not None
        return self.repo.get_signal(signal.id)

    def test_btc_holds_while_eth_keeps_analyzing(self):
        btc = self._create_open("BTCUSDT")
        btc_calls: list = []
        btc_sched = self._scheduler("BTCUSDT", make_analyzer("WAIT", btc_calls))
        self.assertEqual(
            btc_sched.tick(now=NOW).outcome, SchedulerOutcome.BLOCKED_ACTIVE_SIGNAL
        )
        self.assertEqual(btc_calls, [])
        eth_calls: list = []
        eth_sched = self._scheduler("ETHUSDT", make_analyzer("WAIT", eth_calls))
        self.assertEqual(eth_sched.tick(now=NOW).outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(eth_calls), 1)
        # BTC monitor watches BTC only; a TP touch closes BTC, ETH untouched.
        feed = SymbolFeed({"BTCUSDT": klines_1m((64100.0, 63900.0))})
        from signal_engine import SignalMonitor

        btc_mon = SignalMonitor(self.repo, market_data=feed, symbol="BTCUSDT")
        result = btc_mon.poll()
        self.assertEqual(result.outcome, MonitorOutcome.TP_HIT)
        self.assertEqual(feed.seen, ["BTCUSDT"])
        trade = self.executor.close_position(self.repo.get_signal(btc.id))
        self.assertIsNotNone(trade)
        self.assertGreater(float(trade.net_pnl), 0)
        # Freed BTC slot analyzes again next tick.
        btc_calls2: list = []
        btc_sched2 = self._scheduler(
            "BTCUSDT", make_analyzer("WAIT", btc_calls2), count=221
        )
        later = datetime(2026, 9, 10, 13, 0, tzinfo=timezone.utc)
        self.assertEqual(btc_sched2.tick(now=later).outcome, SchedulerOutcome.WAIT)
        self.assertEqual(len(btc_calls2), 1)

    def test_portfolio_line_states(self):
        from signal_engine.config import ConfigService
        from web.server import WebApplication

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        service = ConfigService(
            self.harness.db,
            secrets_path=os.path.join(tmp.name, "secrets.json"),
            env={},
        )
        service.update_symbols({"symbols": ["BTCUSDT", "ETHUSDT"]})
        app = WebApplication(self.harness.db, config_service=service)

        def states():
            portfolio = app.dashboard()["portfolio"]
            return (
                {e["symbol"]: e["state"] for e in portfolio["symbols"]},
                portfolio["full"],
            )

        # Both seeking.
        self.assertEqual(
            states(), ({"BTCUSDT": "SEEKING", "ETHUSDT": "SEEKING"}, False)
        )
        # BTC holds, ETH seeks.
        self._create_open("BTCUSDT")
        by_symbol, full = states()
        self.assertEqual(by_symbol["BTCUSDT"], "OPEN")
        self.assertEqual(by_symbol["ETHUSDT"], "SEEKING")
        self.assertFalse(full)
        # FULL when both hold.
        self._create_open("ETHUSDT")
        by_symbol, full = states()
        self.assertTrue(full)
        self.assertEqual(
            by_symbol, {"BTCUSDT": "OPEN", "ETHUSDT": "OPEN"}
        )

    def test_portfolio_keeps_deselected_holding_symbol(self):
        from signal_engine.config import ConfigService
        from web.server import WebApplication

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        service = ConfigService(
            self.harness.db,
            secrets_path=os.path.join(tmp.name, "secrets.json"),
            env={},
        )
        # Combo tracks BTC only, but ETH holds a blocked OPEN stop.
        service.update_symbols({"symbols": ["BTCUSDT"]})
        self._create_open("ETHUSDT")
        app = WebApplication(self.harness.db, config_service=service)
        portfolio = app.dashboard()["portfolio"]
        by_symbol = {e["symbol"]: e["state"] for e in portfolio["symbols"]}
        self.assertEqual(by_symbol["ETHUSDT"], "OPEN")
        self.assertEqual(by_symbol["BTCUSDT"], "SEEKING")
        self.assertFalse(portfolio["full"])


if __name__ == "__main__":
    unittest.main()
