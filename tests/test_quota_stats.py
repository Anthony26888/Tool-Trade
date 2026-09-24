"""Quota statistics tests (Phase Q): aggregation, ranges, endpoint, UI markers.

Covers ``CandleLogRepository.quota_summary`` (totals, metered/~estimated
split, Vietnam-day grouping, ``since`` filtering), ``GET /api/quota`` range
handling, and the Statistics-tab UI (markers + table/pager logic in node).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from database.database import CandleLogRepository, Database
from web.server import WebApplication, WebServer

MS = 3_600_000


def seed(repo: CandleLogRepository) -> None:
    rows = [
        # (ts, recorded_at, outcome, model, calls, in, out, total, estimated)
        (1, "2026-09-20T18:00:00.000Z", "WAIT", "openai/minimax-m3", 5, 1000, 200, 1200, False),
        (2, "2026-09-21T02:00:00.000Z", "WAIT", "openai/minimax-m3", 1, 500, 100, 600, True),
        (3, "2026-09-21T03:00:00.000Z", "WAIT", None, 0, None, None, None, False),
        (4, "2026-09-22T01:00:00.000Z", "CREATED", "openai/gpt-4o", 5, 2000, 400, 2400, False),
        (5, "2026-09-22T02:00:00.000Z", "ERROR", "openai/gpt-4o", 2, None, None, None, False),
        (6, "2026-09-22T03:00:00.000Z", "RATE_LIMITED", "openai/minimax-m3", 0, None, None, None, False),
        (7, "2026-09-22T04:00:00.000Z", "BLOCKED_ACTIVE_SIGNAL", None, 0, None, None, None, False),
    ]
    for ts, recorded, outcome, model, calls, prompt, completion, total, estimated in rows:
        repo.upsert(
            symbol="BTCUSDT", timeframe="1h", candle_timestamp_ms=ts,
            recorded_at=recorded, outcome=outcome, decision="WAIT" if outcome in ("WAIT", "CREATED") else "NONE",
            llm_calls=calls, prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=total, tokens_estimated=estimated,
            model=model, provider="openai" if model else None,
        )


@pytest.mark.unit
class TestQuotaSummary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "q.db"))
        self.db.initialize()
        self.repo = CandleLogRepository(self.db)
        seed(self.repo)

    def test_totals_and_split(self):
        summary = self.repo.quota_summary()
        self.assertEqual(summary["candles"], 7)
        self.assertEqual(summary["llm_calls"], 13)
        self.assertEqual(summary["prompt_tokens"], 3500)
        self.assertEqual(summary["completion_tokens"], 700)
        self.assertEqual(summary["total_tokens"], 4200)
        self.assertEqual(summary["prompt_tokens_metered"], 3000)
        self.assertEqual(summary["prompt_tokens_estimated"], 500)
        self.assertEqual(summary["total_tokens_metered"], 3600)
        self.assertEqual(summary["total_tokens_estimated"], 600)
        self.assertAlmostEqual(summary["success_rate"], 4 / 6)

    def test_by_model(self):
        summary = self.repo.quota_summary()
        by_model = {m["model"]: m for m in summary["by_model"]}
        mini = by_model["openai/minimax-m3"]
        self.assertEqual(mini["llm_calls"], 6)
        self.assertEqual(mini["total_tokens"], 1800)
        self.assertAlmostEqual(mini["success_rate"], 2 / 3)
        gpt = by_model["openai/gpt-4o"]
        self.assertEqual(gpt["llm_calls"], 7)
        self.assertAlmostEqual(gpt["success_rate"], 1 / 2)
        unknown = by_model["unknown"]
        self.assertEqual(unknown["llm_calls"], 0)
        self.assertAlmostEqual(unknown["success_rate"], 1.0)

    def test_vietnam_day_grouping(self):
        summary = self.repo.quota_summary()
        by_day = {d["day"]: d for d in summary["days"]}
        # 20/09 18:00Z belongs to VN 21/09.
        self.assertEqual(by_day["2026-09-21"]["candles"], 3)
        self.assertEqual(by_day["2026-09-21"]["llm_calls"], 6)
        self.assertEqual(by_day["2026-09-22"]["candles"], 4)
        self.assertEqual(by_day["2026-09-22"]["total_tokens"], 2400)
        days = [d["day"] for d in summary["days"]]
        self.assertEqual(days, sorted(days, reverse=True))

    def test_since_filter(self):
        summary = self.repo.quota_summary("2026-09-22T00:00:00Z")
        self.assertEqual(summary["candles"], 4)
        self.assertEqual(summary["llm_calls"], 7)

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "empty.db"))
            db.initialize()
            summary = CandleLogRepository(db).quota_summary()
        self.assertEqual(summary["candles"], 0)
        self.assertEqual(summary["days"], [])
        self.assertEqual(summary["total_tokens"], 0)


@pytest.mark.unit
class TestQuotaEndpoint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(os.path.join(self._tmp.name, "web.db"))
        self.db.initialize()
        seed(CandleLogRepository(self.db))
        self.app = WebApplication(self.db)
        self.server = WebServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.bound_port}"

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _post(self, path, body):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_all_range(self):
        status, payload = self._get("/api/quota?range=all")
        self.assertEqual(status, 200)
        self.assertEqual(payload["range"], "all")
        self.assertEqual(payload["llm_calls"], 13)
        self.assertEqual(len(payload["days"]), 2)
        self.assertEqual(payload["per_call_vnd"], 0)
        self.assertEqual(payload["cost_vnd"], 0)
        self.assertEqual(len(payload["by_model"]), 3)

    def test_cost_applied_from_settings(self):
        self.app.config_service.update_quota_cost({"per_call_vnd": 5})
        status, payload = self._get("/api/quota?range=all")
        self.assertEqual(status, 200)
        self.assertEqual(payload["per_call_vnd"], 5)
        self.assertEqual(payload["cost_vnd"], 13 * 5)
        day_costs = {d["day"]: d["cost_vnd"] for d in payload["days"]}
        self.assertEqual(day_costs["2026-09-21"], 6 * 5)
        model_costs = {m["model"]: m["cost_vnd"] for m in payload["by_model"]}
        self.assertEqual(model_costs["openai/minimax-m3"], 6 * 5)

    def test_quota_cost_settings_validation(self):
        status, payload = self._get("/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(payload["settings"]["quota_cost"]["per_call_vnd"], 0)
        _, updated = self._post("/api/settings", {"quota_cost": {"per_call_vnd": 5}})
        self.assertEqual(updated["settings"]["quota_cost"]["per_call_vnd"], 5)
        with self.assertRaises(urllib.error.HTTPError):
            self._post("/api/settings", {"quota_cost": {"per_call_vnd": -1}})
        with self.assertRaises(urllib.error.HTTPError):
            self._post("/api/settings", {"quota_cost": {"nope": 1}})

    def test_bad_range_falls_back_to_all(self):
        status, payload = self._get("/api/quota?range=someday")
        self.assertEqual(status, 200)
        self.assertEqual(payload["range"], "all")
        self.assertEqual(payload["candles"], 7)

    def test_today_range_matches_server_clock(self):
        from datetime import timedelta

        status, payload = self._get("/api/quota?range=today")
        self.assertEqual(status, 200)
        self.assertEqual(payload["range"], "today")
        now_vn = datetime.now(timezone.utc) + timedelta(hours=7)
        # Seeded rows are Sep 2026; today (real clock) holds none.
        if now_vn.date().strftime("%Y-%m-%d") not in ("2026-09-21", "2026-09-22"):
            self.assertEqual(payload["candles"], 0)
            self.assertEqual(payload["days"], [])


@pytest.mark.unit
class TestQuotaUi(unittest.TestCase):
    def test_markers(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(os.path.join(tmp.name, "web.db"))
        db.initialize()
        app = WebApplication(db)
        server = WebServer(app, host="127.0.0.1", port=0)
        server.start()
        self.addCleanup(server.stop)
        base = f"http://127.0.0.1:{server.bound_port}"

        def raw(path):
            with urllib.request.urlopen(base + path, timeout=10) as resp:
                return resp.read().decode("utf-8")

        html = raw("/")
        for marker in ("quotaCards", "quotaTable", "quotaPager", "quotaRange"):
            self.assertIn(f'id="{marker}"', html)
        js = raw("/static/app.js")
        for marker in (
            "loadQuota",
            "renderQuotaTable",
            "renderQuotaPager",
            "renderQuotaModels",
            "drawQuotaChart",
            "quotaChart",
            "quotaTip",
            "quotaSavePrice",
            "quotaPrice",
            "fmtCompact",
            "fmtVND",
            "fmtPct1",
            "quota.thDay",
            "quota.mCalls",
            "quota.mEst",
            "quota.mSuccess",
            "quota.mCost",
            "quota.thModel",
            "quota.thCost",
            "quota.perCall",
            "quota.byModel",
            "quota.empty",
        ):
            self.assertIn(marker, js)

    def test_quota_table_logic(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        with open(
            os.path.join(os.path.dirname(__file__), "..", "web", "static", "app.js"),
            encoding="utf-8",
        ) as handle:
            js = handle.read()

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            "const QUOTA_PAGE_SIZE = 15;\n"
            "let quotaPage = 0;\n"
            "let quotaDays = [];\n"
            "const TABLE = { hidden: false, innerHTML: '' };\n"
            "const PAGER = { hidden: false, innerHTML: '' };\n"
            "const $ = (sel) => sel === '#quotaTable' ? TABLE : PAGER;\n"
            "const t = (k) => k;\n"
            "const emptyBox = (x) => 'EMPTY:' + x;\n"
            "const fmtCompact = (v) => 'C' + v;\n"
            "const fmtVnDay = (d) => 'D' + d;\n"
            "const fmtVND = (v) => 'V' + v;\n"
            "let quotaPrice = 0;\n"
            + _extract_fn("pagerHtml") + "\n" + _extract_fn("renderQuotaTable") + "\n" + _extract_fn("renderQuotaPager") + "\n"
            "quotaDays = []; renderQuotaTable();\n"
            "const emptyHtml = TABLE.innerHTML;\n"
            "quotaDays = Array.from({length: 20}, (_, i) => ({day: '2026-09-' + (i + 1), llm_calls: i, prompt_tokens: i, completion_tokens: i, total_tokens: i}));\n"
            "quotaPage = 0; renderQuotaTable();\n"
            "const page0 = TABLE.innerHTML;\n"
            "quotaPage = 1; renderQuotaTable();\n"
            "const page1 = TABLE.innerHTML;\n"
            "console.log(JSON.stringify([emptyHtml, page0.includes('D2026-09-1') && !page0.includes('D2026-09-16'), page1.includes('D2026-09-16')]));"
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
        empty_html, page0_ok, page1_ok = json.loads(
            proc.stdout.strip().splitlines()[-1]
        )
        self.assertIn("EMPTY:", empty_html)
        self.assertTrue(page0_ok)
        self.assertTrue(page1_ok)

    def test_pure_helpers(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        with open(
            os.path.join(os.path.dirname(__file__), "..", "web", "static", "app.js"),
            encoding="utf-8",
        ) as handle:
            js = handle.read()

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            _extract_fn("fmtCompact") + "\n"
            "const fmtNum = (v, d) => Number(v).toFixed(d);\n"
            + _extract_fn("fmtVND")
            + "\n" + _extract_fn("fmtPct1")
            + "\n" + _extract_fn("niceStep")
            + "\nlet quotaDays = [];\n"
            + _extract_fn("quotaChartRows")
            + "\nconst out = [];\n"
            "out.push([fmtVND(260), fmtVND(1500), fmtVND(null), fmtPct1(0.995), fmtPct1(null), niceStep(1825), niceStep(3)]);\n"
            "quotaDays = Array.from({length: 35}, (_, i) => ({day: '2026-09-' + (i + 1), llm_calls: 1, prompt_tokens: 10, completion_tokens: 2, total_tokens: 12, cost_vnd: 5}));\n"
            "const batched = quotaChartRows();\n"
            "out.push([batched.weekly, batched.rows.length, batched.rows[0].total_tokens, batched.rows[0].cost_vnd]);\n"
            "quotaDays = [{day: '2026-09-01', llm_calls: 2, prompt_tokens: 20, completion_tokens: 4, total_tokens: 24, cost_vnd: 10}];\n"
            "const single = quotaChartRows();\n"
            "out.push([single.weekly, single.rows.length]);\n"
            "console.log(JSON.stringify(out));"
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
        fmt, batched, single = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(
            fmt, ["260₫", "1.5k₫", "—", "99.5%", "—", 2000, 5]
        )
        self.assertEqual(batched, [True, 5, 84, 35])
        self.assertEqual(single, [False, 1])

    def test_hover_wiring_and_tooltip(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        with open(
            os.path.join(os.path.dirname(__file__), "..", "web", "static", "app.js"),
            encoding="utf-8",
        ) as handle:
            js = handle.read()
        # The four listeners must exist (their absence was the reported bug).
        for marker in (
            '"mousemove", (ev) => quotaHover(ev.clientX, ev.clientY)',
            '"mouseleave", () => quotaTooltip(null)',
            '"touchstart"',
            '"touchmove"',
            "quotaHover(touch.clientX, touch.clientY)",
        ):
            self.assertIn(marker, js)

        def _extract_fn(name):
            start = js.index(f"function {name}(")
            depth = 0
            for pos in range(start, len(js)):
                if js[pos] == "{":
                    depth += 1
                elif js[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return js[start : pos + 1]
            raise AssertionError(f"unbalanced braces in {name}")

        program = (
            "let quotaLayout = [];\n"
            "let quotaPrice = 5;\n"
            "const TIP = { hidden: true, innerHTML: '', style: {} };\n"
            "const CANVAS = { parentElement: { clientWidth: 600 }, "
            "getBoundingClientRect: () => ({ left: 0, top: 0 }) };\n"
            "const $ = (sel) => sel === '#quotaTip' ? TIP : CANVAS;\n"
            "const t = (k) => k;\n"
            "const escapeHtml = (s) => s;\n"
            "const fmtCompact = (v) => 'C' + v;\n"
            "const fmtVnDay = (d) => 'D' + d;\n"
            "const fmtVND = (v) => 'V' + v;\n"
            + _extract_fn("quotaTooltip")
            + "\n" + _extract_fn("quotaHover")
            + "\nquotaLayout = [{ x0: 0, x1: 100, day: { day: '2026-09-23', llm_calls: 52, prompt_tokens: 27800, completion_tokens: 4400, total_tokens: 32200, cost_vnd: 260 } }];\n"
            "quotaHover(50, 30);\n"
            "const shown = [TIP.hidden, TIP.innerHTML.includes('D2026-09-23'), TIP.innerHTML.includes('C52'), TIP.innerHTML.includes('V260')];\n"
            "quotaTooltip(null);\n"
            "const hiddenAfter = TIP.hidden;\n"
            "quotaLayout = [];\n"
            "TIP.hidden = false;\n"
            "quotaHover(50, 30);\n"
            "const stillHidden = TIP.hidden;\n"
            "console.log(JSON.stringify([shown, hiddenAfter, stillHidden]));"
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
        shown, hidden_after, still_hidden = json.loads(
            proc.stdout.strip().splitlines()[-1]
        )
        self.assertEqual(shown, [False, True, True, True])
        self.assertTrue(hidden_after)
        self.assertTrue(still_hidden)


if __name__ == "__main__":
    unittest.main()
