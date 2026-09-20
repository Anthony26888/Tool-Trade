"""Chart view-logic tests (zoom/pan anchoring + pads + default window).

The canvas view helpers in ``web/static/app.js`` are pure w.r.t. a tiny
surface (``chartView``/``lastChart``/``redrawChart``), so this test extracts
their exact source from the shipped file (balanced-brace parser, no copies)
and executes behavioral assertions with node. A source drift that renames or
removes a helper fails loudly instead of testing a stale copy.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parent.parent / "web" / "static" / "app.js"
HELPERS = ("clampView", "zoomChart", "panChart", "resetChart")


def _extract(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    depth = 0
    for pos in range(start, len(source)):
        char = source[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _const(source: str, name: str) -> str:
    match = re.search(rf"const {name} = ([^;]+);", source)
    assert match, f"const {name} not found"
    return match.group(1).strip()


NODE_ASSERTIONS = """
const results = [];
const check = (name, cond) => results.push([name, !!cond]);
// fresh state, 500 candles
chartView = null; lastChart = { candles: new Array(500).fill(0) };
let v = clampView(500);
check("default window is 50", v.count === 50 && v.backFromNewest === 0);
// zoom in at newest stays glued to newest
zoomChart(1 / 1.25);
check("zoom-in keeps back 0", chartView.backFromNewest === 0);
check("zoom-in shrinks", chartView.count === 40);
// zoom out grows, still glued
zoomChart(1.25);
check("zoom-out keeps back 0", chartView.backFromNewest === 0);
check("zoom-out grows", chartView.count === 50);
// pan back then zoom: position preserved (the reported bug)
panChart(1); panChart(1);
const backBefore = clampView(500).backFromNewest;
check("pan moved back", backBefore > 0);
const countBefore = clampView(500).count;
zoomChart(1 / 1.25);
check("zoom preserves pan", chartView.backFromNewest === backBefore);
check("zoom shrinks in past", chartView.count < countBefore);
// zoom out clamps back into range
chartView = { count: 100, backFromNewest: 400 };
zoomChart(4);
check("zoom-out clamps back", chartView.backFromNewest === 500 - chartView.count);
// pan clamps at both edges
chartView = { count: 100, backFromNewest: 0 };
panChart(-1);
check("pan-right clamps at 0", clampView(500).backFromNewest === 0);
chartView = { count: 100, backFromNewest: 400 };
panChart(1);
check("pan-left clamps at max", clampView(500).backFromNewest === 400);
// min window respected
chartView = { count: 16, backFromNewest: 0 };
zoomChart(1 / 1.25);
check("min window 15", clampView(500).count === 15);
// reset restores default
resetChart();
check("reset restores default", clampView(500).count === 50 && clampView(500).backFromNewest === 0);
console.log(JSON.stringify(results));
"""


@pytest.mark.unit
class TestChartViewLogic(unittest.TestCase):
    def test_view_helpers_behave(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        source = APP_JS.read_text(encoding="utf-8")
        preamble = "\n".join(
            [
                "let chartView = null;",
                "let lastChart = null;",
                f"const CHART_WINDOW_DEFAULT = {_const(source, 'CHART_WINDOW_DEFAULT')};",
                f"const CHART_WINDOW_MIN = {_const(source, 'CHART_WINDOW_MIN')};",
                "const redrawChart = () => {};",
            ]
        )
        program = (
            preamble + "\n" + "\n".join(_extract(source, h) for h in HELPERS) + "\n" + NODE_ASSERTIONS
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=60
            )
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        results = json.loads(proc.stdout.strip().splitlines()[-1])
        failed = [name for name, ok in results if not ok]
        self.assertEqual(failed, [], msg=f"view assertions failed: {failed}")

    def test_default_window_is_50(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertEqual(_const(source, "CHART_WINDOW_DEFAULT"), "50")

    def test_drag_pad_matches_draw_pad(self):
        source = APP_JS.read_text(encoding="utf-8")
        drag = re.search(r"const chartPad = \{([^}]+)\}", source)
        self.assertIsNotNone(drag, msg="chartPad not found")
        draw = re.search(
            r"const pad = \{ top: 8, right: (\d+), bottom: 18, left: (\d+) \};", source
        )
        self.assertIsNotNone(draw, msg="draw pad not found")
        drag_left = re.search(r"left:\s*(\d+)", drag.group(1)).group(1)
        drag_right = re.search(r"right:\s*(\d+)", drag.group(1)).group(1)
        self.assertEqual((drag_left, drag_right), (draw.group(2), draw.group(1)))

    def test_zoom_does_not_reset_pan(self):
        source = APP_JS.read_text(encoding="utf-8")
        body = _extract(source, "zoomChart")
        self.assertNotIn("backFromNewest: 0", body)
        self.assertIn("cur.backFromNewest", body)


if __name__ == "__main__":
    unittest.main()
