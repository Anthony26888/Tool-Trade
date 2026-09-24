"""Theme + i18n tests (web/static only, no backend).

- Every ``t("...")`` key used in app.js and every ``data-i18n`` key in
  index.html exists in BOTH language dicts (evaluated with node, so the real
  shipped source is checked, never a copy).
- Theme toggle + language toggle markers and the light-theme variable block.
- No leftover Vietnamese-only hardcode in dynamic render paths: the
  Vietnamese strings that remain must live inside the ``vi`` dict.
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

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
STYLE_CSS = STATIC / "style.css"


def _extract_i18n(source: str) -> str:
    start = source.index("const I18N = {")
    depth = 0
    for pos in range(start, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1].replace("const I18N = ", "module.exports = ")
    raise AssertionError("unbalanced braces in I18N")


def _load_dicts() -> dict:
    source = APP_JS.read_text(encoding="utf-8")
    program = _extract_i18n(source) + "\nconsole.log(JSON.stringify({en: Object.keys(module.exports.en), vi: Object.keys(module.exports.vi)}));"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(program)
        path = handle.name
    try:
        proc = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.unit
class TestI18nParity(unittest.TestCase):
    def test_keys_match_both_languages(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        keys = _load_dicts()
        en, vi = set(keys["en"]), set(keys["vi"])
        self.assertEqual(en - vi, set(), msg=f"missing in vi: {sorted(en - vi)}")
        self.assertEqual(vi - en, set(), msg=f"missing in en: {sorted(vi - en)}")
        self.assertGreater(len(en), 100)

    def test_every_used_key_exists(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        keys = _load_dicts()
        available = set(keys["en"])
        source = APP_JS.read_text(encoding="utf-8")
        used = set(re.findall(r'\bt\("([^"]+)"', source))
        used |= set(re.findall(r"\bt\('([^']+)'", source))
        # Dynamic prefixes (title./subtitle./daemon.) resolve per panel/group name.
        missing = {k for k in used if k not in available and k not in {"title.", "subtitle.", "daemon."}}
        self.assertEqual(missing, set(), msg=f"undefined i18n keys: {sorted(missing)}")
        group_suffixes = set(re.findall(r'\["(g[A-Z][A-Za-z]*)",', source))
        group_missing = {f"daemon.{s}" for s in group_suffixes} - available
        self.assertEqual(group_missing, set(), msg=f"undefined group keys: {sorted(group_missing)}")
        html = INDEX_HTML.read_text(encoding="utf-8")
        static_keys = set(re.findall(r'data-i18n="([^"]+)"', html))
        static_missing = {k for k in static_keys if k not in available}
        self.assertEqual(static_missing, set(), msg=f"undefined static keys: {sorted(static_missing)}")

    def test_no_vietnamese_outside_vi_dict(self):
        source = APP_JS.read_text(encoding="utf-8")
        # The vi dict ends where the I18N literal ends.
        i18n_src = _extract_i18n(source)
        vi_block = i18n_src[i18n_src.index("vi: {") :]
        stripped = source.replace(vi_block, "")
        vietnamese = re.findall(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]", stripped)
        self.assertEqual(vietnamese, [], msg="Vietnamese text outside the vi dict")


@pytest.mark.unit
class TestThemeMarkers(unittest.TestCase):
    def test_toggle_and_variables(self):
        html = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('id="themeBtn"', html)
        self.assertIn('id="langBtn"', html)
        self.assertIn("nextra.theme", html)
        self.assertIn("nextra.lang", html)
        css = STYLE_CSS.read_text(encoding="utf-8")
        self.assertIn('[data-theme="light"]', css)
        self.assertIn("color-scheme: dark", css)
        self.assertIn("color-scheme: light", css)
        for selector in (
            ".toolbar-item select",
            ".chart-interval",
            ".chart-btn",
            ".chat-chip",
            "#chatInput",
        ):
            self.assertIn(f'[data-theme="light"] {selector}', css)
        self.assertRegex(css, r"\.btn\.danger\s*\{[^}]*color:\s*#fff")
        self.assertRegex(css, r"\.btn\.toolbar-item\s*\{[^}]*color:\s*#fff")
        js = APP_JS.read_text(encoding="utf-8")
        for marker in ("applyTheme", "applyStaticI18n", "applyI18n", "MONTHS", "function t("):
            self.assertIn(marker, js)


@pytest.mark.unit
class TestI18nRenderSmoke(unittest.TestCase):
    """Execute the shipped app.js under a stub DOM and render key sections
    in both languages: headers/labels must translate, and no ``undefined``
    or raw ``t("...")`` key may leak into the HTML."""

    PREAMBLE = """
globalThis.localStorage = { _s: {}, getItem(k) { return this._s[k] ?? null; }, setItem(k, v) { this._s[k] = String(v); } };
function __mkEl() {
  const el = {
    children: [], dataset: {}, style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {}, appendChild(c) { return c; },
    setAttribute() {}, removeAttribute() {},
    querySelector() { return __mkEl(); },
    querySelectorAll() { return []; },
    getContext() { return new Proxy({}, { get: () => () => {} }); },
    focus() {}, click() {},
  };
  return new Proxy(el, {
    get(o, k) {
      if (k === "then") return undefined;
      const v = o[k];
      if (typeof v === "function") return v.bind(o);
      return v !== undefined ? v : "";
    },
    set(o, k, v) { o[k] = v; return true; },
  });
}
globalThis.document = {
  querySelector() { return __mkEl(); },
  querySelectorAll() { return []; },
  createElement() { return __mkEl(); },
  documentElement: __mkEl(),
  addEventListener() {},
};
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.devicePixelRatio = 1;
globalThis.fetch = async () => ({ ok: true, json: async () => ({}) });
"""

    DRIVER = """
lang = "vi";
const ROW = {id:1,direction:"LONG",entry:2,stop_loss:1,take_profit:3,confidence:80,status:"OPEN",result:null,model_name:"m",created_at:"2026-01-01T00:00:00Z"};
const sigVi = renderSignalTable([ROW]);
const DD = {id:1,recorded_at:"2026-01-01T00:00:00.000Z",candle_timestamp_ms:1,decision:"WAIT",confidence:50,outcome:"WAIT",signal_id:null,model:null,provider:"p",close_price:1,llm_calls:0,prompt_tokens:null,completion_tokens:null,total_tokens:null,tokens_estimated:null,entry:null,stop_loss:null,take_profit:null,temperature:null,reasoning:"ly do",error_notes:null,indicators:{rsi14:36.7}};
const ddVi = renderDaemonTable([DD]);
eventsMonth = "2026-10";
eventsDays = {"2026-10-28":[{name:"FOMC decision",title:"FOMC T",impact:"high",trading_paused:true,event_ms:1,event_vn:"v",event_utc:"u",blackout_start_vn:"a",blackout_end_vn:"b",url:""}]};
renderEventsCalendar();
lang = "en";
const sigEn = renderSignalTable([ROW]);
const ddEn = renderDaemonTable([DD]);
console.log(JSON.stringify({sigVi, ddVi, sigEn, ddEn}));
process.exit(0);
"""

    def test_render_both_languages(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        source = APP_JS.read_text(encoding="utf-8")
        program = self.PREAMBLE + "\n" + source + "\n" + self.DRIVER
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(program)
            path = handle.name
        try:
            proc = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=120
            )
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-3000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        for name, html in out.items():
            self.assertNotIn("undefined", html, msg=name)
            self.assertNotIn('t("', html, msg=name)
        self.assertIn("Hướng", out["sigVi"])
        self.assertNotIn("Direction", out["sigVi"])
        self.assertIn("Direction", out["sigEn"])
        self.assertIn("Nhận định AI", out["ddVi"])
        self.assertIn("Quyết định", out["ddVi"])


if __name__ == "__main__":
    unittest.main()
