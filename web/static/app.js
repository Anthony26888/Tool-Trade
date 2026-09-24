"use strict";

const $ = (sel) => document.querySelector(sel);

// -- helpers -----------------------------------------------------------------

function toast(message, kind) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast show" + (kind ? " " + kind : "");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => (el.className = "toast"), 3500);
}

async function api(path, options) {
  const resp = await fetch(path, options);
  let data = {};
  try { data = await resp.json(); } catch (_) { /* empty */ }
  if (!resp.ok) {
    throw new Error(data && data.error ? data.error : "HTTP " + resp.status);
  }
  return data;
}

const fmtNum = (v, digits) => {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: digits || 2, maximumFractionDigits: digits || 2 });
};

const pct = (v) => {
  if (v === null || v === undefined || v === "") return "—";
  return fmtNum(Number(v) * 100, 1) + "%";
};

// Fixed Vietnam time (UTC+7, no DST) rendering for ISO-8601 UTC timestamps.
const fmtVn = (iso) => {
  if (!iso) return "—";
  const s = String(iso).includes("Z") || String(iso).includes("+") ? String(iso) : String(iso) + "Z";
  const ms = Date.parse(s);
  if (!Number.isFinite(ms)) return String(iso);
  const e = new Date(ms + 7 * 3600 * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(e.getUTCDate())}/${p(e.getUTCMonth() + 1)}/${e.getUTCFullYear()} ${p(e.getUTCHours())}:${p(e.getUTCMinutes())}:${p(e.getUTCSeconds())} (+07)`;
};

// Compact ``DD/MM HH:MM`` label for a UTC epoch-millisecond candle timestamp in Vietnam time.
const fmtVnMs = (ms) => {
  if (!Number.isFinite(Number(ms))) return "—";
  const e = new Date(Number(ms) + 7 * 3600 * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(e.getUTCDate())}/${p(e.getUTCMonth() + 1)} ${p(e.getUTCHours())}:${p(e.getUTCMinutes())}`;
};

function badge(text) {
  const cls = String(text).toLowerCase().replace("_", "-");
  return `<span class="badge ${cls}">${text}</span>`;
}

function emptyBox(text) {
  return `<p class="muted">${text}</p>`;
}

// -- i18n + theme ---------------------------------------------------------------

let lang = "en";
let theme = "dark";

const I18N = {
en: {
"title.dashboard": "Dashboard",
"subtitle.dashboard": "Signal & demo trading overview",
"title.signals": "Signals",
"subtitle.signals": "AI signal ledger",
"title.daemon": "Daemon Log",
"subtitle.daemon": "Per-1H-candle AI activity",
"title.events": "Events",
"subtitle.events": "Macro calendar — trading pauses inside ⏸ windows",
"title.trades": "Trades",
"subtitle.trades": "Demo trade ledger",
"title.statistics": "Statistics",
"subtitle.statistics": "Performance metrics",
"title.settings": "Settings",
"subtitle.settings": "AI provider, demo account & notifications",
"nav.dashboard": "Dashboard",
"nav.signals": "Signals",
"nav.daemon": "Daemon Log",
"nav.daemonShort": "Daemon",
"nav.events": "Events",
"nav.trades": "Trades",
"nav.statistics": "Statistics",
"nav.statsShort": "Stats",
"nav.settings": "Settings",
"side.symbol": "Symbol",
"side.mode": "Mode",
"common.all": "All",
"common.today": "Today",
"common.time": "Time (VN)",
"common.loading": "Loading…",
"chart.reset": "Reset",
"chart.updated": "updated {time}",
"chart.unavailable": "Chart data unavailable — {msg}",
"chart.noData": "No candle data yet.",
"chart.legend": "Chart updates every 15s",
"chart.rangeAll": "all · {n} candles",
"chart.range": "{a} → {b} · {n} candles",
"dash.signal": "Current Signal",
"dash.health": "System Health",
"dash.noSignal": "No signal has been created yet. The daemon analyzes {symbol} on each closed 1H candle.",
"dash.healthState": "state",
"dash.healthScheduler": "scheduler",
"dash.healthMonitor": "monitor",
"dash.healthLastTick": "last tick",
"dash.healthLastPoll": "last poll",
"dash.healthLastError": "last error",
"dash.kDirection": "Direction",
"dash.kEntry": "Entry",
"dash.kTP": "Take Profit",
"dash.kSL": "Stop Loss",
"dash.kConf": "Confidence",
"dash.kModel": "Model",
"dash.kAnalysis": "Analysis",
"dash.kSide": "Side",
"dash.kQty": "Quantity",
"dash.kMarginSize": "Margin / Size",
"dash.kLeverage": "Leverage",
"dash.kOpenPnl": "Open PnL",
"dash.kTpOutcome": "TP outcome",
"dash.kSlOutcome": "SL outcome",
"dash.bal": "bal.",
"dash.mBalance": "Balance",
"dash.mEquity": "Equity",
"dash.mNetPnl": "Net PnL",
"dash.mWinRate": "Win Rate",
"dash.mPeak": "Peak Equity",
"dash.mDD": "Max Drawdown",
"dash.statusActive": "SIGNAL ACTIVE — {status}",
"dash.statusMulti": "{n} ACTIVE",
"dash.statusIdle": "IDLE — READY",
"dash.priceTitle": "{symbol} Price",
"health.running": "RUNNING",
"health.stopped": "STOPPED",
"sig.title": "Signal History",
"sig.direction": "Direction",
"sig.status": "Status",
"sig.selectAll": "Select all",
"sig.deselectAll": "Deselect all",
"sig.deleteSel": "Delete selected ({n})",
"sig.deleteAll": "Delete all history",
"sig.empty": "No signals recorded yet.",
"sig.confirmDelSel": "Delete {n} selected signal(s)?\nLinked demo positions/trades will also be deleted.",
"sig.confirmDelAll": "Delete ALL signal history (except any OPEN position)?\nLinked demo positions/trades will also be deleted. This cannot be undone.",
"sig.deleted": "Deleted {n} signal(s){kept}",
"sig.keptOpen": ", kept {n} OPEN",
"sig.thId": "ID",
"sig.thSymbol": "Symbol",
"sig.thDir": "Direction",
"sig.thEntry": "Entry",
"sig.thSL": "Stop Loss",
"sig.thTP": "Take Profit",
"sig.thConf": "Conf",
"sig.thStatus": "Status",
"sig.thResult": "Result",
"sig.thModel": "Model",
"sig.thTime": "Time",
"sig.thProvider": "Provider",
"sig.thAnalysis": "Analysis",
"daemon.title": "Daemon Activity per 1H Candle",
"daemon.decision": "Decision",
"daemon.clear": "Clear log",
"daemon.hint": "Click a row for reasoning, indicator snapshot & prices.",
"daemon.empty": "No daemon activity recorded yet. The daemon writes one row per closed 1H candle once it runs.",
"daemon.confirmClear": "Delete the entire daemon activity log? This only removes diagnostic rows, not signals/positions.",
"daemon.cleared": "Cleared {n} log row(s)",
"daemon.selectAll": "Select all",
"daemon.deselectAll": "Deselect all",
"daemon.deleteSel": "Delete selected ({n})",
"daemon.deleted": "Deleted {n} log row(s)",
"daemon.confirmDelSel": "Delete {n} selected log row(s)? Signals/positions are never affected.",
"daemon.thRecorded": "Recorded",
"daemon.thCandle": "Candle close (VN)",
"daemon.thSymbol": "Symbol",
"daemon.thDecision": "Decision",
"daemon.thConf": "Conf",
"daemon.thOutcome": "Outcome",
"daemon.thSignal": "Signal",
"daemon.thModel": "Model",
"daemon.thClose": "Close",
"daemon.thCalls": "Calls",
"daemon.thNotes": "Error / Notes",
"pager.prev": "‹ Prev",
"pager.next": "Next ›",
"pager.label": "Page {n} · rows {from}+",
"daemon.secOrder": "Order",
"daemon.secQuota": "AI & Quota",
"daemon.secIndicators": "Indicators",
"daemon.secReasoning": "AI Reasoning",
"daemon.secError": "Error",
"daemon.noOrder": "— no order created",
"daemon.kModel": "Model",
"daemon.kTemp": "Temp",
"daemon.kCalls": "LLM calls",
"daemon.kTokens": "Tokens in/out/total",
"daemon.kTokensEst": "Tokens in/out/total (~)",
"daemon.tokensEstTitle": "Estimated (~4 chars/token), not a metered count",
"daemon.tokensRealTitle": "Metered count from the provider",
"daemon.gTrend": "Trend",
"daemon.gMomentum": "Momentum",
"daemon.gVol": "Volatility",
"daemon.gStrength": "Trend strength",
"daemon.gVolume": "Volume",
"trades.title": "Demo Trades",
"trades.empty": "No demo trades closed yet.",
"trades.thId": "ID",
"trades.thSymbol": "Symbol",
"trades.thSide": "Side",
"trades.thEntry": "Entry",
"trades.thExit": "Exit",
"trades.thQty": "Qty",
"trades.thLev": "Lev",
"trades.thGross": "Gross",
"trades.thFee": "Fee",
"trades.thNet": "Net PnL",
"trades.thResult": "Result",
"trades.thOpened": "Opened",
"trades.thClosed": "Closed",
"events.title": "Macro Events — new positions pause inside ⏸ windows",
"events.today": "Today",
"events.footer": "Tap a day for details. Times in Vietnam (+07). Calendar data:",
"events.sourceLive": "Source: live",
"events.sourceFallback": "Source: fallback (FOMC only — check network)",
"events.sourceFallbackTitle": "financecalendar.com unreachable, using the static calendar file",
"events.sourceLiveTitle": "Live calendar from financecalendar.com",
"events.emptyDay": "No macro events this day.",
"events.dayTitle": "{d}/{m}/{y} — {n}",
"events.dayTitleEmpty": "{d}/{m}/{y} — no events",
"events.kTime": "VN Time",
"events.allDay": "All day",
"events.kUtc": "UTC",
"events.kImpact": "Impact",
"events.kConsensus": "Consensus",
"events.kPrior": "Prior",
"events.kActual": "Actual",
"events.kSignal": "⛔ Signal",
"events.kSource": "Source",
"events.aiPause": "AI pauses {from} → {to}",
"events.bannerActive": "⏸ BLACKOUT: {title} — AI pauses until {until}",
"events.bannerUpcoming": "⚠️ {title} {when} — new entries stop from {from}",
"events.bannerSoonH": "in ~{n}h",
"events.bannerSoonM": "in {n}m",
"events.monthOf": "{month} · {year}",
"stats.detail": "Performance Detail",
"stats.mTrades": "Trades",
"stats.mWL": "Wins / Losses",
"stats.mFees": "Total Fees",
"quota.title": "LLM Quota Usage",
"quota.mCalls": "LLM calls",
"quota.mIn": "Tokens in",
"quota.mOut": "Tokens out",
"quota.mTotal": "Total tokens",
"quota.mEst": "Estimated (~)",
"quota.mRequests": "Requests",
"quota.mSuccess": "Success",
"quota.mCost": "Cost",
"quota.costNote": "using {price}/request (estimate, not the provider invoice)",
"quota.costUnset": "set a per-request price above to estimate cost",
"quota.perCall": "Cost per request (VND)",
"quota.save": "Save",
"quota.saved": "Quota price saved.",
"quota.saveFail": "Quota price save failed: {msg}",
"quota.byModel": "Usage by model",
"quota.byDay": "Usage by day",
"quota.thModel": "Model",
"quota.thSuccess": "Success",
"quota.thCost": "Cost",
"quota.weekly": "weekly buckets",
"quota.thDay": "Day (VN)",
"quota.thCalls": "Calls",
"quota.thIn": "In",
"quota.thOut": "Out",
"quota.thTotal": "Total",
"quota.empty": "No quota recorded in this range yet.",
"set.ai": "AI Provider",
"set.demo": "Demo Account",
"set.telegram": "Telegram Notifications",
"set.strategy": "Strategy Tuning",
"set.runtime": "Runtime & System",
"set.symbol": "Active Symbol (analysed)",
"set.symbolDisplay": "Display Symbol (dashboard default)",
"set.symbolHint": "Display default for the chart and single-symbol runs. Which symbols actually run is decided by Trading Symbols below — each runs its own daemon.",
"set.saveSymbol": "Save Symbol",
"set.clearError": "Clear last error",
"set.symbolsTitle": "Trading Symbols (parallel)",
"set.symbolsHint": "Tick 1–3 symbols, Save, then run ./run-symbols.sh sync. Each symbol runs its own daemon on the shared account. At least one must stay ticked.",
"set.saveSymbols": "Save Symbols",
"set.audit": "Settings Audit",
"chat.empty": "Start a conversation about the current signal.",
"chat.q1": "Why this pick?",
"chat.q1q": "Why was this signal chosen?",
"chat.q2": "SL risk?",
"chat.q2q": "What is the risk if Stop Loss is hit?",
"chat.q3": "Strategy summary",
"chat.q3q": "Summarize the strategy of this signal.",
"chat.placeholder": "Ask about this signal…",
"chat.send": "Send",
"chat.clear": "Clear",
"set.pendingBadge": "CHANGE WILL APPLY WHEN IDLE",
"set.appliedBadge": "APPLIED",
"set.ollama": "Ollama (local)",
"set.apiProv": "API (OpenAI-compatible)",
"set.preset": "API Preset",
"set.baseUrl": "Base URL",
"set.model": "Model",
"set.analysisMode": "Analysis Mode",
"set.modeSingle": "Single (1 AI call)",
"set.modeMulti": "Multi-agent debate (5 AI calls)",
"set.multiHint": "\"Multi\" runs the analyst > bull > bear > trader > risk-manager committee (5 LLM calls per closed 1H candle, same provider/model). Applies to the NEXT analysis once no signal is active.",
"set.temperature": "Temperature (0–5)",
"set.maxTokens": "Max Tokens",
"set.timeout": "Timeout (s)",
"set.apiKey": "API Key (API preset)",
"set.apiKeyOptional": "optional",
"set.aiHint": "AI provider changes apply to the NEXT analysis.",
"set.aiPending": "A change is waiting — it will apply automatically when no signal is active.",
"set.saveAi": "Save AI Provider",
"set.testConn": "Test Connection",
"set.connTesting": "Testing…",
"set.connReachable": "reachable",
"set.connModelMissing": "model NOT found on server",
"set.connModels": "Models: ",
"set.connFailed": "FAILED",
"set.demoLocked": "Demo settings are LOCKED while a signal is active (AI Lock + one-active-signal rule).",
"set.dBalance": "Initial Balance (USDT)",
"set.dMargin": "Margin per Trade (USDT)",
"set.dLeverage": "Leverage (x)",
"set.dRisk": "Risk %",
"set.dFee": "Fee Rate (fraction, e.g. 0.0004 = 0.04%)",
"set.demoHint": "Changes apply to FUTURE trades only — the ledger and current position are never touched. To apply a new Initial Balance, Save it, then click Reset Demo Account.",
"set.saveDemo": "Save Demo Settings",
"set.resetDemo": "Reset Demo Account",
"set.tgEnabled": "Enabled",
"set.tgChatSignals": "Chat ID — Signals (trade lifecycle)",
"set.tgChatReports": "Chat ID — Reports (reports + notices)",
"set.tgToken": "Bot Token",
"set.tgHint": "One bot, two groups: add the bot to both groups, then paste the chat ids. Leave one channel empty to merge into the other. Tokens and chat ids are stored encrypted-side (0600 secrets file); the UI always masks them.",
"set.saveTg": "Save Telegram",
"set.testSignals": "Test Signals",
"set.testReports": "Test Reports",
"set.configured": "configured (masked)",
"set.stConfidence": "Min confidence (0–100)",
"set.stConfidenceHint": "AI below threshold → WAIT",
"set.stRR": "Min risk/reward",
"set.stRRHint": "How many times TP must exceed SL",
"set.stFunding": "Crowded funding (fraction/8h)",
"set.stFundingHint": "E.g. 0.0005 = 0.05% — above threshold blocks crowded entries",
"set.stFee": "Fee rate (fraction)",
"set.stFeeHint": "E.g. 0.0004 = 0.04% — TP must cover fees ×2",
"set.stExpiry": "Pending expiry (hours)",
"set.stExpiryHint": "Stale PENDING auto-cancels, 0 = wait forever",
"set.stWarn": "Pre-news warning (hours)",
"set.stWarnHint": "Telegram heads-up before FOMC/CPI",
"set.stHtf": "4H bias (block counter-trend entries)",
"set.stHint": "Save applies from the next 1H candle — no restart, OPEN positions unaffected. Reset falls back to .env.",
"set.saveStrategy": "Save Strategy",
"set.resetStrategy": "Reset to defaults",
"set.symState": "state",
"set.symScheduler": "scheduler",
"set.symMonitor": "monitor",
"set.symLastTick": "last tick",
"set.symLastPoll": "last poll",
"set.symLastError": "last error",
"set.rtSymbol": "Symbol",
"set.rtTimeframe": "Timeframe (analysis)",
"set.rtSchedPoll": "Scheduler poll",
"set.rtMonPoll": "Monitor poll",
"set.rtDb": "Database",
"set.auditEmpty": "No audit entries yet.",
"set.auditThTime": "Time",
"set.auditThNs": "Namespace",
"set.auditThAction": "Action",
"set.auditThSummary": "Summary",
"chat.title": "Signal Chat · NEXTRA AI",
"chat.about": "Ask about this signal",
"chat.model": "Model",
"chat.analysis": "Analysis",
"chat.noSignal": "no active signal — chat will explain once one exists",
"chat.openTitle": "🤖 AI",
"toast.aiSaved": "AI provider saved — applies to next analysis.",
"toast.aiSaveFail": "AI provider save failed: {msg}",
"toast.demoSaved": "Demo settings saved for future trades.",
"toast.demoSaveFail": "Demo save failed: {msg}",
"toast.demoReset": "Demo account reset to a fresh start.",
"toast.demoResetFail": "Demo reset failed: {msg}",
"toast.tgSaved": "Telegram settings saved.",
"toast.tgSaveFail": "Telegram save failed: {msg}",
"toast.tgTestOk": "Test to {channel} sent ✓",
"toast.tgTestFail": "Test failed: {msg}",
"toast.strategySaved": "Strategy saved — applies from the next candle.",
"toast.strategySaveFail": "Strategy save failed: {msg}",
"toast.strategyReset": "Strategy reset to defaults.",
"toast.strategyResetFail": "Strategy reset failed: {msg}",
"toast.symbolSaved": "Active symbol set to {symbol}.",
"toast.symbolsSaved": "Symbols saved — run ./run-symbols.sh sync to apply.",
"toast.symbolsSaveFail": "Symbols save failed: {msg}",
"toast.symbolSaveFail": "Symbol save failed: {msg}",
"toast.errorCleared": "Last error cleared.",
"toast.clearFail": "Clear failed: {msg}",
"confirm.demoReset": "Reset the demo account to a fresh start?\n\nBalance / Equity / Peak Equity are set to the saved Initial Balance,\nand ALL demo positions and trades are cleared.\nSignal history is kept.",
"confirm.strategyReset": "Clear all saved Strategy settings and fall back to .env/defaults?",
"confirm.clearError": "Clear the stored \"last error\" shown in System Health?",
"pos.unavailable": "Positioning unavailable",
"pos.waiting": "Waiting entry",
"pos.close": "Close",
"pos.closeConfirm": "Close {dir} {symbol} now at market?\n\nEst. PnL ≈ {pnl}",
"pos.closed": "Position closed: {status} {pnl}",
"pos.closeFail": "Close failed: {msg}",
"pos.manual": "Manual",
"port.seeking": "seeking",
"port.full": "FULL — analysis idle, monitors watching",
"pos.lsUnavailable": "Long/Short unavailable",
"pos.long": "Long {n}%",
"pos.short": "Short {n}%",
"wd.mon": "Mon",
"wd.tue": "Tue",
"wd.wed": "Wed",
"wd.thu": "Thu",
"wd.fri": "Fri",
"wd.sat": "Sat",
"wd.sun": "Sun",
"month.1": "January",
"month.2": "February",
"month.3": "March",
"month.4": "April",
"month.5": "May",
"month.6": "June",
"month.7": "July",
"month.8": "August",
"month.9": "September",
"month.10": "October",
"month.11": "November",
"month.12": "December",
},
vi: {
"title.dashboard": "Bảng điều khiển",
"subtitle.dashboard": "Tổng quan signal & demo trading",
"title.signals": "Signals",
"subtitle.signals": "Sổ lệnh AI",
"title.daemon": "Nhật ký Daemon",
"subtitle.daemon": "Hoạt động AI theo từng nến 1H",
"title.events": "Sự kiện",
"subtitle.events": "Lịch vĩ mô — tạm dừng mở lệnh trong cửa sổ ⏸",
"title.trades": "Lệnh đã đóng",
"subtitle.trades": "Sổ lệnh demo",
"title.statistics": "Thống kê",
"subtitle.statistics": "Chỉ số hiệu suất",
"title.settings": "Cài đặt",
"subtitle.settings": "AI, tài khoản demo & thông báo",
"nav.dashboard": "Tổng quan",
"nav.signals": "Signals",
"nav.daemon": "Nhật ký",
"nav.daemonShort": "Nhật ký",
"nav.events": "Sự kiện",
"nav.trades": "Lệnh",
"nav.statistics": "Thống kê",
"nav.statsShort": "TKê",
"nav.settings": "Cài đặt",
"side.symbol": "Symbol",
"side.mode": "Chế độ",
"common.all": "Tất cả",
"common.today": "Hôm nay",
"common.time": "Giờ (VN)",
"common.loading": "Đang tải…",
"chart.reset": "Đặt lại",
"chart.updated": "cập nhật {time}",
"chart.unavailable": "Không tải được chart — {msg}",
"chart.noData": "Chưa có dữ liệu nến.",
"chart.legend": "Chart cập nhật mỗi 15s",
"chart.rangeAll": "tất cả · {n} nến",
"chart.range": "{a} → {b} · {n} nến",
"dash.signal": "Signal hiện tại",
"dash.health": "Trạng thái hệ thống",
"dash.noSignal": "Chưa có signal nào. Daemon phân tích {symbol} mỗi nến 1H đóng.",
"dash.healthState": "trạng thái",
"dash.healthScheduler": "scheduler",
"dash.healthMonitor": "monitor",
"dash.healthLastTick": "tick cuối",
"dash.healthLastPoll": "poll cuối",
"dash.healthLastError": "lỗi cuối",
"dash.kDirection": "Hướng",
"dash.kEntry": "Entry",
"dash.kTP": "Chốt lời",
"dash.kSL": "Cắt lỗ",
"dash.kConf": "Tự tin",
"dash.kModel": "Model",
"dash.kAnalysis": "Phân tích",
"dash.kSide": "Phe",
"dash.kQty": "Khối lượng",
"dash.kMarginSize": "Margin / Size",
"dash.kLeverage": "Đòn bẩy",
"dash.kOpenPnl": "PnL mở",
"dash.kTpOutcome": "Kịch bản TP",
"dash.kSlOutcome": "Kịch bản SL",
"dash.bal": "số dư",
"dash.mBalance": "Số dư",
"dash.mEquity": "Equity",
"dash.mNetPnl": "PnL ròng",
"dash.mWinRate": "Tỉ lệ thắng",
"dash.mPeak": "Equity đỉnh",
"dash.mDD": "Sụt giảm tối đa",
"dash.statusActive": "CÓ SIGNAL — {status}",
"dash.statusMulti": "{n} lệnh active",
"dash.statusIdle": "RẢNH — SẴN SÀNG",
"dash.priceTitle": "Giá {symbol}",
"health.running": "ĐANG CHẠY",
"health.stopped": "ĐÃ DỪNG",
"sig.title": "Lịch sử Signal",
"sig.direction": "Hướng",
"sig.status": "Trạng thái",
"sig.selectAll": "Chọn hết",
"sig.deselectAll": "Bỏ chọn hết",
"sig.deleteSel": "Xóa đã chọn ({n})",
"sig.deleteAll": "Xóa toàn bộ lịch sử",
"sig.empty": "Chưa ghi nhận signal nào.",
"sig.confirmDelSel": "Xóa {n} signal đã chọn?\nVị thế và lệnh demo liên quan cũng bị xóa.",
"sig.confirmDelAll": "Xóa TOÀN BỘ lịch sử signal (trừ lệnh OPEN đang chạy)?\nVị thế và lệnh demo liên quan cũng bị xóa. Không thể hoàn tác.",
"sig.deleted": "Đã xóa {n} signal(s){kept}",
"sig.keptOpen": ", giữ {n} OPEN",
"sig.thId": "ID",
"sig.thSymbol": "Symbol",
"sig.thDir": "Hướng",
"sig.thEntry": "Entry",
"sig.thSL": "Cắt lỗ",
"sig.thTP": "Chốt lời",
"sig.thConf": "Tự tin",
"sig.thStatus": "Trạng thái",
"sig.thResult": "Kết quả",
"sig.thModel": "Model",
"sig.thTime": "Thời gian",
"sig.thProvider": "Provider",
"sig.thAnalysis": "Phân tích",
"daemon.title": "Hoạt động Daemon theo nến 1H",
"daemon.decision": "Quyết định",
"daemon.clear": "Xóa log",
"daemon.hint": "Bấm vào 1 dòng để xem lập luận, chỉ báo & giá.",
"daemon.empty": "Chưa có hoạt động daemon. Daemon ghi 1 dòng mỗi nến 1H đóng khi chạy.",
"daemon.confirmClear": "Xóa toàn bộ nhật ký daemon? Chỉ xóa dòng chẩn đoán, không xóa signals/vị thế.",
"daemon.cleared": "Đã xóa {n} dòng log",
"daemon.selectAll": "Chọn hết",
"daemon.deselectAll": "Bỏ chọn hết",
"daemon.deleteSel": "Xóa đã chọn ({n})",
"daemon.deleted": "Đã xóa {n} dòng log",
"daemon.confirmDelSel": "Xóa {n} dòng log đã chọn? Signals/vị thế không bao giờ bị ảnh hưởng.",
"daemon.thRecorded": "Đã ghi",
"daemon.thCandle": "Nến đóng (VN)",
"daemon.thSymbol": "Symbol",
"daemon.thDecision": "Quyết định",
"daemon.thConf": "Tự tin",
"daemon.thOutcome": "Kết quả",
"daemon.thSignal": "Signal",
"daemon.thModel": "Model",
"daemon.thClose": "Đóng",
"daemon.thCalls": "Calls",
"daemon.thNotes": "Lỗi / Ghi chú",
"pager.prev": "‹ Trước",
"pager.next": "Sau ›",
"pager.label": "Trang {n} · dòng {from}+",
"daemon.secOrder": "Lệnh",
"daemon.secQuota": "AI & Quota",
"daemon.secIndicators": "Chỉ báo",
"daemon.secReasoning": "Nhận định AI",
"daemon.secError": "Lỗi",
"daemon.noOrder": "— không tạo lệnh",
"daemon.kModel": "Model",
"daemon.kTemp": "Temp",
"daemon.kCalls": "LLM calls",
"daemon.kTokens": "Tokens vào/ra/tổng",
"daemon.kTokensEst": "Tokens vào/ra/tổng (~)",
"daemon.tokensEstTitle": "Ước tính (~4 ký tự/token), không phải số đo thật",
"daemon.tokensRealTitle": "Số đo thật từ provider",
"daemon.gTrend": "Xu hướng",
"daemon.gMomentum": "Động lượng",
"daemon.gVol": "Biến động",
"daemon.gStrength": "Sức trend",
"daemon.gVolume": "Volume",
"trades.title": "Lệnh Demo đã đóng",
"trades.empty": "Chưa có lệnh demo nào đóng.",
"trades.thId": "ID",
"trades.thSymbol": "Symbol",
"trades.thSide": "Phe",
"trades.thEntry": "Entry",
"trades.thExit": "Thoát",
"trades.thQty": "SL",
"trades.thLev": "Lev",
"trades.thGross": "Gộp",
"trades.thFee": "Phí",
"trades.thNet": "PnL ròng",
"trades.thResult": "Kết quả",
"trades.thOpened": "Mở lúc",
"trades.thClosed": "Đóng lúc",
"events.title": "Sự kiện vĩ mô — tạm dừng mở lệnh trong cửa sổ ⏸",
"events.today": "Hôm nay",
"events.footer": "Chạm vào ngày để xem chi tiết. Giờ Việt Nam (+07). Dữ liệu lịch:",
"events.sourceLive": "Nguồn: trực tiếp",
"events.sourceFallback": "Nguồn: dự phòng (chỉ FOMC — kiểm tra mạng)",
"events.sourceFallbackTitle": "Không với tới financecalendar.com, đang dùng file lịch tĩnh",
"events.sourceLiveTitle": "Lịch trực tiếp từ financecalendar.com",
"events.emptyDay": "Ngày này không có sự kiện vĩ mô.",
"events.dayTitle": "{d}/{m}/{y} — {n} sự kiện",
"events.dayTitleEmpty": "{d}/{m}/{y} — không có sự kiện",
"events.kTime": "Giờ VN",
"events.allDay": "Cả ngày",
"events.kUtc": "UTC",
"events.kImpact": "Mức độ",
"events.kConsensus": "Đồng thuận",
"events.kPrior": "Kỳ trước",
"events.kActual": "Thực tế",
"events.kSignal": "⛔ Signal",
"events.kSource": "Nguồn",
"events.aiPause": "AI nghỉ {from} → {to}",
"events.bannerActive": "⏸ BLACKOUT: {title} — AI nghỉ đến {until}",
"events.bannerUpcoming": "⚠️ {title} {when} — ngừng mở lệnh từ {from}",
"events.bannerSoonH": "sau ~{n}h",
"events.bannerSoonM": "sau {n}p",
"events.monthOf": "Tháng {month} · {year}",
"stats.detail": "Chi tiết hiệu suất",
"stats.mTrades": "Số lệnh",
"stats.mWL": "Thắng / Thua",
"stats.mFees": "Tổng phí",
"quota.title": "Quota LLM đã dùng",
"quota.mCalls": "LLM calls",
"quota.mIn": "Tokens vào",
"quota.mOut": "Tokens ra",
"quota.mTotal": "Tổng tokens",
"quota.mEst": "Ước tính (~)",
"quota.mRequests": "Requests",
"quota.mSuccess": "Thành công",
"quota.mCost": "Chi phí",
"quota.costNote": "đang dùng giá {price}/request (ước tính, không phải hóa đơn provider)",
"quota.costUnset": "nhập giá mỗi request ở trên để ước tính chi phí",
"quota.perCall": "Chi phí mỗi request (VND)",
"quota.save": "Lưu",
"quota.saved": "Đã lưu giá quota.",
"quota.saveFail": "Lưu giá quota thất bại: {msg}",
"quota.byModel": "Dùng theo model",
"quota.byDay": "Dùng theo ngày",
"quota.thModel": "Model",
"quota.thSuccess": "Thành công",
"quota.thCost": "Chi phí",
"quota.weekly": "gộp theo tuần",
"quota.thDay": "Ngày (VN)",
"quota.thCalls": "Calls",
"quota.thIn": "Vào",
"quota.thOut": "Ra",
"quota.thTotal": "Tổng",
"quota.empty": "Khoảng này chưa ghi nhận quota.",
"set.ai": "AI Provider",
"set.demo": "Tài khoản Demo",
"set.telegram": "Thông báo Telegram",
"set.strategy": "Tinh chỉnh chiến lược",
"set.runtime": "Runtime & Hệ thống",
"set.symbol": "Symbol đang phân tích",
"set.symbolDisplay": "Symbol hiển thị (mặc định dashboard)",
"set.symbolHint": "Mặc định hiển thị cho chart và chạy đơn-symbol. Symbols nào thực sự chạy do Trading Symbols bên dưới quyết định — mỗi symbol 1 daemon riêng.",
"set.saveSymbol": "Lưu Symbol",
"set.clearError": "Xóa lỗi cuối",
"set.symbolsTitle": "Symbols giao dịch (song song)",
"set.symbolsHint": "Tick 1–3 symbols, Save, rồi chạy ./run-symbols.sh sync. Mỗi symbol chạy daemon riêng trên tài khoản chung. Phải giữ ít nhất 1 cái.",
"set.saveSymbols": "Lưu Symbols",
"set.audit": "Nhật ký cài đặt",
"chat.empty": "Bắt đầu trò chuyện về signal hiện tại.",
"chat.q1": "Vì sao chọn?",
"chat.q1q": "Vì sao chọn signal này?",
"chat.q2": "Rủi ro SL?",
"chat.q2q": "Rủi ro nếu Stop Loss bị chạm là như thế nào?",
"chat.q3": "Tóm tắt chiến lược",
"chat.q3q": "Tóm tắt chiến lược của signal này.",
"chat.placeholder": "Hỏi về signal này…",
"chat.send": "Gửi",
"chat.clear": "Xóa",
"set.pendingBadge": "ĐỔI SẼ ÁP DỤNG KHI RẢNH",
"set.appliedBadge": "ĐÃ ÁP DỤNG",
"set.ollama": "Ollama (local)",
"set.apiProv": "API (OpenAI-compatible)",
"set.preset": "API Preset",
"set.baseUrl": "Base URL",
"set.model": "Model",
"set.analysisMode": "Chế độ phân tích",
"set.modeSingle": "Single (1 AI call)",
"set.modeMulti": "Multi-agent debate (5 AI calls)",
"set.multiHint": "\"Multi\" chạy ủy ban analyst > bull > bear > trader > risk-manager (5 LLM calls mỗi nến 1H đóng, cùng provider/model). Áp dụng từ lần phân tích TIẾP theo khi không có signal active.",
"set.temperature": "Temperature (0–5)",
"set.maxTokens": "Max Tokens",
"set.timeout": "Timeout (s)",
"set.apiKey": "API Key (API preset)",
"set.apiKeyOptional": "không bắt buộc",
"set.aiHint": "Đổi AI provider áp dụng từ lần phân tích TIẾP theo.",
"set.aiPending": "Đang có thay đổi chờ — sẽ tự áp dụng khi không có signal active.",
"set.saveAi": "Lưu AI Provider",
"set.testConn": "Test kết nối",
"set.connTesting": "Đang test…",
"set.connReachable": "tới được",
"set.connModelMissing": "KHÔNG thấy model trên server",
"set.connModels": "Models: ",
"set.connFailed": "THẤT BẠI",
"set.demoLocked": "Cài đặt demo bị KHÓA khi có signal active (AI Lock + luật một lệnh).",
"set.dBalance": "Số dư ban đầu (USDT)",
"set.dMargin": "Margin mỗi lệnh (USDT)",
"set.dLeverage": "Đòn bẩy (x)",
"set.dRisk": "Rủi ro %",
"set.dFee": "Phí (fraction, VD 0.0004 = 0.04%)",
"set.demoHint": "Chỉ áp dụng cho lệnh SAU — sổ cái và vị thế hiện tại không bao giờ bị động. Muốn áp dụng Initial Balance mới: Save rồi bấm Reset Demo Account.",
"set.saveDemo": "Lưu cài đặt Demo",
"set.resetDemo": "Reset tài khoản Demo",
"set.tgEnabled": "Bật",
"set.tgChatSignals": "Chat ID — Signals (vòng đời lệnh)",
"set.tgChatReports": "Chat ID — Reports (báo cáo + thông báo)",
"set.tgToken": "Bot Token",
"set.tgHint": "Một bot, hai group: add bot vào cả 2 group rồi dán chat id. Bỏ trống một kênh = tin dồn về kênh còn lại. Token và chat id lưu file 0600; UI luôn che.",
"set.saveTg": "Lưu Telegram",
"set.testSignals": "Test Signals",
"set.testReports": "Test Reports",
"set.configured": "đã cấu hình (che)",
"set.stConfidence": "Confidence tối thiểu (0–100)",
"set.stConfidenceHint": "AI dưới ngưỡng → WAIT",
"set.stRR": "Risk/Reward tối thiểu",
"set.stRRHint": "TP dài hơn SL bao nhiêu lần",
"set.stFunding": "Funding crowded (fraction/8h)",
"set.stFundingHint": "VD 0.0005 = 0.05% — quá ngưỡng cấm đu đám đông",
"set.stFee": "Fee rate (fraction)",
"set.stFeeHint": "VD 0.0004 = 0.04% — TP phải cover phí ×2",
"set.stExpiry": "Hết hạn lệnh chờ (giờ)",
"set.stExpiryHint": "PENDING quá hạn tự hủy, 0 = chờ mãi",
"set.stWarn": "Báo trước tin (giờ)",
"set.stWarnHint": "Warn Telegram trước FOMC/CPI bao lâu",
"set.stHtf": "Bias 4H (cấm đánh ngược sóng 4H)",
"set.stHint": "Lưu là áp dụng từ nến 1H tiếp theo — không restart, lệnh OPEN không bị ảnh hưởng. Xóa Settings là về lại .env.",
"set.saveStrategy": "Lưu chiến lược",
"set.resetStrategy": "Reset về mặc định",
"set.symState": "trạng thái",
"set.symScheduler": "scheduler",
"set.symMonitor": "monitor",
"set.symLastTick": "tick cuối",
"set.symLastPoll": "poll cuối",
"set.symLastError": "lỗi cuối",
"set.rtSymbol": "Symbol",
"set.rtTimeframe": "Timeframe (phân tích)",
"set.rtSchedPoll": "Scheduler poll",
"set.rtMonPoll": "Monitor poll",
"set.rtDb": "Database",
"set.auditEmpty": "Chưa có bản ghi audit.",
"set.auditThTime": "Thời gian",
"set.auditThNs": "Nhóm",
"set.auditThAction": "Hành động",
"set.auditThSummary": "Tóm tắt",
"chat.title": "Signal Chat · NEXTRA AI",
"chat.about": "Hỏi về signal này",
"chat.model": "Model",
"chat.analysis": "Phân tích",
"chat.noSignal": "chưa có signal active — có signal chat sẽ giải thích",
"chat.openTitle": "🤖 AI",
"toast.aiSaved": "Đã lưu AI provider — áp dụng từ lần phân tích sau.",
"toast.aiSaveFail": "Lưu AI provider thất bại: {msg}",
"toast.demoSaved": "Đã lưu cài đặt demo cho lệnh sau.",
"toast.demoSaveFail": "Lưu demo thất bại: {msg}",
"toast.demoReset": "Đã reset tài khoản demo.",
"toast.demoResetFail": "Reset demo thất bại: {msg}",
"toast.tgSaved": "Đã lưu cài đặt Telegram.",
"toast.tgSaveFail": "Lưu Telegram thất bại: {msg}",
"toast.tgTestOk": "Đã gửi test tới {channel} ✓",
"toast.tgTestFail": "Test thất bại: {msg}",
"toast.strategySaved": "Đã lưu chiến lược — áp dụng từ nến sau.",
"toast.strategySaveFail": "Lưu chiến lược thất bại: {msg}",
"toast.strategyReset": "Đã reset chiến lược về mặc định.",
"toast.strategyResetFail": "Reset chiến lược thất bại: {msg}",
"toast.symbolSaved": "Đã đặt symbol phân tích: {symbol}.",
"toast.symbolsSaved": "Đã lưu symbols — chạy ./run-symbols.sh sync để áp dụng.",
"toast.symbolsSaveFail": "Lưu symbols thất bại: {msg}",
"toast.symbolSaveFail": "Lưu symbol thất bại: {msg}",
"toast.errorCleared": "Đã xóa lỗi cuối.",
"toast.clearFail": "Xóa thất bại: {msg}",
"confirm.demoReset": "Reset tài khoản demo về ban đầu?\n\nSố dư / Equity / Peak Equity về lại Initial Balance đã lưu,\nvà TOÀN BỘ vị thế + lệnh demo bị xóa.\nLịch sử signal được giữ.",
"confirm.strategyReset": "Xóa toàn bộ Strategy đã lưu và về lại .env/mặc định?",
"confirm.clearError": "Xóa \"lỗi cuối\" đang hiện ở System Health?",
"pos.unavailable": "Không có dữ liệu positioning",
"pos.waiting": "Chờ khớp entry",
"pos.close": "Chốt",
"pos.closeConfirm": "Chốt {dir} {symbol} ngay theo giá thị trường?\n\nPnL ước tính ≈ {pnl}",
"pos.closed": "Đã chốt lệnh: {status} {pnl}",
"pos.closeFail": "Chốt thất bại: {msg}",
"pos.manual": "Tay",
"port.seeking": "tìm signal",
"port.full": "FULL — nghỉ phân tích, vẫn trông lệnh",
"pos.lsUnavailable": "Không có Long/Short",
"pos.long": "Long {n}%",
"pos.short": "Short {n}%",
"wd.mon": "T2",
"wd.tue": "T3",
"wd.wed": "T4",
"wd.thu": "T5",
"wd.fri": "T6",
"wd.sat": "T7",
"wd.sun": "CN",
"month.1": "Tháng 1",
"month.2": "Tháng 2",
"month.3": "Tháng 3",
"month.4": "Tháng 4",
"month.5": "Tháng 5",
"month.6": "Tháng 6",
"month.7": "Tháng 7",
"month.8": "Tháng 8",
"month.9": "Tháng 9",
"month.10": "Tháng 10",
"month.11": "Tháng 11",
"month.12": "Tháng 12",
},
};

const MONTHS = ["month.1","month.2","month.3","month.4","month.5","month.6","month.7","month.8","month.9","month.10","month.11","month.12"];

function t(key, vars) {
  let s = (I18N[lang] && I18N[lang][key] !== undefined) ? I18N[lang][key] : (I18N.en[key] !== undefined ? I18N.en[key] : key);
  if (vars) {
    for (const [k, v] of Object.entries(vars)) s = s.split("{" + k + "}").join(String(v));
  }
  return s;
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", theme === "light" ? "light" : "dark");
  const btn = $("#themeBtn");
  if (btn) btn.textContent = theme === "light" ? "☀️" : "🌙";
  try { localStorage.setItem("nextra.theme", theme); } catch (_) {}
  try {
    if (typeof drawQuotaChart === "function" && quotaDays.length) drawQuotaChart();
  } catch (_) {}
}

function applyStaticI18n() {
  document.documentElement.setAttribute("lang", lang);
  const btn = $("#langBtn");
  if (btn) btn.textContent = lang.toUpperCase();
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.setAttribute("placeholder", t(el.dataset.i18nPh)); });
  document.querySelectorAll(".chat-chip[data-i18n]").forEach((el) => {
    const base = el.dataset.i18n;
    if (base === "chat.q1") el.dataset.q = t("chat.q1q");
    if (base === "chat.q2") el.dataset.q = t("chat.q2q");
    if (base === "chat.q3") el.dataset.q = t("chat.q3q");
  });
  try { localStorage.setItem("nextra.lang", lang); } catch (_) {}
  $("#pageTitle").textContent = panelTitle(current)[0];
  $("#pageSub").textContent = panelTitle(current)[1];
}

function applyI18n() {
  applyStaticI18n();
  const seq = ++loadSeq;
  showLoader();
  refresh(current).then(() => { if (seq === loadSeq) hideLoader(); });
}

$("#themeBtn").addEventListener("click", () => { theme = theme === "light" ? "dark" : "light"; applyTheme(); });
$("#langBtn").addEventListener("click", () => { lang = lang === "en" ? "vi" : "en"; applyI18n(); });

// -- navigation ---------------------------------------------------------------

const PANELS = ["dashboard", "signals", "daemon", "events", "trades", "statistics", "settings"];

function panelTitle(name) {
  return [t("title." + name), t("subtitle." + name)];
}
let current = "dashboard";

let loadSeq = 0;

function showLoader() {
  const el = $("#loading");
  if (el) el.classList.remove("hidden");
}

function hideLoader() {
  const el = $("#loading");
  if (el) el.classList.add("hidden");
}

function showPanel(name) {
  current = name;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.panel === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  const gear = document.getElementById("settingsGear");
  if (gear) gear.classList.toggle("active", name === "settings");
  $("#pageTitle").textContent = panelTitle(name)[0];
  $("#pageSub").textContent = panelTitle(name)[1];
  const seq = ++loadSeq;
  showLoader();
  refresh(name).then(() => { if (seq === loadSeq) hideLoader(); });
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => showPanel(btn.dataset.panel));
});

$("#settingsGear").addEventListener("click", () => showPanel("settings"));
$("#eventsBtn").addEventListener("click", () => showPanel("events"));

// -- dashboard ----------------------------------------------------------------

let chartSymbol = null;
let lastPortfolio = null;
let lastSymbols = [];
let lastPositions = [];

function renderPortfolioBar(portfolio, symbols) {
  const el = $("#portfolioBar");
  if (!el) return;
  const entries = (portfolio && portfolio.symbols) || [];
  const combo = (symbols && symbols.length ? symbols : []).filter((s) => s);
  if (!entries.length && !combo.length) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  const options = (combo.length ? combo : entries.map((e) => e.symbol)).filter((s) => s);
  if (chartSymbol === null || !options.includes(chartSymbol)) {
    chartSymbol = options[0] || window._symbol || "BTCUSDT";
  }
  const select = `<select id="chartSymbol" class="chart-interval" aria-label="Chart symbol">` +
    options.map((s) => `<option value="${s}"${s === chartSymbol ? " selected" : ""}>${s}</option>`).join("") +
    `</select>`;
  const current = entries.find((e) => e.symbol === chartSymbol) || null;
  const chip = current
    ? (() => {
      const holding = current.state !== "SEEKING";
      const label = holding ? current.state : t("port.seeking");
      const dot = holding ? " holding" : " ready";
      return `<span class="portfolio-chip"><span class="portfolio-dot${dot}"></span><strong>${escapeHtml(current.symbol)}</strong>&nbsp;${escapeHtml(label)}</span>`;
    })()
    : "";
  const full = portfolio && portfolio.full
    ? `<span class="portfolio-full">${t("port.full")}</span>`
    : "";
  el.hidden = false;
  el.innerHTML = select + chip + full;
}

$("#portfolioBar").addEventListener("change", (ev) => {
  if (ev.target && ev.target.id === "chartSymbol") {
    chartSymbol = ev.target.value;
    chartView = null;
    // Re-render the bar instantly from cached payload so the status chip
    // follows the dropdown without waiting for the next 15s refresh.
    renderPortfolioBar(lastPortfolio, lastSymbols);
    loadChart().catch(() => {});
  }
});

function renderHealth(health) {
  const stateText = (state) => state === "RUNNING" ? t("health.running") : (state === "STOPPED" ? t("health.stopped") : state);
  const pill = (state, ok) => `<span class="status-pill ${ok ? "ok" : "warn"}">${stateText(state)}</span>`;
  const rows = `
    <div class="kv"><span>${t("dash.healthState")}</span><span>${pill(health.state, health.state === "RUNNING")}</span></div>
    <div class="kv"><span>${t("dash.healthScheduler")}</span><span>${pill(health.scheduler, health.scheduler === "RUNNING")}</span></div>
    <div class="kv"><span>${t("dash.healthMonitor")}</span><span>${pill(health.monitor, health.monitor === "RUNNING")}</span></div>
    ${health.scheduler_last_tick ? `<div class="kv"><span>${t("dash.healthLastTick")}</span><span>${health.scheduler_last_tick}</span></div>` : ""}
    ${health.monitor_last_poll ? `<div class="kv"><span>${t("dash.healthLastPoll")}</span><span>${health.monitor_last_poll}</span></div>` : ""}
    ${health.last_error ? `<div class="kv"><span>${t("dash.healthLastError")}</span><span>${escapeHtml(health.last_error)}${health.last_error_at ? ` <span class="hint" style="color:var(--muted)">(${fmtVn(health.last_error_at)})</span>` : ""}</span></div>` : ""}`;
  return `<div class="form" style="gap:2px">${rows}</div>`;
}

function baseAsset(symbol) {
  const s = String(symbol || "");
  return s.endsWith("USDT") ? s.slice(0, -4) : s;
}

function renderPositionCards(list) {
  if (!list || !list.length) return emptyBox(t("dash.noSignal", { symbol: window._symbol || "BTCUSDT" }));
  return `<div class="pos-grid">` + list.map((item) => {
    const s = item.signal || {};
    const p = item.position || null;
    const o = item.position_outcomes || null;
    const u = item.unrealized || null;
    const dir = s.direction || (p && p.side) || "";
    const sideCls = dir === "SHORT" ? "short" : "long";
    const conf = (s.confidence !== null && s.confidence !== undefined) ? ` · ${fmtNum(s.confidence, 0)}` : "";
    let hero = "";
    if (u) {
      const up = Number(u.gross_pnl) >= 0;
      hero = `<div class="pos-pnl ${up ? "up" : "down"}">${up ? "+" : ""}$${fmtNum(u.gross_pnl, 2)} <span>(${fmtNum(u.pnl_percent, 2)}%)</span></div>`;
    } else {
      hero = `<div class="pos-pnl flat">${t("pos.waiting")}</div>`;
    }
    const tpNet = o ? Number(o.take_profit.net_pnl) : null;
    const slNet = o ? Number(o.stop_loss.net_pnl) : null;
    const tpRow = o ? `<div class="pos-line tp"><span>TP $${fmtNum(s.take_profit, 2)}</span><span>${tpNet >= 0 ? "+" : ""}${fmtNum(o.take_profit.net_pnl, 2)}</span></div>` : "";
    const slRow = o ? `<div class="pos-line sl"><span>SL $${fmtNum(s.stop_loss, 2)}</span><span>${slNet >= 0 ? "+" : ""}${fmtNum(o.stop_loss.net_pnl, 2)}</span></div>` : "";
    const meta = [
      p ? `${fmtNum(p.quantity, 6)} ${baseAsset(s.symbol || (p && p.symbol))}` : null,
      p ? `${fmtNum(p.margin, 2)}/${fmtNum(p.position_size, 2)}` : null,
      s.model_name || s.provider || null,
      s.analysis_timestamp ? fmtVn(s.analysis_timestamp) : null,
    ].filter(Boolean).join(" · ");
    const levChip = p ? `<span class="lev-chip">${p.leverage}x</span>` : "";
    const closeBtn = p ? `<button class="pos-close" data-close-sid="${s.id}">${t("pos.close")}</button>` : "";
    return `<div class="pos-card ${sideCls}">
      <div class="pos-head"><span><strong>${escapeHtml(s.symbol || "")}</strong> <span class="muted">${conf.replace(/^ · /, "")}</span></span><span class="pos-side">${badge(dir)}${levChip}</span></div>
      ${hero}
      <div class="pos-line entry"><span>Entry $${fmtNum(s.entry, 2)}</span>${closeBtn}</div>
      ${tpRow}${slRow}
      ${meta ? `<div class="pos-meta muted">${escapeHtml(meta)}</div>` : ""}
    </div>`;
  }).join("") + `</div>`;
}

async function loadDashboard() {
  const d = await api("/api/dashboard");
  const acc = d.account;
  const stats = d.statistics;
  const health = d.health;

  let cards = `
    <div class="card metric"><div class="metric-label">${t("dash.mBalance")}</div><div class="metric-value">$${acc ? fmtNum(acc.balance, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mEquity")}</div><div class="metric-value">$${acc ? fmtNum(acc.equity, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mNetPnl")}</div><div class="metric-value">$${stats ? fmtNum(stats.net_pnl, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mWinRate")}</div><div class="metric-value">${stats ? pct(stats.win_rate) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mPeak")}</div><div class="metric-value">$${acc ? fmtNum(acc.peak_equity, 2) : "—"}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mDD")}</div><div class="metric-value">${stats ? pct(stats.max_drawdown) : "—"}</div></div>`;
  $("#dashboardCards").innerHTML = cards;
  const positions = d.positions || [];
  lastPositions = positions;
  $("#currentSignal").innerHTML = renderPositionCards(positions);
  $("#systemHealth").innerHTML = renderHealth(health);
  lastPortfolio = d.portfolio || null;
  lastSymbols = d.symbols || [];
  renderPortfolioBar(lastPortfolio, lastSymbols);

  const statusEl = $("#statusPill");
  window._activeSignal = d.active_signal;
  window._symbol = d.symbol || window._symbol || "BTCUSDT";
  $("#sideSymbol").textContent = window._symbol;
  $("#priceCardTitle").textContent = t("dash.priceTitle", { symbol: window._symbol });
  if (positions.length > 1) {
    statusEl.className = "status-pill pending";
    statusEl.textContent = t("dash.statusMulti", { n: positions.length });
  } else if (d.active_signal) {
    statusEl.className = "status-pill pending";
    statusEl.textContent = t("dash.statusActive", { status: d.active_signal.status.toUpperCase() });
  } else {
    statusEl.className = "status-pill ok";
    statusEl.textContent = t("dash.statusIdle");
  }
}

// -- price chart ----------------------------------------------------------------

function tsMillis(t) {
  return t > 9999999999 ? t : t * 1000;
}

function drawPriceChart(canvas, chart, view) {
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = parent.clientWidth;
  const height = canvas.clientHeight || 360;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const allCandles = chart.candles || [];
  const overlay = chart.overlay || {};
  const total = allCandles.length;
  if (!total) { $("#chartMsg").textContent = t("chart.noData"); return; }
  $("#chartMsg").textContent = "";

  const end = total - (view ? view.backFromNewest : 0);
  const start = Math.max(0, end - (view ? view.count : total));
  const candles = allCandles.slice(start, end);
  const n = candles.length;

  const pad = { top: 8, right: 60, bottom: 18, left: 8 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const volH = plotH * 0.2;
  const priceH = plotH - volH;

  let min = Infinity, max = -Infinity;
  for (const c of candles) { if (c.l < min) min = c.l; if (c.h > max) max = c.h; }
  if (overlay.has_signal) {
    for (const p of [overlay.entry, overlay.take_profit, overlay.stop_loss]) {
      if (p < min) min = p;
      if (p > max) max = p;
    }
  }
  let spread = max - min;
  if (!spread || spread <= 0) spread = max || 1;
  const padRange = spread * 0.05;
  min -= padRange; max += padRange;
  const yFor = (price) => pad.top + priceH - ((price - min) / (max - min)) * priceH;

  const slotW = plotW / n;
  const bodyW = Math.max(1, slotW * 0.62);
  const font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";

  ctx.font = font;
  const step = (max - min) / 4;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const price = min + step * i;
    const y = yFor(price);
    ctx.strokeStyle = "rgba(128,128,128,0.13)";
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillStyle = "rgba(160,160,160,0.85)";
    ctx.fillText(price.toFixed(0), width - 8, y);
  }

  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xEvery = Math.max(1, Math.ceil(n / 6));
  for (let i = 0; i < n; i += xEvery) {
    ctx.fillStyle = "rgba(160,160,160,0.85)";
    ctx.fillText(fmtVnMs(tsMillis(candles[i].t)), pad.left + slotW * i + slotW / 2, pad.top + priceH + 4);
  }

  for (let i = 0; i < n; i++) {
    const c = candles[i];
    const x = pad.left + slotW * i + slotW / 2;
    const color = c.c >= c.o ? "#26a69a" : "#ef5350";
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, yFor(c.h)); ctx.lineTo(x, yFor(c.l)); ctx.stroke();
    const yTop = Math.min(yFor(c.o), yFor(c.c));
    const bodyH = Math.max(1, Math.abs(yFor(c.o) - yFor(c.c)));
    ctx.fillRect(x - bodyW / 2, yTop, bodyW, bodyH);
  }

  let maxVol = 0;
  for (const c of candles) if (c.v > maxVol) maxVol = c.v;
  if (maxVol > 0) {
    for (let i = 0; i < n; i++) {
      const c = candles[i];
      const x = pad.left + slotW * i + slotW / 2;
      const vh = (c.v / maxVol) * volH;
      ctx.fillStyle = c.c >= c.o ? "rgba(38,166,154,0.35)" : "rgba(239,83,80,0.35)";
      ctx.fillRect(x - bodyW / 2, pad.top + priceH + volH - vh, bodyW, vh);
    }
  }

  const chipH = 16;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  if (overlay.has_signal) {
    const levels = [
      { price: Number(overlay.stop_loss), text: "SL " + fmtNum(overlay.stop_loss, 1), color: "#ef5350" },
      { price: Number(overlay.entry), text: "Entry " + fmtNum(overlay.entry, 1), color: "#42a5f5" },
      { price: Number(overlay.take_profit), text: "TP " + fmtNum(overlay.take_profit, 1), color: "#26a69a" },
    ];
    for (const l of levels) {
      const y = yFor(l.price);
      if (y < pad.top - 1 || y > pad.top + priceH + 1) continue;
      ctx.strokeStyle = l.color; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
      ctx.setLineDash([]);
      const tw = ctx.measureText(l.text).width + 10;
      ctx.fillStyle = l.color;
      ctx.fillRect(pad.left + 2, y - 8, tw, chipH);
      ctx.fillStyle = "#0f1115"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(l.text, pad.left + 2 + 5, y);
    }
  }

  const newest = candles[n - 1];
  if (newest && end === total) {
    const yLast = yFor(newest.c);
    if (yLast >= pad.top && yLast <= pad.top + priceH) {
      const rightEdge = pad.left + slotW * n;
      const text = fmtNum(newest.c, 1);
      const tw = ctx.measureText(text).width + 10;
      const chipX = Math.max(rightEdge + 4, width - tw - 4);
      const chipW = Math.min(tw, width - chipX - 4);
      ctx.strokeStyle = "rgba(245,197,66,0.55)"; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, yLast); ctx.lineTo(Math.max(pad.left, chipX - 4), yLast); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f5c542";
      ctx.fillRect(chipX, yLast - 8, chipW, chipH);
      ctx.fillStyle = "#0f1115"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(text, chipX + 5, yLast);
    }
  }
}

let chartView = null;
let lastChart = null;
const CHART_WINDOW_DEFAULT = 50;
const CHART_WINDOW_MIN = 15;

function clampView(n) {
  const stored = chartView || { count: Math.min(CHART_WINDOW_DEFAULT, n), backFromNewest: 0 };
  const count = Math.max(CHART_WINDOW_MIN, Math.min(stored.count, n));
  const backFromNewest = Math.max(0, Math.min(stored.backFromNewest, n - count));
  return { count, backFromNewest };
}

function updateChartRange(view, candles) {
  const el = $("#chartRange");
  if (!el) return;
  const n = candles.length;
  const endC = n - view.backFromNewest;
  const start = Math.max(0, endC - view.count);
  if (start === 0 && endC === n) {
    el.textContent = t("chart.rangeAll", { n });
    return;
  }
  const fmtT = (c) => fmtVnMs(tsMillis(c.t));
  el.textContent = t("chart.range", { a: fmtT(candles[start]), b: fmtT(candles[endC - 1]), n: endC - start });
}

function redrawChart() {
  if (!lastChart || !lastChart.candles || !lastChart.candles.length) return;
  const candles = lastChart.candles;
  const view = clampView(candles.length);
  drawPriceChart($("#priceChart"), lastChart, view);
  updateChartRange(view, candles);
}

function zoomChart(factor) {
  if (!lastChart || !lastChart.candles.length) return;
  const n = lastChart.candles.length;
  const cur = clampView(n);
  // Keep the current pan position: zooming where you look. At the newest
  // candle (backFromNewest 0) this stays glued to the latest candle; in the
  // past it zooms in place instead of jumping back to newest.
  const count = Math.max(CHART_WINDOW_MIN, Math.min(Math.round(cur.count * factor), n));
  chartView = {
    count,
    backFromNewest: Math.max(0, Math.min(cur.backFromNewest, n - count)),
  };
  redrawChart();
}

function panChart(direction) {
  if (!lastChart || !lastChart.candles.length) return;
  const n = lastChart.candles.length;
  const cur = clampView(n);
  const step = Math.max(1, Math.round(cur.count * 0.2));
  chartView = {
    count: cur.count,
    backFromNewest: Math.max(0, Math.min(cur.backFromNewest + direction * step, n - cur.count)),
  };
  redrawChart();
}

function resetChart() {
  chartView = null;
  redrawChart();
}

const chartPad = { left: 8, right: 60 };
let chartDrag = null;

$("#priceChart").addEventListener("mousedown", (e) => {
  if (!lastChart || !lastChart.candles.length) return;
  const cur = clampView(lastChart.candles.length);
  chartDrag = { x: e.clientX, back: cur.backFromNewest, count: cur.count };
  e.currentTarget.classList.add("dragging");
  e.preventDefault();
});
window.addEventListener("mousemove", (e) => {
  if (!chartDrag) return;
  const canvas = $("#priceChart");
  const n = lastChart.candles.length;
  const slotW = (canvas.clientWidth - chartPad.left - chartPad.right) / chartDrag.count;
  const delta = Math.round((e.clientX - chartDrag.x) / slotW);
  chartView = {
    count: chartDrag.count,
    backFromNewest: Math.max(0, Math.min(chartDrag.back + delta, n - chartDrag.count)),
  };
  redrawChart();
});
window.addEventListener("mouseup", () => {
  if (!chartDrag) return;
  chartDrag = null;
  $("#priceChart").classList.remove("dragging");
});

$("#priceChart").addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomChart(e.deltaY < 0 ? 1 / 1.25 : 1.25);
}, { passive: false });

$("#chartZoomIn").addEventListener("click", () => zoomChart(1 / 1.25));
$("#chartZoomOut").addEventListener("click", () => zoomChart(1.25));
$("#chartPanLeft").addEventListener("click", () => panChart(1));
$("#chartPanRight").addEventListener("click", () => panChart(-1));
$("#chartReset").addEventListener("click", resetChart);

async function loadChart() {
  const interval = $("#chartInterval").value;
  const symbol = chartSymbol || window._symbol || "BTCUSDT";
  try {
    const data = (await api("/api/chart?interval=" + interval + "&limit=500&symbol=" + encodeURIComponent(symbol))).chart;
    lastChart = data;
    redrawChart();
    $("#priceCardTitle").textContent = t("dash.priceTitle", { symbol });
    $("#chartLive").textContent = t("chart.updated", { time: new Date().toTimeString().slice(0, 8) });
    const o = data.overlay || {};
    const candles = data.candles || [];
    let parts = [(chartSymbol || window._symbol || "BTCUSDT"), interval.toUpperCase()];
    if (o.has_signal) {
      parts.push(o.direction + (o.position_open ? " · position OPEN" : " · " + o.status));
    }
    if (candles.length) parts.push("last $" + fmtNum(candles[candles.length - 1].c, 2));
    $("#chartLegend").innerHTML = '<span class="muted">' + parts.join("  ·  ") + '</span>';
    loadPositioning().catch(() => {});
  } catch (err) {
    $("#chartMsg").textContent = t("chart.unavailable", { msg: err.message });
    lastChart = null;
    $("#chartLive").textContent = "";
  }
}

// -- futures positioning bar (Phase P1, under the chart) ----------------------

async function loadPositioning() {
  const el = $("#lsBar");
  if (!el) return;
  let data;
  try {
    data = await api("/api/positioning");
  } catch (_) {
    el.innerHTML = `<span class="muted">${t("pos.unavailable")}</span>`;
    return;
  }
  const fundOf = (d) => (d.funding_rate !== null && d.funding_rate !== undefined)
    ? ` · Funding ${d.funding_rate >= 0 ? "+" : ""}${fmtNum(d.funding_rate * 100, 3)}%/8h` : "";
  if (!data || !data.available || !(data.long_pct > 0)) {
    el.innerHTML = `<span class="muted">${t("pos.lsUnavailable")}${data ? fundOf(data) : ""}</span>`;
    return;
  }
  const long = Number(data.long_pct);
  const short = 100 - long;
  const oi = (data.open_interest !== null && data.open_interest !== undefined)
    ? ` · OI ${fmtNum(data.open_interest, 1)} BTC` : "";
  el.innerHTML =
    `<div class="ls-track"><div class="ls-long" style="width:${long.toFixed(1)}%"></div>` +
    `<div class="ls-short" style="width:${short.toFixed(1)}%"></div></div>` +
    `<div class="ls-label"><span class="ls-long-t">${t("pos.long", { n: long.toFixed(1) })}</span>` +
    `<span class="muted">${escapeHtml(data.symbol || "")}${fundOf(data)}${oi}</span>` +
    `<span class="ls-short-t">${t("pos.short", { n: short.toFixed(1) })}</span></div>`;
}

// -- signal chat (read-only Q&A drawer) ----------------------------------------

function openSignalChat() {
  const active = window._activeSignal;
  $("#chatContext").textContent = active
    ? ("#" + active.id + " · " + active.direction + " · " + fmtVn(active.created_at))
    : t("chat.noSignal");
  $("#chatDrawer").classList.add("open");
  $("#chatDrawer").setAttribute("aria-hidden", "false");
  $("#chatBackdrop").classList.remove("hidden");
  $("#chatInput").focus();
}

function closeSignalChat() {
  $("#chatDrawer").classList.remove("open");
  $("#chatDrawer").setAttribute("aria-hidden", "true");
  $("#chatBackdrop").classList.add("hidden");
}

function chatMsg(kind, text) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.textContent = text;
  $("#chatMessages").appendChild(el);
  $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
  return el;
}

async function sendSignalChat(questionText) {
  const question = (questionText || $("#chatInput").value).trim();
  if (!question) return;
  $("#chatInput").value = "";
  chatMsg("user", question);
  const bot = chatMsg("bot", "…");
  bot.classList.add("typing");
  const sendBtn = $("#chatSend");
  sendBtn.disabled = true;
  let first = true;
  try {
    const resp = await fetch("/api/signal-chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    if (!resp.ok) {
      let message = "HTTP " + resp.status;
      try { message = (await resp.json()).error || message; } catch (_) { /* empty */ }
      throw new Error(message);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let event;
        try { event = JSON.parse(line); } catch (_) { continue; }
        if (event.delta) {
          if (first) { bot.textContent = ""; bot.classList.remove("typing"); first = false; }
          bot.textContent += event.delta;
        } else if (event.text) {
          if (first) { bot.classList.remove("typing"); first = false; }
          bot.textContent = event.text;
        } else if (event.error) {
          if (first) { bot.classList.remove("typing"); first = false; }
          bot.textContent = event.error;
          bot.classList.add("error");
        }
      }
      $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
    }
    if (first) { bot.classList.remove("typing"); bot.textContent = "—"; }
  } catch (err) {
    bot.classList.remove("typing");
    bot.textContent = "❌ " + err.message;
    bot.classList.add("error");
  } finally {
    sendBtn.disabled = false;
    $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
  }
}

$("#chatOpenBtn").addEventListener("click", openSignalChat);
$("#chatClose").addEventListener("click", closeSignalChat);
$("#chatBackdrop").addEventListener("click", closeSignalChat);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSignalChat(); });
$("#chatSend").addEventListener("click", () => sendSignalChat());
$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendSignalChat();
});
$("#chatClear").addEventListener("click", () => {
  $("#chatMessages").innerHTML = "";
  const empty = document.createElement("p");
  empty.className = "muted";
  empty.textContent = t("chat.empty");
  $("#chatMessages").appendChild(empty);
});
document.querySelectorAll(".chat-chip").forEach((chip) =>
  chip.addEventListener("click", () => sendSignalChat(chip.dataset.q))
);

// -- signals ------------------------------------------------------------------

const VN_OFFSET_MIN = 7 * 3600 * 1000;

function vnDayStartUtc(daysAgo) {
  const nowVn = Date.now() + VN_OFFSET_MIN;
  const startVn = Math.floor(nowVn / 86400000) * 86400000 - daysAgo * 86400000;
  return new Date(startVn - VN_OFFSET_MIN).toISOString();
}

function sigTimeRange() {
  const v = $("#sigTime").value;
  if (!v) return null;
  const days = v === "today" ? 0 : v === "7d" ? 6 : v === "30d" ? 29 : NaN;
  if (!Number.isFinite(days)) return null;
  return { since: vnDayStartUtc(days), until: new Date().toISOString() };
}

function sigSelectedCount() {
  return document.querySelectorAll("#signalsTable .sig-check:checked").length;
}

function updateSigDeleteCount() {
  const n = sigSelectedCount();
  $("#sigDeleteSel").disabled = n === 0;
  $("#sigDeleteSel").textContent = t("sig.deleteSel", { n });
}

function renderSignalTable(list) {
  if (!list.length) return emptyBox(t("sig.empty"));
  const head = `<tr><th></th><th>${t("sig.thId")}</th><th>${t("sig.thSymbol")}</th><th>${t("sig.thDir")}</th><th>${t("sig.thEntry")}</th><th>${t("sig.thTP")}</th><th>${t("sig.thSL")}</th><th>${t("sig.thConf")}</th><th>${t("sig.thStatus")}</th><th>${t("sig.thResult")}</th><th>${t("sig.thModel")}</th><th>${t("sig.thTime")}</th></tr>`;
  const body = list.map((s) => `<tr>
    <td class="sig-checkbox-cell"><input type="checkbox" class="sig-check" data-id="${s.id}"></td>
    <td data-label="${t("sig.thId")}">${s.id}</td>
    <td data-label="${t("sig.thSymbol")}">${escapeHtml(s.symbol || "—")}</td>
    <td data-label="${t("sig.thDir")}">${badge(s.direction)}</td>
    <td data-label="${t("sig.thEntry")}">$${fmtNum(s.entry, 2)}</td>
    <td data-label="${t("sig.thTP")}">$${fmtNum(s.take_profit, 2)}</td>
    <td data-label="${t("sig.thSL")}">$${fmtNum(s.stop_loss, 2)}</td>
    <td data-label="${t("sig.thConf")}">${fmtNum(s.confidence, 0)}</td>
    <td data-label="${t("sig.thStatus")}">${badge(s.status)}</td>
    <td data-label="${t("sig.thResult")}">${s.result ? badge(s.result) : "—"}${s.close_reason === "MANUAL" ? ` <span class="manual-tag">${t("pos.manual")}</span>` : ""}</td>
    <td data-label="${t("sig.thModel")}">${s.model_name}</td>
    <td data-label="${t("sig.thTime")}">${fmtVn(s.created_at)}</td>
  </tr>`).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function daemonTimeRange() {
  const v = $("#ddTime").value;
  if (!v) return null;
  return sigTimeRange(); // same "today/7d/30d" boundaries as the signals tab
}

function renderDaemonTable(list) {
  if (!list.length) return emptyBox(t("daemon.empty"));
  const head = `<tr><th></th><th>${t("daemon.thRecorded")}</th><th>${t("daemon.thCandle")}</th><th>${t("daemon.thSymbol")}</th><th>${t("daemon.thDecision")}</th><th>${t("daemon.thConf")}</th><th>${t("daemon.thOutcome")}</th><th>${t("daemon.thSignal")}</th><th>${t("daemon.thModel")}</th><th>${t("daemon.thClose")}</th><th>${t("daemon.thCalls")}</th><th>${t("daemon.thNotes")}</th></tr>`;
  const body = list.map((e) => {
    const decision = e.decision ? badge(e.decision) : "—";
    const outcomeCls = String(e.outcome || "").toLowerCase();
    const outcome = `<span class="badge ${outcomeCls.replace("_", "-")}">${e.outcome || ""}</span>`;
    const sig = e.signal_id
      ? `<button type="button" class="btn-link sig-link" data-id="${e.signal_id}">#${e.signal_id}</button>`
      : "—";
    const notes = (e.error_notes || "") ? `<span class="note-cell">${escapeHtml(truncate(e.error_notes, 60))}</span>` : "—";
    const res = `$${fmtNum(e.close_price, 2)}`;
    const calls = (e.llm_calls === null || e.llm_calls === undefined) ? "—" : e.llm_calls;
    return `<tr class="dd-row" data-row="${e.id}">
      <td class="sig-checkbox-cell"><input type="checkbox" class="dd-check" data-id="${e.id}"></td>
      <td data-label="${t("daemon.thRecorded")}">${fmtVn(e.recorded_at)}</td>
      <td data-label="${t("daemon.thCandle")}">${fmtVnMs(e.candle_timestamp_ms)}</td>
      <td data-label="${t("daemon.thSymbol")}">${escapeHtml(e.symbol || "—")}</td>
      <td data-label="${t("daemon.thDecision")}">${decision}</td>
      <td data-label="${t("daemon.thConf")}">${fmtNum(e.confidence, 0)}</td>
      <td data-label="${t("daemon.thOutcome")}">${outcome}</td>
      <td data-label="${t("daemon.thSignal")}">${sig}</td>
      <td data-label="${t("daemon.thModel")}">${escapeHtml(e.model || e.provider || "—")}</td>
      <td data-label="${t("daemon.thClose")}">${res}</td>
      <td data-label="${t("daemon.thCalls")}">${calls}</td>
      <td data-label="${t("daemon.thNotes")}">${notes}</td>
    </tr>
    <tr class="dd-detail" id="dd-detail-${e.id}" hidden><td colspan="11">
      <div class="dd-detail-box">
        ${ddLevelsSection(e)}
        ${ddQuotaSection(e)}
        ${ddIndicatorsSection(e)}
        ${e.reasoning ? `<div class="dd-sec"><div class="dd-sec-title">${t("daemon.secReasoning")}</div><div class="dd-reason">${escapeHtml(e.reasoning)}</div></div>` : ""}
        ${e.error_notes ? `<div class="dd-sec"><div class="dd-sec-title">${t("daemon.secError")}</div><div class="dd-reason err">${escapeHtml(e.error_notes)}</div></div>` : ""}
      </div>
    </td></tr>`;
  }).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function truncate(text, n) {
  return String(text).length > n ? String(text).slice(0, n) + "…" : String(text);
}

const DD_IND_GROUPS = [
  ["gTrend", ["ema20", "ema50", "ema200"]],
  ["gMomentum", ["rsi14", "macd", "macd_signal", "macd_histogram"]],
  ["gVol", ["atr14", "bb_mid", "bb_upper", "bb_lower"]],
  ["gStrength", ["adx14", "adx_plus_di", "adx_minus_di"]],
  ["gVolume", ["volume_sma20", "volume_ratio"]],
];

function ddRiskReward(e) {
  const toN = (v) => (v === null || v === undefined || v === "" ? null : Number(v));
  const entry = toN(e.entry), sl = toN(e.stop_loss), tp = toN(e.take_profit);
  if (![entry, sl, tp].every((v) => v !== null && Number.isFinite(v))) return "";
  let rr = null;
  if (e.decision === "LONG" && sl < entry && tp > entry) rr = (tp - entry) / (entry - sl);
  if (e.decision === "SHORT" && sl > entry && tp < entry) rr = (entry - tp) / (sl - entry);
  return rr === null ? "" : `RR ${fmtNum(rr, 2)}`;
}

function ddLevelsSection(e) {
  if (e.decision !== "LONG" && e.decision !== "SHORT") {
    return `<div class="dd-sec"><div class="dd-sec-title">${t("daemon.secOrder")}</div><div class="muted">${badge(e.decision || "WAIT")} ${t("daemon.noOrder")}</div></div>`;
  }
  const rr = ddRiskReward(e);
  return `<div class="dd-sec"><div class="dd-sec-title">${t("daemon.secOrder")} ${badge(e.decision)}${e.confidence !== null && e.confidence !== undefined ? ` · Conf ${fmtNum(e.confidence, 0)}` : ""}${rr ? ` · ${rr}` : ""}</div>
    <div class="dd-levels">
      <div class="dd-level"><span>${t("dash.kEntry")}</span><strong>$${fmtNum(e.entry, 2)}</strong></div>
      <div class="dd-level sl"><span>${t("dash.kSL")}</span><strong>$${fmtNum(e.stop_loss, 2)}</strong></div>
      <div class="dd-level tp"><span>${t("dash.kTP")}</span><strong>$${fmtNum(e.take_profit, 2)}</strong></div>
    </div></div>`;
}

function ddQuotaSection(e) {
  const hasTokens = [e.prompt_tokens, e.completion_tokens, e.total_tokens].some((v) => v !== null && v !== undefined);
  const tilde = e.tokens_estimated ? "~" : "";
  const tok = (v) => (v === null || v === undefined ? "—" : tilde + v);
  return `<div class="dd-sec"><div class="dd-sec-title">${t("daemon.secQuota")}</div>
    <div class="dd-stats">
      <div class="dd-stat"><span>${t("daemon.kModel")}</span><strong>${escapeHtml(e.model || e.provider || "—")}</strong></div>
      <div class="dd-stat"><span>${t("daemon.kTemp")}</span><strong>${e.temperature ?? "—"}</strong></div>
      <div class="dd-stat"><span>${t("daemon.kCalls")}</span><strong>${e.llm_calls ?? "—"}</strong></div>
      <div class="dd-stat" title="${e.tokens_estimated ? t("daemon.tokensEstTitle") : t("daemon.tokensRealTitle")}"><span>${e.tokens_estimated ? t("daemon.kTokensEst") : t("daemon.kTokens")}</span><strong>${hasTokens ? `${tok(e.prompt_tokens)} / ${tok(e.completion_tokens)} / ${tok(e.total_tokens)}` : "—"}</strong></div>
    </div></div>`;
}

function ddIndicatorsSection(e) {
  const ind = e.indicators || {};
  const groups = DD_IND_GROUPS.map(([labelKey, keys]) => {
    const cells = keys.filter((k) => ind[k] !== null && ind[k] !== undefined && ind[k] !== "")
      .map((k) => `<div class="kv"><span>${k}</span><span>${fmtNum(ind[k], 2)}</span></div>`).join("");
    return cells ? `<div class="dd-ind-group"><div class="dd-ind-title">${t("daemon." + labelKey)}</div>${cells}</div>` : "";
  }).join("");
  if (!groups) return "";
  return `<div class="dd-sec"><div class="dd-sec-title">${t("daemon.secIndicators")}</div><div class="dd-ind-grid">${groups}</div></div>`;
}

const DD_PAGE_SIZE = 50;
let ddPage = 0;

async function loadDaemonLog() {
  const p = new URLSearchParams({ limit: String(DD_PAGE_SIZE + 1), offset: String(ddPage * DD_PAGE_SIZE) });
  const dir = $("#ddDir").value;
  const range = daemonTimeRange();
  if (dir) p.set("decision", dir);
  if (range) p.set("since", range.since);
  const data = await api("/api/daemon-log?" + p.toString());
  const rows = data.rows || [];
  $("#daemonTable").innerHTML = renderDaemonTable(rows.slice(0, DD_PAGE_SIZE));
  $("#ddSelAll").textContent = t("daemon.selectAll");
  updateDdDeleteCount();
  renderDaemonPager(rows.length > DD_PAGE_SIZE);
}

function pagerHtml(page, pageSize, hasMore) {
  if (page === 0 && !hasMore) return null;
  const from = page * pageSize + 1;
  return `<button type="button" class="btn secondary" data-pg="prev"${page === 0 ? " disabled" : ""}>${t("pager.prev")}</button>` +
    `<span class="pager-label">${t("pager.label", { n: page + 1, from })}</span>` +
    `<button type="button" class="btn secondary" data-pg="next"${hasMore ? "" : " disabled"}>${t("pager.next")}</button>`;
}

function renderDaemonPager(hasMore) {
  const el = $("#daemonPager");
  if (!el) return;
  const html = pagerHtml(ddPage, DD_PAGE_SIZE, hasMore);
  if (html === null) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = html;
}

$("#daemonPager").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-pg]");
  if (!btn || btn.disabled) return;
  ddPage = btn.dataset.pg === "next" ? ddPage + 1 : Math.max(0, ddPage - 1);
  showLoader();
  loadDaemonLog().catch((e) => toast(e.message, "error")).finally(hideLoader);
});

async function clearDaemonLog() {
  if (!confirm(t("daemon.confirmClear"))) return;
  const res = await api("/api/daemon-log", { method: "DELETE" });
  toast(t("daemon.cleared", { n: res.cleared || 0 }), "ok");
  ddPage = 0;
  loadDaemonLog().catch((e) => toast(e.message, "error"));
}

const SIG_PAGE_SIZE = 25;
let sigPage = 0;

async function loadSignals() {
  const p = new URLSearchParams({ limit: String(SIG_PAGE_SIZE + 1), offset: String(sigPage * SIG_PAGE_SIZE) });
  const dir = $("#sigDir").value;
  const st = $("#sigStatus").value;
  if (dir) p.set("direction", dir);
  if (st) p.set("status", st);
  const range = sigTimeRange();
  if (range) {
    p.set("since", range.since);
    p.set("until", range.until);
  }
  const data = await api("/api/signals?" + p.toString());
  const list = data.signals || [];
  $("#signalsTable").innerHTML = renderSignalTable(list.slice(0, SIG_PAGE_SIZE));
  $("#sigSelAll").textContent = t("sig.selectAll");
  updateSigDeleteCount();
  renderSignalsPager(list.length > SIG_PAGE_SIZE);
}

function renderSignalsPager(hasMore) {
  const el = $("#signalsPager");
  if (!el) return;
  const html = pagerHtml(sigPage, SIG_PAGE_SIZE, hasMore);
  if (html === null) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = html;
}

$("#signalsPager").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-pg]");
  if (!btn || btn.disabled) return;
  sigPage = btn.dataset.pg === "next" ? sigPage + 1 : Math.max(0, sigPage - 1);
  showLoader();
  loadSignals().catch((e) => toast(e.message, "error")).finally(hideLoader);
});

async function deleteSignals(body) {
  const res = await api("/api/signals", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const c = res.counts || {};
  const skipped = (c.skipped_open || []).length;
  toast(
    t("sig.deleted", { n: (c.deleted || []).length, kept: skipped ? t("sig.keptOpen", { n: skipped }) : "" }),
    skipped ? "warn" : "ok"
  );
  await Promise.allSettled([loadSignals(), loadTrades(), loadStatistics()]);
  if (sigPage > 0 && !document.querySelector("#signalsTable tbody tr")) {
    sigPage -= 1;
    await loadSignals().catch((e) => toast(e.message, "error"));
  }
}

async function loadTrades() {
  const data = await api("/api/trades?limit=100");
  const list = data.trades || [];
  if (!list.length) { $("#tradesTable").innerHTML = emptyBox(t("trades.empty")); return; }
  const head = `<tr><th>${t("trades.thId")}</th><th>${t("trades.thSymbol")}</th><th>${t("trades.thSide")}</th><th>${t("trades.thEntry")}</th><th>${t("trades.thExit")}</th><th>${t("trades.thQty")}</th><th>${t("trades.thLev")}</th><th>${t("trades.thGross")}</th><th>${t("trades.thFee")}</th><th>${t("trades.thNet")}</th><th>${t("trades.thResult")}</th><th>${t("trades.thOpened")}</th><th>${t("trades.thClosed")}</th></tr>`;
  const body = list.map((row) => `<tr>
    <td data-label="${t("trades.thId")}">${row.id}</td>
    <td data-label="${t("trades.thSymbol")}">${escapeHtml(row.symbol || "—")}</td>
    <td data-label="${t("trades.thSide")}">${badge(row.side)}</td>
    <td data-label="${t("trades.thEntry")}">$${fmtNum(row.entry_price, 2)}</td>
    <td data-label="${t("trades.thExit")}">$${fmtNum(row.exit_price, 2)}</td>
    <td data-label="${t("trades.thQty")}">${fmtNum(row.quantity, 6)}</td>
    <td data-label="${t("trades.thLev")}">${row.leverage != null ? `${row.leverage}x` : "—"}</td>
    <td data-label="${t("trades.thGross")}">$${fmtNum(row.gross_pnl, 2)}</td>
    <td data-label="${t("trades.thFee")}">$${fmtNum(row.fee, 4)}</td>
    <td data-label="${t("trades.thNet")}" style="color:${Number(row.net_pnl) >= 0 ? "var(--green)" : "var(--red)"}">$${fmtNum(row.net_pnl, 2)}</td>
    <td data-label="${t("trades.thResult")}">${badge(row.result)}</td>
    <td data-label="${t("trades.thOpened")}">${fmtVn(row.opened_at)}</td>
    <td data-label="${t("trades.thClosed")}">${fmtVn(row.closed_at)}</td>
  </tr>`).join("");
  $("#tradesTable").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function loadStatistics() {
  const d = (await api("/api/statistics")).statistics || {};
  const cards = `
    <div class="card metric"><div class="metric-label">${t("dash.mNetPnl")}</div><div class="metric-value">$${fmtNum(d.net_pnl, 2)}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mWinRate")}</div><div class="metric-value">${pct(d.win_rate)}</div></div>
    <div class="card metric"><div class="metric-label">${t("stats.mTrades")}</div><div class="metric-value">${fmtNum(d.total_trades, 0)}</div></div>
    <div class="card metric"><div class="metric-label">${t("stats.mWL")}</div><div class="metric-value">${fmtNum(d.wins, 0)} / ${fmtNum(d.losses, 0)}</div></div>
    <div class="card metric"><div class="metric-label">${t("dash.mDD")}</div><div class="metric-value">${pct(d.max_drawdown)}</div></div>
    <div class="card metric"><div class="metric-label">${t("stats.mFees")}</div><div class="metric-value">$${fmtNum(d.total_fees, 4)}</div></div>`;
  $("#statsCards").innerHTML = cards;
  const detail = Object.entries(d).filter(([k]) => !["wins", "losses", "total_trades"].includes(k))
    .map(([k, v]) => `<div class="kv"><span>${k.replace(/_/g, " ")}</span><span>${fmtNum(v, 4)}</span></div>`).join("");
  $("#statsDetail").innerHTML = `<div class="form" style="gap:2px">${detail}</div>`;
}

// -- quota usage (Phase Q, Statistics tab) --------------------------------------

const QUOTA_PAGE_SIZE = 15;
let quotaPage = 0;
let quotaDays = [];

function fmtCompact(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + "k";
  return String(Math.round(n));
}

function fmtVnDay(isoDay) {
  const parts = String(isoDay || "").split("-");
  return parts.length === 3 ? `${parts[2]}/${parts[1]}` : String(isoDay || "—");
}

let quotaLayout = [];
let quotaPrice = 0;

function fmtVND(v) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return fmtCompact(n) + "₫";
}

function fmtPct1(v) {
  if (v === null || v === undefined || v === "") return "—";
  return fmtNum(Number(v) * 100, 1) + "%";
}

async function loadQuota() {
  const range = ($("#quotaRange") && $("#quotaRange").value) || "7d";
  const data = await api("/api/quota?range=" + encodeURIComponent(range));
  quotaPrice = Number(data.per_call_vnd || 0);
  const priceInput = $("#quotaPrice");
  if (priceInput && document.activeElement !== priceInput) {
    priceInput.value = quotaPrice || "";
  }
  const note = $("#quotaCostNote");
  if (note) {
    note.textContent = quotaPrice > 0
      ? t("quota.costNote", { price: fmtVND(quotaPrice) })
      : t("quota.costUnset");
  }
  const cards = `
    <div class="card metric"><div class="metric-label">${t("quota.mRequests")}</div><div class="metric-value">${fmtCompact(data.llm_calls)}</div></div>
    <div class="card metric"><div class="metric-label">${t("quota.mTotal")}</div><div class="metric-value">${fmtCompact(data.total_tokens)}</div></div>
    <div class="card metric"><div class="metric-label">${t("quota.mIn")}</div><div class="metric-value">↓ ${fmtCompact(data.prompt_tokens)}</div></div>
    <div class="card metric"><div class="metric-label">${t("quota.mOut")}</div><div class="metric-value">↑ ${fmtCompact(data.completion_tokens)}</div></div>
    <div class="card metric"><div class="metric-label">${t("quota.mSuccess")}</div><div class="metric-value">${fmtPct1(data.success_rate)}</div></div>
    <div class="card metric"><div class="metric-label">${t("quota.mCost")}</div><div class="metric-value">${quotaPrice > 0 ? fmtVND(data.cost_vnd) : "—"}</div></div>`;
  $("#quotaCards").innerHTML = cards;
  quotaDays = (data && data.days) || [];
  quotaPage = 0;
  renderQuotaModels(data.by_model || []);
  renderQuotaTable();
  drawQuotaChart();
}

function renderQuotaModels(rows) {
  const box = $("#quotaModels");
  if (!box) return;
  if (!rows.length) {
    box.innerHTML = emptyBox(t("quota.empty"));
    return;
  }
  const head = `<tr><th>${t("quota.thModel")}</th><th>${t("quota.thCalls")}</th><th>${t("quota.thTotal")}</th><th>${t("quota.thSuccess")}</th><th>${t("quota.thCost")}</th></tr>`;
  const body = rows.map((m) => `<tr>
    <td data-label="${t("quota.thModel")}">${escapeHtml(m.model || "?")}</td>
    <td data-label="${t("quota.thCalls")}">${fmtCompact(m.llm_calls)}</td>
    <td data-label="${t("quota.thTotal")}">${fmtCompact(m.total_tokens)}</td>
    <td data-label="${t("quota.thSuccess")}">${fmtPct1(m.success_rate)}</td>
    <td data-label="${t("quota.thCost")}">${quotaPrice > 0 ? fmtVND(m.cost_vnd) : "—"}</td>
  </tr>`).join("");
  box.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function quotaChartRows() {
  // Oldest-first for the chart; bucket by week past 31 days (noted in UI).
  const asc = quotaDays.slice().reverse();
  if (asc.length <= 31) return { rows: asc, weekly: false };
  const buckets = [];
  for (let i = 0; i < asc.length; i += 7) {
    const chunk = asc.slice(i, i + 7);
    const sum = (k) => chunk.reduce((a, d) => a + (Number(d[k]) || 0), 0);
    buckets.push({
      day: chunk[0].day + "…",
      llm_calls: sum("llm_calls"),
      prompt_tokens: sum("prompt_tokens"),
      completion_tokens: sum("completion_tokens"),
      total_tokens: sum("total_tokens"),
      cost_vnd: sum("cost_vnd"),
    });
  }
  return { rows: buckets, weekly: true };
}

function cssVar(name, fallback) {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (_) {
    return fallback;
  }
}

function drawQuotaChart() {
  const canvas = $("#quotaChart");
  if (!canvas) return;
  quotaLayout = [];
  const tip = $("#quotaTip");
  if (tip) tip.hidden = true;
  const { rows, weekly } = quotaChartRows();
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = parent.clientWidth || 600;
  const height = 220;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!rows.length) return;
  const pad = { top: 10, right: 52, bottom: 20, left: 44 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxTok = Math.max(1, ...rows.map((d) => Number(d.total_tokens) || 0));
  const maxCost = Math.max(0, ...rows.map((d) => Number(d.cost_vnd) || 0));
  const tokStep = niceStep(maxTok / 4);
  const yMax = Math.ceil(maxTok / tokStep) * tokStep;
  const costMax = maxCost > 0 ? maxCost * 1.15 : 1;
  const font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.font = font;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const gridColor = cssVar("--border", "#1e2a37");
  const textColor = cssVar("--muted", "#8b9aab");
  const inColor = cssVar("--accent", "#3b82f6");
  const outColor = cssVar("--amber", "#f59e0b");
  const costColor = cssVar("--red", "#ef4444");
  for (let g = 0; g <= 4; g++) {
    const value = (yMax / 4) * g;
    const y = pad.top + plotH - (value / yMax) * plotH;
    ctx.strokeStyle = gridColor;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = textColor;
    ctx.fillText(fmtCompact(value), pad.left - 6, y);
  }
  if (maxCost > 0) {
    ctx.fillStyle = textColor;
    ctx.fillText(fmtVND(costMax), width - 4, pad.top + 6);
  }
  const slotW = plotW / rows.length;
  const bodyW = Math.max(2, Math.min(34, slotW * 0.55));
  const yForTok = (v) => pad.top + plotH - (Math.min(v, yMax) / yMax) * plotH;
  const yForCost = (v) => pad.top + plotH - (v / costMax) * plotH;
  const labelEvery = Math.max(1, Math.ceil(rows.length / 8));
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  let prevCostXY = null;
  rows.forEach((d, i) => {
    const cx = pad.left + slotW * i + slotW / 2;
    const prompt = Number(d.prompt_tokens) || 0;
    const completion = Number(d.completion_tokens) || 0;
    const baseY = pad.top + plotH;
    const inH = (Math.min(prompt, yMax) / yMax) * plotH;
    const outH = (Math.min(prompt + completion, yMax) / yMax) * plotH - inH;
    ctx.fillStyle = inColor;
    ctx.fillRect(cx - bodyW / 2, baseY - inH, bodyW, Math.max(inH, 0));
    ctx.fillStyle = outColor;
    ctx.fillRect(cx - bodyW / 2, baseY - inH - outH, bodyW, Math.max(outH, 0));
    quotaLayout.push({ x0: cx - slotW / 2, x1: cx + slotW / 2, day: d });
    if (maxCost > 0) {
      const cy = yForCost(Number(d.cost_vnd) || 0);
      ctx.fillStyle = costColor;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
      if (prevCostXY !== null) {
        ctx.strokeStyle = costColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(prevCostXY[0], prevCostXY[1]);
        ctx.lineTo(cx, cy);
        ctx.stroke();
        ctx.lineWidth = 1;
      }
      prevCostXY = [cx, cy];
    }
    if (i % labelEvery === 0) {
      ctx.fillStyle = textColor;
      ctx.fillText(fmtVnDay(d.day), cx, pad.top + plotH + 4);
    }
  });
  if (weekly) {
    ctx.fillStyle = textColor;
    ctx.textAlign = "left";
    ctx.fillText(t("quota.weekly"), pad.left, pad.top + 10);
  }
}

function niceStep(raw) {
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1))));
  const norm = raw / mag;
  const pick = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10;
  return pick * mag;
}

function quotaTooltip(day) {
  const tip = $("#quotaTip");
  if (!tip || !day) {
    if (tip) tip.hidden = true;
    return;
  }
  const estBit = Number(day.total_tokens_estimated || 0) > 0;
  tip.innerHTML =
    `<div class="quota-tip-day">${fmtVnDay(day.day)}</div>` +
    `<div>${t("quota.thCalls")}: <strong>${fmtCompact(day.llm_calls)}</strong></div>` +
    `<div>↓ ${fmtCompact(day.prompt_tokens)} · ↑ ${fmtCompact(day.completion_tokens)} · Σ ${fmtCompact(day.total_tokens)}${estBit ? " (~)" : ""}</div>` +
    (quotaPrice > 0 ? `<div>💰 ${fmtVND(day.cost_vnd)} (${fmtCompact(day.llm_calls)} × ${fmtVND(quotaPrice)})</div>` : "");
  tip.hidden = false;
}

function quotaHover(clientX, clientY) {
  const canvas = $("#quotaChart");
  const tip = $("#quotaTip");
  if (!canvas || !tip || !quotaLayout.length) {
    if (tip) tip.hidden = true;
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  let best = -1;
  let bestDist = Infinity;
  quotaLayout.forEach((cell, i) => {
    const cx = (cell.x0 + cell.x1) / 2;
    const dist = Math.abs(cx - x);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  });
  if (best < 0) return;
  const cell = quotaLayout[best];
  const cx = (cell.x0 + cell.x1) / 2;
  quotaTooltip(cell.day);
  const wrapWidth = canvas.parentElement.clientWidth || 600;
  tip.style.left = Math.min(Math.max(cx + 12, 8), Math.max(8, wrapWidth - 190)) + "px";
  tip.style.top = Math.max(clientY - rect.top - 10, 8) + "px";
}

$("#quotaChart").addEventListener("mousemove", (ev) => quotaHover(ev.clientX, ev.clientY));
$("#quotaChart").addEventListener("mouseleave", () => quotaTooltip(null));
$("#quotaChart").addEventListener("touchstart", (ev) => {
  const touch = ev.touches && ev.touches[0];
  if (touch) quotaHover(touch.clientX, touch.clientY);
}, { passive: true });
$("#quotaChart").addEventListener("touchmove", (ev) => {
  const touch = ev.touches && ev.touches[0];
  if (touch) quotaHover(touch.clientX, touch.clientY);
}, { passive: true });

$("#quotaRange").addEventListener("change", () => {
  quotaPage = 0;
  showLoader();
  loadQuota().catch((e) => toast(e.message, "error")).finally(hideLoader);
});

$("#quotaSavePrice").addEventListener("click", async () => {
  const raw = $("#quotaPrice").value;
  try {
    await api("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quota_cost: { per_call_vnd: raw === "" ? 0 : Number(raw) } }),
    });
    toast(t("quota.saved"));
    loadQuota().catch((e) => toast(e.message, "error"));
  } catch (err) {
    toast(t("quota.saveFail", { msg: err.message }), "error");
  }
});

function renderQuotaTable() {
  const table = $("#quotaTable");
  if (!table) return;
  if (!quotaDays.length) {
    table.innerHTML = emptyBox(t("quota.empty"));
  } else {
    const rows = quotaDays.slice(quotaPage * QUOTA_PAGE_SIZE, quotaPage * QUOTA_PAGE_SIZE + QUOTA_PAGE_SIZE);
    const head = `<tr><th>${t("quota.thDay")}</th><th>${t("quota.thCalls")}</th><th>${t("quota.thIn")}</th><th>${t("quota.thOut")}</th><th>${t("quota.thTotal")}</th><th>${t("quota.thCost")}</th></tr>`;
    const body = rows.map((d) => `<tr>
      <td data-label="${t("quota.thDay")}">${fmtVnDay(d.day)}</td>
      <td data-label="${t("quota.thCalls")}">${fmtCompact(d.llm_calls)}</td>
      <td data-label="${t("quota.thIn")}">${fmtCompact(d.prompt_tokens)}</td>
      <td data-label="${t("quota.thOut")}">${fmtCompact(d.completion_tokens)}</td>
      <td data-label="${t("quota.thTotal")}">${fmtCompact(d.total_tokens)}</td>
      <td data-label="${t("quota.thCost")}">${quotaPrice > 0 ? fmtVND(d.cost_vnd) : "—"}</td>
    </tr>`).join("");
    table.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }
  renderQuotaPager(quotaDays.length > (quotaPage + 1) * QUOTA_PAGE_SIZE || quotaPage > 0);
}

function renderQuotaPager(show) {
  const el = $("#quotaPager");
  if (!el) return;
  if (!show) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = pagerHtml(quotaPage, QUOTA_PAGE_SIZE, quotaDays.length > (quotaPage + 1) * QUOTA_PAGE_SIZE);
}

$("#quotaPager").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-pg]");
  if (!btn || btn.disabled) return;
  quotaPage = btn.dataset.pg === "next" ? quotaPage + 1 : Math.max(0, quotaPage - 1);
  renderQuotaTable();
});

// -- settings -----------------------------------------------------------------

const PRESETS = ["openai", "deepseek", "openai_compatible", "custom"];
let settingsCache = null;

async function loadSettings() {
  const s = (await api("/api/settings")).settings;
  settingsCache = s;
  renderSettings(s);
}

function renderSettings(s) {
  const ai = s.ai_provider;
  const state = s.config_state;
  const stateBadge = state === "pending"
    ? `<span class="status-pill pending">${t("set.pendingBadge")}</span>`
    : `<span class="status-pill ok">${t("set.appliedBadge")}</span>`;

  const provider = ai.provider;
  const radio = `<div class="radio-row">
    <label><input type="radio" name="provider" value="ollama" ${provider === "ollama" ? "checked" : ""}> ${t("set.ollama")}</label>
    <label><input type="radio" name="provider" value="api" ${provider === "api" ? "checked" : ""}> ${t("set.apiProv")}</label>
  </div>`;

  const presetRow = `<div class="field" id="presetField" style="${provider === "api" ? "" : "display:none"}">
    <label>${t("set.preset")}</label>
    <select id="preset">${PRESETS.map((p) => `<option value="${p}" ${ai.preset === p ? "selected" : ""}>${p}</option>`).join("")}</select>
  </div>`;

  const fields = `
    ${radio}
    ${presetRow}
    <div class="field"><label>${t("set.baseUrl")}</label><input type="text" id="baseUrl" value="${escapeHtml(ai.base_url || "")}" placeholder="${provider === "ollama" ? "e.g. http://localhost:11434" : "e.g. https://api.openai.com/v1"}" /></div>
    <div class="field"><label>${t("set.model")}</label><input type="text" id="model" value="${escapeHtml(ai.model || "")}" placeholder="qwen3:4b" /></div>
    <div class="field"><label>${t("set.analysisMode")}</label><select id="analysisMode">
      <option value="single" ${ai.analysis_mode === "multi" ? "" : "selected"}>${t("set.modeSingle")}</option>
      <option value="multi" ${ai.analysis_mode === "multi" ? "selected" : ""}>${t("set.modeMulti")}</option>
    </select></div>
    <div class="hint">${t("set.multiHint")}</div>
    <div class="row2">
      <div class="field"><label>${t("set.temperature")}</label><input type="number" step="0.1" min="0" max="5" id="temperature" value="${ai.temperature}" /></div>
      <div class="field"><label>${t("set.maxTokens")}</label><input type="number" step="1" min="1" id="maxTokens" value="${ai.max_tokens}" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>${t("set.timeout")}</label><input type="number" step="1" min="1" id="timeout" value="${ai.timeout}" /></div>
      <div class="field"><label>${t("set.apiKey")}</label><input type="password" id="apiKey" placeholder="${ai.api_key_configured ? t("set.configured") + ": " + (ai.api_key_masked || "••••") : t("set.apiKeyOptional")}" autocomplete="off" /></div>
    </div>
    <div class="hint">${t("set.aiHint")} ${state === "pending" ? t("set.aiPending") : ""}</div>
    <div class="row2">
      <button type="submit" class="btn">${t("set.saveAi")}</button>
      <button type="button" class="btn secondary" id="testConn">${t("set.testConn")}</button>
    </div>
    <div id="connResult">${stateBadge}</div>`;

  $("#aiForm").innerHTML = fields;

  $("#aiForm").querySelectorAll('input[name="provider"]').forEach((r) =>
    r.addEventListener("change", () => {
      const isApi = $("#aiForm").querySelector('input[name="provider"]:checked').value === "api";
      $("#presetField").style.display = isApi ? "" : "none";
    })
  );

  $("#aiForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    await saveAiProvider();
  });
  $("#testConn").addEventListener("click", testConnection);

  renderDemoForm(s.demo);
  renderTelegramForm(s.telegram);
  renderStrategyForm(s.strategy);
  renderSymbolForm(s.symbol);
  renderSymbolsForm(s.symbols);
  renderRuntime(s.runtime);
  renderAudit(s.audit);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function aiPayload() {
  const f = $("#aiForm");
  const provider = f.querySelector('input[name="provider"]:checked').value;
  const apiKey = $("#apiKey").value.trim();
  return {
    provider,
    preset: provider === "api" ? $("#preset").value : "ollama",
    base_url: $("#baseUrl").value.trim(),
    model: $("#model").value.trim(),
    temperature: Number($("#temperature").value),
    max_tokens: Number($("#maxTokens").value),
    timeout: Number($("#timeout").value),
    api_key: apiKey,
    analysis_mode: $("#analysisMode").value,
  };
}

async function saveAiProvider() {
  try {
    const payload = aiPayload();
    const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ai_provider: payload }) });
    toast(t("toast.aiSaved"));
    renderSettings(result.settings);
    refresh("settings");
  } catch (err) {
    toast(t("toast.aiSaveFail", { msg: err.message }), "error");
  }
}

async function testConnection() {
  const payload = aiPayload();
  const btn = $("#testConn");
  btn.disabled = true;
  btn.textContent = t("set.connTesting");
  try {
    const result = (await api("/api/settings/test-connection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })).result;
    const ms = result.latency_ms;
    const status = result.success
      ? `<span class="status-pill ok">OK · ${ms}ms · ${result.model_available === false ? t("set.connModelMissing") : t("set.connReachable")}</span>`
      : `<span class="status-pill bad">${t("set.connFailed")} (${ms}ms)</span>`;
    const models = result.available_models && result.available_models.length
      ? `<div class="hint">${t("set.connModels")}${result.available_models.join(", ")}</div>` : "";
    $("#connResult").innerHTML = `${status}${models}${result.error ? `<div class="hint" style="color:var(--red)">${escapeHtml(result.error)}</div>` : ""}`;
  } catch (err) {
    $("#connResult").innerHTML = `<span class="status-pill bad">${t("set.connFailed")}</span><div class="hint" style="color:var(--red)">${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = t("set.testConn");
  }
}

function renderDemoForm(demo) {
  const lockNote = demo.locked
    ? `<div class="lock-note">${t("set.demoLocked")}</div>` : "";
  const fields = `
    ${lockNote}
    <div class="field"><label>${t("set.dBalance")}</label><input type="number" step="0.01" id="dBalance" value="${demo.initial_balance}" ${demo.locked ? "disabled" : ""} /></div>
    <div class="row2">
      <div class="field"><label>${t("set.dMargin")}</label><input type="number" step="0.01" id="dMargin" value="${demo.margin_per_trade}" ${demo.locked ? "disabled" : ""} /></div>
      <div class="field"><label>${t("set.dLeverage")}</label><input type="number" step="1" min="1" id="dLeverage" value="${demo.leverage}" ${demo.locked ? "disabled" : ""} /></div>
    </div>
    <div class="row2">
      <div class="field"><label>${t("set.dRisk")}</label><input type="number" step="0.01" id="dRisk" value="${demo.risk_percent}" ${demo.locked ? "disabled" : ""} /></div>
      <div class="field"><label>${t("set.dFee")}</label><input type="number" step="0.0001" id="dFee" value="${demo.fee_rate}" ${demo.locked ? "disabled" : ""} /></div>
    </div>
    <div class="hint">${t("set.demoHint")}</div>
    <div class="row2">
      <button type="button" class="btn" id="saveDemo" ${demo.locked ? "disabled" : ""}>${t("set.saveDemo")}</button>
      <button type="button" class="btn danger" id="resetDemo" ${demo.locked ? "disabled" : ""}>${t("set.resetDemo")}</button>
    </div>`;
  $("#demoForm").innerHTML = fields;
  $("#saveDemo").addEventListener("click", async () => {
    try {
      const body = {
        initial_balance: $("#dBalance").value,
        margin_per_trade: $("#dMargin").value,
        leverage: Number($("#dLeverage").value),
        risk_percent: $("#dRisk").value,
        fee_rate: $("#dFee").value,
      };
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ demo: body }) });
      toast(t("toast.demoSaved"));
      renderSettings(result.settings);
    } catch (err) {
      toast(t("toast.demoSaveFail", { msg: err.message }), "error");
    }
  });
  $("#resetDemo").addEventListener("click", async () => {
    const ok = window.confirm(t("confirm.demoReset"));
    if (!ok) return;
    try {
      const result = await api("/api/demo/reset", { method: "POST" });
      toast(t("toast.demoReset"));
      renderSettings(result.settings);
      refresh("dashboard");
    } catch (err) {
      toast(t("toast.demoResetFail", { msg: err.message }), "error");
    }
  });
}

function renderTelegramForm(cfg) {
  const fields = `
    <div class="radio-row">
      <label><input type="checkbox" id="tgEnabled" ${cfg.enabled ? "checked" : ""} /> ${t("set.tgEnabled")}</label>
    </div>
    <div class="field"><label>${t("set.tgChatSignals")}</label><input type="text" id="tgChatSignals" value="" placeholder="e.g. -100111222333" />${cfg.chat_id_signals_configured ? `<span class="hint">${t("set.configured")}</span>` : ""}</div>
    <div class="field"><label>${t("set.tgChatReports")}</label><input type="text" id="tgChatReports" value="" placeholder="e.g. -100444555666" />${cfg.chat_id_reports_configured ? `<span class="hint">${t("set.configured")}</span>` : ""}</div>
    <div class="field"><label>${t("set.tgToken")}</label><input type="password" id="tgToken" autocomplete="off" placeholder="${cfg.token_configured ? t("set.configured") + ": " + (cfg.token_masked || "••••") : t("set.apiKeyOptional")}" /></div>
    <div class="hint">${t("set.tgHint")}</div>
    <div class="row2">
      <button type="button" class="btn" id="saveTelegram">${t("set.saveTg")}</button>
      <button type="button" class="btn secondary" id="testTelegramSignals">${t("set.testSignals")}</button>
      <button type="button" class="btn secondary" id="testTelegramReports">${t("set.testReports")}</button>
    </div>`;
  $("#telegramForm").innerHTML = fields;
  $("#saveTelegram").addEventListener("click", async () => {
    try {
      const body = {
        enabled: $("#tgEnabled").checked,
        chat_id_signals: $("#tgChatSignals").value.trim(),
        chat_id_reports: $("#tgChatReports").value.trim(),
        bot_token: $("#tgToken").value.trim(),
      };
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ telegram: body }) });
      toast(t("toast.tgSaved"));
      renderSettings(result.settings);
    } catch (err) {
      toast(t("toast.tgSaveFail", { msg: err.message }), "error");
    }
  });
  const sendTest = async (channel) => {
    try {
      const r = await api("/api/settings/test-telegram", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ channel }) });
      toast(r.result.success ? t("toast.tgTestOk", { channel }) : t("toast.tgTestFail", { msg: r.result.error }), r.result.success ? "" : "error");
    } catch (err) { toast(t("toast.tgTestFail", { msg: err.message }), "error"); }
  };
  $("#testTelegramSignals").addEventListener("click", () => sendTest("signals"));
  $("#testTelegramReports").addEventListener("click", () => sendTest("reports"));
}

function renderStrategyForm(st) {
  st = st || {};
  const num = (id, label, hint, attrs) => `
    <div class="field"><label>${label}</label><input type="number" id="${id}" value="${escapeHtml(st[id] ?? "")}" ${attrs || ""} /><span class="hint">${hint}</span></div>`;
  const fields = `
    <div class="row2">
      ${num("stConfidence", t("set.stConfidence"), t("set.stConfidenceHint"), 'min="0" max="100" step="1"')}
      ${num("stRiskReward", t("set.stRR"), t("set.stRRHint"), 'min="0" max="10" step="0.1"')}
    </div>
    <div class="row2">
      ${num("stFunding", t("set.stFunding"), t("set.stFundingHint"), 'min="0" max="0.05" step="0.0001"')}
      ${num("stFee", t("set.stFee"), t("set.stFeeHint"), 'min="0" max="0.01" step="0.0001"')}
    </div>
    <div class="row2">
      ${num("stExpiry", t("set.stExpiry"), t("set.stExpiryHint"), 'min="0" max="168" step="0.5"')}
      ${num("stWarn", t("set.stWarn"), t("set.stWarnHint"), 'min="0" max="168" step="1"')}
    </div>
    <div class="radio-row">
      <label><input type="checkbox" id="stHtf" ${Number(st.htf_bias) !== 0 ? "checked" : ""} /> ${t("set.stHtf")}</label>
    </div>
    <div class="hint">${t("set.stHint")}</div>
    <div class="row2">
      <button type="button" class="btn" id="saveStrategy">${t("set.saveStrategy")}</button>
      <button type="button" class="btn secondary" id="resetStrategy">${t("set.resetStrategy")}</button>
    </div>`;
  $("#strategyForm").innerHTML = fields;
  const collect = () => ({
    min_confidence: $("#stConfidence").value,
    min_risk_reward: $("#stRiskReward").value,
    max_funding_rate: $("#stFunding").value,
    fee_rate: $("#stFee").value,
    pending_expiry_hours: $("#stExpiry").value,
    warn_hours: $("#stWarn").value,
    htf_bias: $("#stHtf").checked ? 1 : 0,
  });
  $("#saveStrategy").addEventListener("click", async () => {
    try {
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ strategy: collect() }) });
      toast(t("toast.strategySaved"));
      renderSettings(result.settings);
    } catch (err) {
      toast(t("toast.strategySaveFail", { msg: err.message }), "error");
    }
  });
  $("#resetStrategy").addEventListener("click", async () => {
    if (!confirm(t("confirm.strategyReset"))) return;
    try {
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ strategy: { reset: true } }) });
      toast(t("toast.strategyReset"));
      renderSettings(result.settings);
    } catch (err) {
      toast(t("toast.strategyResetFail", { msg: err.message }), "error");
    }
  });
}

function renderSymbolForm(sym) {
  window._symbol = sym.symbol || window._symbol || "BTCUSDT";
  $("#sideSymbol").textContent = window._symbol;
  $("#priceCardTitle").textContent = t("dash.priceTitle", { symbol: window._symbol });
  const select = $("#symbolSelect");
  select.innerHTML = (sym.supported || ["BTCUSDT", "ETHUSDT", "XAUUSDT"]).map((s) => `<option value="${s}">${s}</option>`).join("");
  select.value = sym.symbol;
  $("#saveSymbol").addEventListener("click", async () => {
    try {
      const chosen = $("#symbolSelect").value;
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol: { symbol: chosen } }) });
      toast(t("toast.symbolSaved", { symbol: chosen }));
      renderSettings(result.settings);
      if (current === "dashboard") refresh("dashboard");
    } catch (err) {
      toast(t("toast.symbolSaveFail", { msg: err.message }), "error");
    }
  });

  $("#clearErrorBtn").addEventListener("click", async () => {
    if (!confirm(t("confirm.clearError"))) return;
    try {
      const result = await api("/api/runtime/clear-error", { method: "POST" });
      toast(t("toast.errorCleared"));
      const h = result.health || {};
      $("#systemHealth").innerHTML = renderHealth(h);
      if (current === "dashboard") refresh("dashboard");
    } catch (err) {
      toast(t("toast.clearFail", { msg: err.message }), "error");
    }
  });
}

function renderSymbolsForm(combo) {
  combo = combo || {};
  const enabled = combo.symbols || [];
  const supported = combo.supported || ["BTCUSDT", "ETHUSDT", "XAUUSDT"];
  const boxes = supported.map((s) => `
    <label><input type="checkbox" class="sym-check" value="${s}" ${enabled.includes(s) ? "checked" : ""} /> ${s}</label>`).join("");
  $("#symbolsForm").innerHTML = `
    <div class="field"><label>${t("set.symbolsTitle")}</label></div>
    <div class="radio-row">${boxes}</div>
    <div class="hint">${t("set.symbolsHint")}</div>
    <div class="row2">
      <button type="button" class="btn" id="saveSymbols">${t("set.saveSymbols")}</button>
    </div>`;
  $("#saveSymbols").addEventListener("click", async () => {
    try {
      const chosen = Array.from(document.querySelectorAll("#symbolsForm .sym-check:checked"))
        .map((c) => c.value);
      const result = await api("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbols: { symbols: chosen } }) });
      toast(t("toast.symbolsSaved"));
      renderSettings(result.settings);
    } catch (err) {
      toast(t("toast.symbolsSaveFail", { msg: err.message }), "error");
    }
  });
}
function renderRuntime(runtime) {
  const rows = `
    <div class="kv"><span>${t("set.rtSymbol")}</span><span>${runtime.symbol}</span></div>
    <div class="kv"><span>${t("set.rtTimeframe")}</span><span>${runtime.timeframe}</span></div>
    <div class="kv"><span>${t("set.rtSchedPoll")}</span><span>${runtime.scheduler_poll}s</span></div>
    <div class="kv"><span>${t("set.rtMonPoll")}</span><span>${runtime.monitor_poll}s</span></div>
    <div class="kv"><span>${t("set.rtDb")}</span><span class="${runtime.db_connected ? "" : "status-pill bad"}">${escapeHtml(runtime.db_path)}</span></div>`;
  $("#runtimeBox").innerHTML = `<div class="form" style="gap:2px">${rows}</div>`;
}

function renderAudit(audit) {
  const list = audit || [];
  if (!list.length) {
    $("#auditBox").innerHTML = emptyBox(t("set.auditEmpty"));
    return;
  }
  const head = `<tr><th>${t("set.auditThTime")}</th><th>${t("set.auditThNs")}</th><th>${t("set.auditThAction")}</th><th>${t("set.auditThSummary")}</th></tr>`;
  const body = list.map((a) => `<tr>
    <td data-label="${t("set.auditThTime")}">${fmtVn(a.created_at)}</td>
    <td data-label="${t("set.auditThNs")}">${escapeHtml(a.namespace)}</td>
    <td data-label="${t("set.auditThAction")}">${escapeHtml(a.action)}</td>
    <td data-label="${t("set.auditThSummary")}" class="audit-summary">${escapeHtml(a.summary)}</td>
  </tr>`).join("");
  $("#auditBox").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// -- macro events (banner + calendar tab) --------------------------------------

async function loadEventBanner() {
  const el = $("#eventBanner");
  if (!el) return;
  let data;
  try {
    data = await api("/api/events");
  } catch (_) {
    el.hidden = true;
    return;
  }
  const active = data && data.active;
  const upcoming = (data && data.upcoming) || [];
  const srcNote = data && data.source === "fallback"
    ? " (" + t("events.sourceFallback") + ")"
    : "";
  if (active) {
    el.hidden = false;
    el.className = "event-banner active";
    el.title = (active.title || "") + " — " + (active.event_utc || "") + srcNote;
    el.textContent = t("events.bannerActive", { title: active.title, until: active.blackout_end_vn });
    return;
  }
  const next = upcoming.find((e) => e && e.event_ms && e.event_ms > Date.now());
  if (next && next.event_ms - Date.now() <= 24 * 3600 * 1000) {
    const mins = Math.max(1, Math.round((next.event_ms - Date.now()) / 60000));
    const when = mins >= 90 ? t("events.bannerSoonH", { n: Math.round(mins / 60) }) : t("events.bannerSoonM", { n: mins });
    el.hidden = false;
    el.className = "event-banner upcoming";
    el.title = (next.title || "") + " — " + (next.event_utc || "") + srcNote;
    el.textContent = t("events.bannerUpcoming", { title: next.title, when, from: next.blackout_start_vn });
    return;
  }
  el.hidden = true;
}

$("#eventBanner").addEventListener("click", () => showPanel("events"));

let eventsMonth = null; // "YYYY-MM", null = current month
let eventsDays = {};

/* VN_MONTHS retired: month names now come from I18N month.* via MONTHS. */

// Short label for calendar chips: strip the trailing " Month YYYY" the feed
// appends and cap length so one long title can never stretch the grid
// (CSS ellipsis is the primary guard; this is belt-and-braces).
function shortEventName(name) {
  const s = String(name || "?").replace(/\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$/, "");
  return s.length > 22 ? s.slice(0, 21) + "…" : s;
}

function shiftMonth(key, delta) {
  const [y, m] = key.split("-").map(Number);
  const d = new Date(y, m - 1 + delta, 1);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}`;
}

async function loadEvents() {
  if (!eventsMonth) {
    const now = new Date();
    const p = (n) => String(n).padStart(2, "0");
    eventsMonth = `${now.getFullYear()}-${p(now.getMonth() + 1)}`;
  }
  const data = await api("/api/events/calendar?month=" + encodeURIComponent(eventsMonth));
  eventsMonth = data.month || eventsMonth;
  eventsDays = (data && data.days) || {};
  const srcEl = $("#calSource");
  if (srcEl) {
    const fallback = data && data.source === "fallback";
    srcEl.textContent = fallback ? t("events.sourceFallback") : t("events.sourceLive");
    srcEl.title = fallback ? t("events.sourceFallbackTitle") : t("events.sourceLiveTitle");
  }
  renderEventsCalendar();
}

function renderEventsCalendar() {
  const grid = $("#calGrid");
  if (!grid) return;
  const [y, m] = eventsMonth.split("-").map(Number);
  $("#calMonth").textContent = t("events.monthOf", { month: t(MONTHS[m - 1]), year: y });
  const first = new Date(y, m - 1, 1);
  const lead = (first.getDay() + 6) % 7; // Monday-first
  const dim = new Date(y, m, 0).getDate();
  const dimPrev = new Date(y, m - 1, 0).getDate();
  const today = new Date();
  const todayKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  const cells = [];
  for (let i = 0; i < lead; i++) {
    cells.push(`<span class="cal-day dim">${dimPrev - lead + 1 + i}</span>`);
  }
  for (let d = 1; d <= dim; d++) {
    const key = `${eventsMonth}-${String(d).padStart(2, "0")}`;
    const list = eventsDays[key] || [];
    const chips = list.slice(0, 2).map((e) => {
      const imp = String(e.impact || "").toLowerCase() === "high" ? "imp-high" : (String(e.impact || "").toLowerCase() === "medium" ? "imp-medium" : "imp-low");
      const mark = e.trading_paused ? "⏸ " : "";
      const paused = e.trading_paused ? " paused" : "";
      return `<span class="cal-chip ${imp}${paused}" title="${escapeHtml(String(e.title || e.name || "?"))}">${mark}${escapeHtml(shortEventName(e.name))}</span>`;
    }).join("");
    const more = list.length > 2 ? `<span class="cal-more">+${list.length - 2}</span>` : "";
    const dayPaused = list.some((e) => e.trading_paused);
    const cls = "cal-day" + (key === todayKey ? " today" : "") + (dayPaused ? " paused" : (list.length ? " has-event" : ""));
    cells.push(`<button type="button" class="${cls}" data-day="${key}"><span class="cal-num">${d}</span>${chips}${more}</button>`);
  }
  while (cells.length % 7 !== 0) {
    const n = cells.length - lead - dim + 1;
    cells.push(`<span class="cal-day dim">${n}</span>`);
  }
  grid.innerHTML = cells.join("");
}

$("#calGrid").addEventListener("click", (ev) => {
  const cell = ev.target.closest("[data-day]");
  if (!cell) return;
  openEventDialog(cell.dataset.day);
});

function openEventDialog(dayKey) {
  const dlg = $("#eventDialog");
  const list = eventsDays[dayKey] || [];
  const [y, m, d] = dayKey.split("-").map(Number);
  $("#eventDialogTitle").textContent = list.length
    ? t("events.dayTitle", { d, m, y, n: list.length })
    : t("events.dayTitleEmpty", { d, m, y });
  $("#eventDialogBody").innerHTML = list.length ? list.map((e) => {
    const imp = String(e.impact || "—").toUpperCase();
    const rows = [
      `<div class="kv"><span>${t("events.kTime")}</span><span>${escapeHtml(e.event_vn || (e.all_day ? t("events.allDay") : "—"))}</span></div>`,
      e.event_utc ? `<div class="kv"><span>${t("events.kUtc")}</span><span>${escapeHtml(e.event_utc)}</span></div>` : "",
      `<div class="kv"><span>${t("events.kImpact")}</span><span>${escapeHtml(imp)}</span></div>`,
      e.consensus ? `<div class="kv"><span>${t("events.kConsensus")}</span><span>${escapeHtml(String(e.consensus))}</span></div>` : "",
      e.prior ? `<div class="kv"><span>${t("events.kPrior")}</span><span>${escapeHtml(String(e.prior))}</span></div>` : "",
      e.actual ? `<div class="kv"><span>${t("events.kActual")}</span><span>${escapeHtml(String(e.actual))}</span></div>` : "",
      e.trading_paused ? `<div class="kv"><span>${t("events.kSignal")}</span><span>${t("events.aiPause", { from: e.blackout_start_vn || "", to: e.blackout_end_vn || "" })}</span></div>` : "",
      e.url ? `<div class="kv"><span>${t("events.kSource")}</span><span><a href="${escapeHtml(e.url)}" target="_blank" rel="noopener">financecalendar.com</a></span></div>` : "",
    ].join("");
    return `<div class="event-item"><h4>${e.trading_paused ? "⏸ " : ""}${escapeHtml(e.title || e.name || "?")}</h4><div class="form" style="gap:2px">${rows}</div></div>`;
  }).join("") : emptyBox(t("events.emptyDay"));
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
}

$("#eventDialogClose").addEventListener("click", () => {
  const dlg = $("#eventDialog");
  if (typeof dlg.close === "function") dlg.close();
  else dlg.removeAttribute("open");
});

$("#eventDialog").addEventListener("click", (ev) => {
  if (ev.target.id === "eventDialog") {
    const dlg = $("#eventDialog");
    if (typeof dlg.close === "function") dlg.close();
  }
});

function currentMonthKey() {
  const now = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${p(now.getMonth() + 1)}`;
}

$("#calPrev").addEventListener("click", () => { eventsMonth = shiftMonth(eventsMonth || currentMonthKey(), -1); loadEvents().catch((e) => toast(e.message, "error")); });
$("#calNext").addEventListener("click", () => { eventsMonth = shiftMonth(eventsMonth || currentMonthKey(), 1); loadEvents().catch((e) => toast(e.message, "error")); });
$("#calToday").addEventListener("click", () => { eventsMonth = null; loadEvents().catch((e) => toast(e.message, "error")); });

// -- refresh ------------------------------------------------------------------

function refresh(panel) {
  const jobs = [loadEventBanner().catch(() => {})];
  if (panel === "dashboard") {
    jobs.push(loadDashboard().catch((e) => toast(e.message, "error")));
    jobs.push(loadChart().catch(() => {}));
  }
  if (panel === "signals") jobs.push(loadSignals().catch((e) => toast(e.message, "error")));
  if (panel === "daemon") jobs.push(loadDaemonLog().catch((e) => toast(e.message, "error")));
  if (panel === "events") jobs.push(loadEvents().catch((e) => toast(e.message, "error")));
  if (panel === "trades") jobs.push(loadTrades().catch((e) => toast(e.message, "error")));
  if (panel === "statistics") {
    jobs.push(loadStatistics().catch((e) => toast(e.message, "error")));
    jobs.push(loadQuota().catch((e) => toast(e.message, "error")));
  }
  if (panel === "settings") jobs.push(loadSettings().catch((e) => toast(e.message, "error")));
  return Promise.allSettled(jobs);
}

{ const seq = ++loadSeq; showLoader(); refresh(current).then(() => { if (seq === loadSeq) hideLoader(); }); }

// Boot: restore persisted theme/language before the first paint of data.
try {
  const storedLang = localStorage.getItem("nextra.lang");
  if (storedLang === "en" || storedLang === "vi") lang = storedLang;
  const storedTheme = localStorage.getItem("nextra.theme");
  if (storedTheme === "light" || storedTheme === "dark") theme = storedTheme;
} catch (_) {}
applyTheme();
applyStaticI18n();
setInterval(() => { if (current === "dashboard") refresh("dashboard"); loadEventBanner().catch(() => {}); if (current === "events") loadEvents().catch(() => {}); }, 15000);

$("#chartInterval").addEventListener("change", () => { chartView = null; loadChart(); });

["sigDir", "sigStatus", "sigTime"].forEach((id) => {
  $("#" + id).addEventListener("change", () => { sigPage = 0; loadSignals().catch((e) => toast(e.message, "error")); });
});

$("#signalsTable").addEventListener("change", (ev) => {
  if (ev.target.classList && ev.target.classList.contains("sig-check")) updateSigDeleteCount();
});

$("#sigSelAll").addEventListener("click", () => {
  const boxes = document.querySelectorAll("#signalsTable .sig-check");
  const allChecked = boxes.length > 0 && Array.from(boxes).every((b) => b.checked);
  boxes.forEach((b) => { b.checked = !allChecked; });
  $("#sigSelAll").textContent = allChecked ? t("sig.deselectAll") : t("sig.selectAll");
  updateSigDeleteCount();
});

$("#sigDeleteSel").addEventListener("click", async () => {
  const ids = Array.from(document.querySelectorAll("#signalsTable .sig-check:checked"))
    .map((c) => Number(c.dataset.id));
  if (!ids.length) return;
  if (!confirm(t("sig.confirmDelSel", { n: ids.length }))) return;
  try { await deleteSignals({ ids }); } catch (e) { toast(e.message, "error"); }
});

$("#sigDeleteAll").addEventListener("click", async () => {
  if (!confirm(t("sig.confirmDelAll"))) return;
  try { await deleteSignals({ all: true }); } catch (e) { toast(e.message, "error"); }
});

["ddDir", "ddTime"].forEach((id) => {
  $("#" + id).addEventListener("change", () => { ddPage = 0; loadDaemonLog().catch((e) => toast(e.message, "error")); });
});

$("#ddClear").addEventListener("click", () => clearDaemonLog().catch((e) => toast(e.message, "error")));

document.addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-close-sid]");
  if (!btn) return;
  closePositionManual(btn.dataset.closeSid, btn).catch((err) => toast(t("pos.closeFail", { msg: err.message }), "error"));
});

async function closePositionManual(sid, btn) {
  const item = (lastPositions || []).find((x) => String((x.signal || {}).id) === String(sid));
  const s = (item && item.signal) || {};
  const u = (item && item.unrealized) || null;
  const pnl = u ? `${Number(u.gross_pnl) >= 0 ? "+" : ""}$${fmtNum(u.gross_pnl, 2)} (${fmtNum(u.pnl_percent, 2)}%)` : "—";
  const ok = window.confirm(t("pos.closeConfirm", { dir: s.direction || "", symbol: s.symbol || "", pnl }));
  if (!ok) return;
  btn.disabled = true;
  try {
    const result = await api(`/api/positions/${encodeURIComponent(sid)}/close`, { method: "POST" });
    const c = (result && result.close) || {};
    const trade = c.trade || null;
    const done = trade ? `${Number(trade.net_pnl) >= 0 ? "+" : ""}$${fmtNum(trade.net_pnl, 2)}` : "";
    toast(t("pos.closed", { status: c.status || "", pnl: done }));
    refresh("dashboard");
  } finally {
    btn.disabled = false;
  }
}

$("#daemonTable").addEventListener("click", (ev) => {  if (ev.target.closest(".dd-check")) return;
  const link = ev.target.closest(".sig-link");
  if (link) {
    showPanel("signals");
    return;
  }
  const row = ev.target.closest(".dd-row");
  if (!row) return;
  const detail = document.getElementById("dd-detail-" + row.dataset.row);
  if (detail) detail.hidden = !detail.hidden;
});

$("#daemonTable").addEventListener("change", (ev) => {
  if (ev.target.classList && ev.target.classList.contains("dd-check")) updateDdDeleteCount();
});

function ddSelectedCount() {
  return document.querySelectorAll("#daemonTable .dd-check:checked").length;
}

function updateDdDeleteCount() {
  const n = ddSelectedCount();
  const btn = $("#ddDeleteSel");
  if (!btn) return;
  btn.disabled = n === 0;
  btn.textContent = t("daemon.deleteSel", { n });
}

$("#ddSelAll").addEventListener("click", () => {
  const boxes = document.querySelectorAll("#daemonTable .dd-check");
  const allChecked = boxes.length > 0 && Array.from(boxes).every((b) => b.checked);
  boxes.forEach((b) => { b.checked = !allChecked; });
  $("#ddSelAll").textContent = allChecked ? t("daemon.deselectAll") : t("daemon.selectAll");
  updateDdDeleteCount();
});

$("#ddDeleteSel").addEventListener("click", async () => {
  const ids = Array.from(document.querySelectorAll("#daemonTable .dd-check:checked"))
    .map((c) => Number(c.dataset.id));
  if (!ids.length) return;
  if (!confirm(t("daemon.confirmDelSel", { n: ids.length }))) return;
  try {
    const res = await api("/api/daemon-log", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }) });
    toast(t("daemon.deleted", { n: res.deleted || 0 }), "ok");
    ddPage = 0;
    await loadDaemonLog().catch((e) => toast(e.message, "error"));
  } catch (e) { toast(e.message, "error"); }
});

let chartResizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(chartResizeTimer);
  chartResizeTimer = setTimeout(() => { if (current === "dashboard") loadChart().catch(() => {}); }, 300);
});